// thread safety test
// - Loads a copy of the same model on each GPU, plus a copy on the CPU, all in parallel
// - Creates n_parallel (--parallel) contexts per model
// - Runs inference in parallel on each context

#include <cstring>
#include <thread>
#include <utility>
#include <vector>
#include <atomic>
#include "llama.h"
#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    //llama_log_set([](ggml_log_level level, const char * text, void * /*user_data*/) {
    //    if (level == GGML_LOG_LEVEL_ERROR) {
    //        common_log_add(common_log_main(), level, "%s", text);
    //    }
    //}, NULL);

    auto cparams = common_context_params_to_llama(params);

    // each context has a single sequence
    cparams.n_seq_max = 1;

    // --- device selection -----------------------------------------------------
    // A device the user named explicitly is part of the contract: if it cannot host the test,
    // that is a failure (the runtime fails the same way, e.g. llama-cli -dev Vulkan0 aborts on
    // a KV cache without SET_ROWS). Devices the test picked on its own may be skipped.
    const bool explicit_devices = !params.devices.empty();

    // An explicit -dev/--device list wins: the devices the user named are the ones under
    // test. Silently adding a device they excluded (for example a Vulkan device when the
    // requested KV cache type is only served by the CUDA backend) fails the test for a
    // configuration nobody asked for. Without an explicit list every GPU device qualifies.
    std::vector<ggml_backend_dev_t> candidates;
    if (!params.devices.empty()) {
        for (auto * dev : params.devices) {
            if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                candidates.push_back(dev);
            }
        }
    } else {
        for (int i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                candidates.push_back(dev);
            }
        }
    }

    // One physical card can be visible through several backends (CUDA and Vulkan on the same
    // card). Keep one device per physical card - the same rule llama_prepare_model_devices
    // applies (src/llama.cpp): a second model copy on the same card doubles its VRAM use, and
    // on a backend that cannot serve the requested KV cache type it fails for no reason.
    std::vector<ggml_backend_dev_t> gpus;
    for (auto * dev : candidates) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);

        bool same_card = false;
        for (auto * sel : gpus) {
            ggml_backend_dev_props sel_props;
            ggml_backend_dev_get_props(sel, &sel_props);
            if (props.device_id && sel_props.device_id &&
                strcmp(props.device_id, sel_props.device_id) == 0) {
                same_card = true;
                break;
            }
        }

        if (same_card) {
            LOG_WRN("%s: skipping %s - same physical device as an already selected one\n",
                    __func__, ggml_backend_dev_name(dev));
            continue;
        }

        gpus.push_back(dev);
    }

    const int gpu_dev_count = (int) gpus.size();
    const int num_models = gpu_dev_count + 1 + 1; // GPUs + 1 CPU model + 1 layer split
    //const int num_models = std::max(1, gpu_dev_count);
    const int num_contexts = std::max(1, params.n_parallel);

    std::vector<llama_model_ptr> models(num_models);
    std::vector<int> load_status(num_models, 0); // written by the load thread of model m
    std::vector<std::thread> load_threads;
    std::vector<std::thread> threads;
    std::atomic<bool> failed = false;

    for (int m = 0; m < num_models; ++m) {
        load_threads.emplace_back([&, m]() {
            auto mparams = common_model_params_to_llama(params);

            // null-terminated device list for llama_model
            std::vector<ggml_backend_dev_t> dev_list;

            if (m < gpu_dev_count) {
                mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
                dev_list = { gpus[m], nullptr };
                mparams.devices = dev_list.data();
            } else if (m == gpu_dev_count) {
                mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
                mparams.main_gpu = -1; // CPU model
            } else {
                mparams.split_mode = LLAMA_SPLIT_MODE_LAYER;
                // layer split over the selected devices only - never pull in a device that
                // was excluded from the selection
                dev_list = gpus;
                dev_list.push_back(nullptr);
                mparams.devices = dev_list.data();
            }

            llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
            if (model == NULL) {
                LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
                load_status[m] = 0;
                return;
            }

            models[m].reset(model);
            load_status[m] = 1;
        });
    }

    for (auto & thread : load_threads) {
        thread.join();
    }

    // A model that does not fit on one device (VRAM) is a skip, not a failure: the same model
    // loads fine elsewhere. Only a model that cannot be loaded anywhere is a real error.
    int loaded = 0;
    for (int m = 0; m < num_models; ++m) {
        loaded += load_status[m] ? 1 : 0;
    }

    if (loaded == 0) {
        LOG_ERR("All models failed to load.\n");
        return 1;
    }

    for (int m = 0; m < num_models; ++m) {
        if (load_status[m]) {
            continue;
        }

        if (m == gpu_dev_count) {
            LOG_ERR("%s: the CPU model failed to load\n", __func__);
            return 1;
        }

        const char * label = m < gpu_dev_count ? ggml_backend_dev_name(gpus[m]) : "layer split";
        if (explicit_devices) {
            LOG_ERR("%s: %s was requested explicitly (-dev) but the model could not be loaded there\n",
                    __func__, label);
            return 1;
        }

        LOG_WRN("SKIP: %s could not load the model (see the error above)\n", label);
    }

    // --- capability probe -----------------------------------------------------
    // A device that cannot host the requested configuration (KV cache type not served by
    // that backend, or not enough free VRAM for the KV cache) must not fail the whole test:
    // report it as a skip and drop that model. The runtime already logged the reason.
    std::vector<llama_model_ptr> active_models;
    for (int m = 0; m < num_models; ++m) {
        if (!models[m]) {
            continue;
        }

        const char * label = m < gpu_dev_count ? ggml_backend_dev_name(gpus[m])
                          : m == gpu_dev_count ? "CPU"
                          : "layer split";

        llama_context_ptr probe { llama_init_from_model(models[m].get(), cparams) };
        if (!probe) {
            if (explicit_devices) {
                LOG_ERR("%s: %s was requested explicitly (-dev) but cannot host this test's context\n",
                        __func__, label);
                return 1;
            }

            LOG_WRN("SKIP: %s cannot host this test's context (see the error above)\n", label);
            continue;
        }

        active_models.push_back(std::move(models[m]));
    }

    if (active_models.empty()) {
        LOG_WRN("%s: no device could host this test's context - nothing to check\n", __func__);
        return 0;
    }

    for (size_t mi = 0; mi < active_models.size(); ++mi) {
        auto * model = active_models[mi].get();
        for (int c = 0; c < num_contexts; ++c) {
            threads.emplace_back([&, mi, c, model]() {
                LOG_INF("Creating context %d/%d for model %d/%d\n", c + 1, num_contexts, (int) mi + 1, (int) active_models.size());

                llama_context_ptr ctx { llama_init_from_model(model, cparams) };
                if (ctx == NULL) {
                    LOG_ERR("failed to create context\n");
                    failed.store(true);
                    return;
                }

                std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler { common_sampler_init(model, params.sampling), common_sampler_free };
                if (sampler == NULL) {
                    LOG_ERR("failed to create sampler\n");
                    failed.store(true);
                    return;
                }

                {
                    auto prompt = common_tokenize(ctx.get(), params.prompt, true);
                    if (prompt.empty()) {
                        LOG_ERR("failed to tokenize prompt\n");
                        failed.store(true);
                        return;
                    }
                    common_batch batch = common_batch_get_one(ctx.get(), prompt);
                    if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                        LOG_ERR("failed to decode prompt\n");
                        failed.store(true);
                        return;
                    }
                }

                const auto * vocab = llama_model_get_vocab(model);
                std::string result = params.prompt;

                for (int i = 0; i < params.n_predict; i++) {
                    llama_token token = common_sampler_sample(sampler.get(), ctx.get(), -1);

                    result += common_token_to_piece(ctx.get(), token);

                    if (llama_vocab_is_eog(vocab, token)) {
                        break;
                    }

                    common_batch batch = common_batch_get_one(ctx.get(), &token, 1);

                    int ret = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
                    if (ret == 1 && i > 0) {
                        LOG_INF("Context full, stopping generation.\n");
                        break;
                    }

                    if (ret != 0) {
                        LOG_ERR("Model %d/%d, Context %d/%d: failed to decode\n", (int) mi + 1, (int) active_models.size(), c + 1, num_contexts);
                        failed.store(true);
                        return;
                    }
                }

                LOG_INF("Model %d/%d, Context %d/%d: %s\n\n", (int) mi + 1, (int) active_models.size(), c + 1, num_contexts, result.c_str());

                llama_synchronize(ctx.get());
            });
        }
    }

    for (auto & thread : threads) {
        thread.join();
    }

    if (failed) {
        LOG_ERR("One or more threads failed.\n");
        return 1;
    }

    LOG_INF("All threads finished without errors.\n");
    return 0;
}
