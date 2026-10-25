llama_add_compile_flags()

# Optional args:
# - LINK: libraries to link against (defaults to llama-common, which also brings llama)
function(llama_build source)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs LINK)
    cmake_parse_arguments(LLAMA_BUILD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_BUILD_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_BUILD_LINK)
        set(LLAMA_BUILD_LINK llama-common)
    endif()

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_BUILD_LINK})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
# - LINK: libraries to link against (defaults to llama-common, which also brings llama)
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS LINK)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_TEST_LINK)
        set(LLAMA_TEST_LINK llama-common)
    endif()

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_TEST_LINK})

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge        LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm    LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon          LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4         LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf --check-decode)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2           LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt             LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3           LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2           LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35          LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact          LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        LABEL behavior
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp  LABEL "main behavior"  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

# PORT-DEVIATION (C4): upstream consolidated its state tests into test-llama-context
# (a17e94d22) and deleted test-recurrent-state-rollback.cpp. The fork's file is not part of
# that consolidation: it holds VBR / SWA-prefix / multi-sequence-replay cases that upstream
# has no equivalent for, so it stays as its own target. Moving 1251 lines into
# test-llama-context.cpp would buy nothing and would make every future merge of that file
# contend with fork code.
llama_build(test-recurrent-state-rollback.cpp)

# llama_context state tests: KV and recurrent state save, load, copy, rollback
set(LLAMA_CONTEXT_TEST_SRC
    test-llama-context.cpp
    llama-context/cases-kv.cpp
    llama-context/cases-rs.cpp
    llama-context/test-case.cpp
    llama-context/test-log.cpp
    llama-context/test-runner.cpp
    llama-context/test-util.cpp
)
llama_build(${LLAMA_CONTEXT_TEST_SRC})

# fork 9l-mtp-adaptive: the MTP draft-depth controller is a header-only class in common/,
# so this needs neither a model nor a device -- it runs on every platform, including the
# Windows shared build where libllama's internals are not exported.
llama_build_and_test(test-speculative-mtp-adaptive.cpp)

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-unicode.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-sampling.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-reasoning-budget.cpp  LABEL behavior)
    llama_build_and_test(test-grammar-parser.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-grammar-integration.cpp  LABEL behavior)
    llama_build_and_test(test-llama-grammar.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-batch-alloc.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-chat.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    llama_build_and_test(test-server-component.cpp  LABEL behavior)
    target_include_directories(test-server-component PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-component PRIVATE server-context)
    llama_build_and_test(test-json-schema-to-grammar.cpp  LABEL behavior)

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp LINK llama)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila     LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon     LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2      LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox   LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt        LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact     LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    # TODO: after `test-llama-archs` is refactored to not link internal symbols, we can move all related tests out of
    #       this branch and run them on all platforms
    llama_build(test-llama-archs.cpp)

    set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
    file(MAKE_DIRECTORY "${MODEL_DIR}")

    llama_test(
        test-llama-archs
        NAME test-generate-models
        LABEL "main behavior backend"
        ARGS -o "${MODEL_DIR}"
    )
    set_tests_properties(test-generate-models PROPERTIES
        FIXTURES_SETUP generate-models
    )

    # Test context state handling across all architectures, using the generated dummy models
    llama_test(
        test-llama-context
        LABEL "main behavior backend"
        ARGS --models "${MODEL_DIR}"
    )
    # abort on graph reallocations that the initial reserve should have covered
    set_tests_properties(test-llama-context PROPERTIES
        FIXTURES_REQUIRED generate-models
        ENVIRONMENT "GGML_SCHED_DEBUG_REALLOC=1"
    )

    # Test recurrent-state rollback across all architectures, using the generated dummy models
    # (fork cases: multi-seq replay, previous-snapshot restore, VBR/SWA prefix sharing).
    llama_test(
        test-recurrent-state-rollback
        LABEL main
        ARGS --models "${MODEL_DIR}"
    )
    # abort on graph reallocations that the initial reserve should have covered
    set_tests_properties(test-recurrent-state-rollback PROPERTIES
        FIXTURES_REQUIRED generate-models
        ENVIRONMENT "GGML_SCHED_DEBUG_REALLOC=1"
    )

    llama_build(test-fusion.cpp)
endif()

# LoRA execution-content digest contract (slice 7d precondition).
#
# Lives outside the block above even though it loads a model and reads internal
# headers (llama_model::get_tensor, llm_arch_name): those are reachable from a shared
# build because the llama target exports all symbols on Windows. The adapters are
# synthesized at run time, so the test needs a base model with an untied output
# projection; point LLAMA_TEST_LORA_MODEL at one to register it with ctest, otherwise
# the executable is only built and stays available for a manual run.
llama_build(test-lora-adapter-digest.cpp)
set(LORA_ADAPTER_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-lora-adapters/")
file(MAKE_DIRECTORY "${LORA_ADAPTER_DIR}")

set(LLAMA_TEST_LORA_MODEL "" CACHE FILEPATH
    "Base GGUF used by test-lora-adapter-digest; the test is not registered when empty")

if (LLAMA_TEST_LORA_MODEL)
    llama_test(
        test-lora-adapter-digest
        NAME test-lora-adapter-digest
        LABEL main
        ARGS "${LLAMA_TEST_LORA_MODEL}" "${LORA_ADAPTER_DIR}"
    )
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp  LABEL behavior)
llama_build_and_test(test-jinja.cpp  LABEL behavior)
llama_test(test-jinja  NAME test-jinja-py  ARGS -py  LABEL "python behavior")
llama_build_and_test(test-json-schema.cpp  LABEL behavior)
llama_build_and_test(test-chat-auto-parser.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp  LABEL behavior)
# debug tool for chat template differential analysis (not registered as a test, run it manually)
llama_build(test-chat-analysis.cpp)
llama_build_and_test(test-log.cpp  LABEL behavior)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
    LABEL behavior
)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${PROJECT_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

# the context state cases against a real model, the sweep above uses the generated dummy ones
llama_test(test-llama-context  NAME test-llama-context-model  LABEL "model behavior backend"  ARGS -m "${MODEL_DEST}")
set_tests_properties(test-llama-context-model PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-thread-safety.cpp  LABEL "main backend"  ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp  LABEL behavior)

# fork 9l-d4b9r: test-llama-archs hosts the device-free Qwen4 VBR memory-accounting contract
# (see test_qwen4_vbr_memory_accounting in that file).  Upstream registers the target inside
# `if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)` because those tests use internal types that are
# not exported with LLAMA_API under a shared-library Windows build -- but this tree sets
# WINDOWS_EXPORT_ALL_SYMBOLS on the llama target (src/CMakeLists.txt), so the internal types
# are linkable here.  Build the executable without registering a ctest: upstream and the fork
# both reach this target only through a manual `-a qwen4exp` invocation, and registering the
# default (all-architectures) run would make `ctest` far slower for no coverage gain.
# Guarded so the `if`-block above (which already creates the target on other configs) wins.
if (NOT TARGET test-llama-archs)
    llama_build(test-llama-archs.cpp)
    # fork 9l-d4b9s: the four `assert(...)` statements this file now carries -- the VBR admission
    # predicates in test_backends -- are plain <cassert>, not GGML_ASSERT.  Release defines NDEBUG,
    # so without /UNDEBUG they compile to nothing and the slice would ship four dead lines.  This
    # was not a guess: flipping one of them to a deliberately false condition still exited 0, which
    # is how the slice found out.  The flag is the convention this file's neighbours already use
    # (see the test-vbr-policy / test-vbr-downward groups below).  The fork itself does not pass it
    # here, so in the fork's Release build the same four lines are inert.
    if (MSVC)
        target_compile_options(test-llama-archs PRIVATE /UNDEBUG)
    else()
        target_compile_options(test-llama-archs PRIVATE -UNDEBUG)
    endif()
endif()

# Pure unit test of tools/llama-bench/llama-bench-vbr.h (the header this tree landed
# in slice 8b). It includes that header by relative path and does `#undef NDEBUG`
# before <cassert>, so its asserts are live without needing /UNDEBUG -- which is
# also why the fork passes /UNDEBUG to its siblings but not to this one.
llama_build_and_test(test-llama-bench-vbr.cpp)

llama_build_and_test(test-kv-cells.cpp)
target_include_directories(test-kv-cells PRIVATE ${PROJECT_SOURCE_DIR}/src)

llama_build_and_test(test-kv-cell-position-index.cpp)

# Cache-control wire/golden-path test. Eleven cache module .cpp files are compiled
# in (all byte-identical to the fork), and the golden JSON it compares against is
# read from the source tree through CACHE_CONTROL_GOLDEN_PATH.
# The fork registers this inside `if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)`; here it
# must sit outside that block or the target is never created on Windows + shared.
llama_build_and_test(
    test-cache-control.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-control.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-control-wire.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-retention-proof.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-vbr-proof.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-vbr-artifact-store.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-prompt-cache-payload.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-retention-sidecar.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-destruction-quote.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-plan-authority.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-yield.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-lease.cpp)
target_include_directories(test-cache-control PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/tools/server)
target_compile_definitions(test-cache-control PRIVATE
    CACHE_CONTROL_GOLDEN_PATH="${PROJECT_SOURCE_DIR}/tests/fixtures/cache-control-v1.json")

# Backend ABI gates. These need ggml internals only and no model, so they are
# registered outside the model-gated block (the fork keeps its siblings inside,
# but this tree must build them on Windows + BUILD_SHARED_LIBS=ON).
# Three sibling candidates from the same fork family were rejected because their
# target code is not in this tree: test-mul-mat-params (GGML_MMID_WINDOW_N_LOCAL,
# ggml_mmid_expert_index, GGML_HINT_FORCE_MMQ, ggml_flash_attn_ext_*_sparse_mask),
# test-expert-window-params (ggml_mul_mat_id_set_expert_window) and
# test-moe-metadata (llama_model_loader's ctor has no 13-argument overload here).
llama_build_and_test(test-backend-expert-transfer.cpp)
target_include_directories(test-backend-expert-transfer PRIVATE ../ggml/src)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-backend-meta-replay.cpp)
    set_tests_properties(test-backend-meta-replay PROPERTIES SKIP_RETURN_CODE 77)
endif()

# The kv-stream planner/config/softmax helpers are internal to libllama, so their
# implementation files are compiled directly into the tests. That keeps the tests
# working on Windows shared builds, where only LLAMA_API symbols are exported.
llama_build_and_test(
    test-kv-stream-plan.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-kv-stream-plan.cpp)
target_include_directories(test-kv-stream-plan PRIVATE ${PROJECT_SOURCE_DIR}/src)
llama_build_and_test(
    test-kv-stream-config.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-kv-stream-config.cpp)
target_include_directories(test-kv-stream-config PRIVATE ${PROJECT_SOURCE_DIR}/src)
llama_build_and_test(
    test-kv-stream-softmax.cpp
    ${PROJECT_SOURCE_DIR}/src/llama-kv-stream-softmax.cpp)
target_include_directories(test-kv-stream-softmax PRIVATE ${PROJECT_SOURCE_DIR}/src)
llama_build_and_test(
    test-kv-stream-bench-config.cpp
    ${PROJECT_SOURCE_DIR}/tools/kv-stream-bench/kv-stream-bench-config.cpp)
target_include_directories(test-kv-stream-bench-config PRIVATE ${PROJECT_SOURCE_DIR}/tools/kv-stream-bench)
# These three call the CUDA backend's C API directly (ggml_backend_cuda_init and the
# ggml_backend_cuda_kv_stream_* / ggml_backend_cuda_phase_arena_* entry points), so they can
# only link when ggml-cuda is a regular library that `ggml` links PUBLICly. With
# GGML_BACKEND_DL=ON the backends are MODULE libraries loaded at runtime and those symbols are
# deliberately not exported, so building them is not possible - skip instead of failing the link.
if (GGML_CUDA AND NOT GGML_BACKEND_DL)
    llama_build_and_test(test-kv-stream-cuda-buffer.cpp LABEL cuda)
    llama_build_and_test(test-kv-stream-cuda-attn.cpp LABEL cuda)
    llama_build_and_test(test-kv-stream-cuda-set-rows.cpp LABEL cuda)
    # CUDA-vs-CPU parity of the TurboQuant KV encoders (SET_ROWS). Needs no model file.
    llama_build_and_test(test-turbo-cuda-parity.cpp LABEL cuda)
    # Differential test of the opt-in TurboQuant MMA FlashAttention path (GGML_CUDA_FA_MMA=1).
    # Run twice on the device: default = mma mode (counter must advance), "fallback" = env off
    # (counter must stay 0). See MERGE_BUUN_PLAN.md section 10.9.
    llama_build_and_test(test-mma-fa.cpp LABEL cuda)
    # fork 9l-d4b9j: the VBR CUDA VMM test (fork tests/test-vbr-vmm.cpp:1-402), landed
    # VERBATIM. It drives the CUDA VBR backend interface directly -- ggml_backend_cuda_vbr_iface()
    # then vmm_pool_{init,map,unmap,mapped,mapped_in_range,residency_epoch,clear} + kv_transcode --
    # so it needs ggml-cuda linked as a regular library, exactly like the kv-stream/turbo tests
    # above. This tree's ggml-vbr.h (the iface struct) is byte-identical to the fork's, and
    # ggml_backend_cuda_vbr_iface() is defined in ggml/src/ggml-cuda/ggml-cuda.cu, so the file
    # compiles unchanged. On a host with no VMM-capable device main() prints
    # "SKIP: GPU device N has no VMM support" and returns 0, i.e. a GPU-less run is a PASS.
    llama_build_and_test(test-vbr-vmm.cpp LABEL cuda)
endif()
llama_build_and_test(test-model-resolution.cpp  LABEL behavior)
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp  LABEL "main behavior backend"  LINK ggml)
endif()
if (GGML_METAL AND NOT GGML_BACKEND_DL)
    # calls the Metal backend's graph optimizer and fusion table directly
    llama_build_and_test(test-metal-graph-optimize.cpp  LABEL "main behavior backend"  LINK ggml ggml-metal)
    target_include_directories(test-metal-graph-optimize PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-metal)
endif()

llama_build(test-backend-ops.cpp LINK ggml)

llama_build_and_test(test-tiled-mulmat.cpp  LABEL "main behavior backend"  LINK ggml)
llama_build_and_test(test-model-load-cancel.cpp LABEL "model behavior")
llama_build_and_test(test-autorelease.cpp       LABEL "model behavior")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model behavior backend")

if (APPLE)
    llama_build(test-rset-release.cpp)
endif()

if (NOT GGML_BACKEND_DL AND GGML_CPU)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp  LABEL "main behavior"  LINK ggml)
    llama_build_and_test(test-quantize-fns.cpp  LABEL "main behavior"  LINK ggml)
    llama_build_and_test(test-ptq1_0-element-map.cpp LINK ggml)
    llama_build_and_test(test-ptq1_0-cuda-dot.cpp LINK ggml)
    llama_build_and_test(test-pq2-row-shapes.cpp LINK ggml)
    llama_build_and_test(test-quantize-perf.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-rope.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-col2im-1d.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-conv-1d.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-gelu-quick.cpp  LABEL "main behavior backend"  LINK ggml)

    # TurboQuant CPU round-trip test (standalone, links ggml only)
    add_executable(test-turbo-quant test-turbo-quant.c)
    target_link_libraries(test-turbo-quant PRIVATE ggml)
    # needs ggml.h for ggml_get_type_traits() / ggml_type_traits (registration check) and
    # ggml/src for ggml-turbo-tcq-tables.h (the TCQ codebooks the library compiles in)
    target_include_directories(test-turbo-quant PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)
    add_test(NAME test-turbo-quant COMMAND test-turbo-quant)

    # End-to-end CPU-backend flash attention with a TurboQuant TCQ KV cache.
    # Links ggml only (no llama), and uses the CPU backend, so it needs no GPU.
    add_executable(test-turbo-tcq-cpu-fa test-turbo-tcq-cpu-fa.cpp)
    target_link_libraries(test-turbo-tcq-cpu-fa PRIVATE ggml)
    add_test(NAME test-turbo-tcq-cpu-fa COMMAND test-turbo-tcq-cpu-fa)
endif()

# libmtmd
llama_build_and_test(test-mtmd-c-api.c  LABEL behavior  LINK mtmd)

llama_build_and_test(test-mtmd-impl.cpp  LABEL behavior  LINK mtmd)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data  LABEL "model behavior")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp  LABEL "model behavior"  LINK llama)
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# VBR pure-arithmetic kernels: header-only, no dependency beyond the standard library, so they
# build and run in every platform/configuration (unlike the tests above, which need internals that
# are not exported). Their whole body is asserts, and Release defines NDEBUG, so /UNDEBUG is the
# flag that keeps them from silently becoming no-ops.
llama_build_and_test(test-vbr-policy.cpp)
llama_build_and_test(test-vbr-physical.cpp)
llama_build_and_test(test-vbr-transaction.cpp)
if (MSVC)
    target_compile_options(test-vbr-policy      PRIVATE /UNDEBUG)
    target_compile_options(test-vbr-physical    PRIVATE /UNDEBUG)
    target_compile_options(test-vbr-transaction PRIVATE /UNDEBUG)
else()
    target_compile_options(test-vbr-policy      PRIVATE -UNDEBUG)
    target_compile_options(test-vbr-physical    PRIVATE -UNDEBUG)
    target_compile_options(test-vbr-transaction PRIVATE -UNDEBUG)
endif()

# The DFlash depth policy is a header-only controller (common/speculative-dflash-adaptive.h)
# whose whole body is asserts, so it needs the same /UNDEBUG treatment. It has no model
# dependency: the test drives it with synthetic cycle costs.
llama_build_and_test(test-dflash-adaptive.cpp)
if (MSVC)
    target_compile_options(test-dflash-adaptive PRIVATE /UNDEBUG)
else()
    target_compile_options(test-dflash-adaptive PRIVATE -UNDEBUG)
endif()

# The repack cache is a Linux-flock design: the whole test body sits under
# `#if defined(__linux__)`, so anywhere else it returns 77. That is what SKIP_RETURN_CODE is
# for, and the fork registers it unconditionally for exactly that reason. It still earns its
# place here as the link-level consumer that keeps src/llama-repack-cache.cpp from being dead
# code we cannot observe.
llama_build_and_test(test-repack-cache.cpp)
target_include_directories(test-repack-cache PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/vendor)
set_tests_properties(test-repack-cache PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)

# VBR leaf kernels.

# Self-reporting rather than assert-based: it returns a status from main and prints explicit PASS
# lines, so unlike the three above it does not need /UNDEBUG to stay meaningful - which is also why
# upstream only attaches that flag to the assert-based ones. It includes its header by bare name,
# hence the src include directory.
llama_build_and_test(test-vbr-hard-seal.cpp)
target_include_directories(test-vbr-hard-seal PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Identity digests for VBR, the next leaf slice. Same shape as hard-seal: self-reporting (a status
# from main plus explicit PASS markers), so no /UNDEBUG is needed. It includes its header by bare
# name, and that header reaches llama.h through llama-vbr-checkpoint-types.h, hence the src include
# directory on top of what linking llama already propagates.
llama_build_and_test(test-vbr-identity-digest.cpp)
target_include_directories(test-vbr-identity-digest PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Downward/upward representation algebra: the fork's test-vbr-downward.cpp, ported whole (fork
# 9l-d4b9n). The five consumer-side cases withheld at stage 2.6d (they need llama-memory.h's preflight
# types and llama-kv-cache.h's stash helpers) now run -- the consumer side landed, so the reason for
# the deferral died (§58.1) and the live file is byte-identical to the fork's again. Assert-based, so
# it gets /UNDEBUG for the reason the 2.6a group does. It includes its header by bare name, hence the
# src include directory.
llama_build_and_test(test-vbr-downward.cpp)
target_include_directories(test-vbr-downward PRIVATE ${PROJECT_SOURCE_DIR}/src)
if (MSVC)
    target_compile_options(test-vbr-downward PRIVATE /UNDEBUG)
else()
    target_compile_options(test-vbr-downward PRIVATE -UNDEBUG)
endif()

# Operation registry. Self-reporting, so no /UNDEBUG. It needs the src include directory because it
# includes llama-vbr-operation.h and llama-cparams.h by bare name.
llama_build_and_test(test-vbr-operation.cpp)
target_include_directories(test-vbr-operation PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Committed-extent store and dual-view ownership index. Self-reporting, so no /UNDEBUG. The
# ownership-index half drives rank_below()/enumerate_owned() against a plain map oracle rather than
# recomputing the Fenwick tree, so an off-by-one in the index arithmetic cannot hide.
llama_build_and_test(test-vbr-extent.cpp)
target_include_directories(test-vbr-extent PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Pinned transport ring. Self-reporting, so no /UNDEBUG. Its backend-dependent pump cannot run
# without ggml, so this test covers the creation contract only: geometry, accounting binding, lane
# binding, the process-wide capacity ceiling and move-only token ownership. The failure codes that
# need a real device are listed in MERGE_BUUN_PLAN §10.33.
llama_build_and_test(test-vbr-pinned-ring.cpp)
target_include_directories(test-vbr-pinned-ring PRIVATE ${PROJECT_SOURCE_DIR}/src)

# The artifact codec / catalog / validator / staging test: the fork's
# tests/test-vbr-artifact.cpp, ported whole (fork 9l-d4b9m). Self-reporting (a failure counter and
# an explicit PASS line), so no /UNDEBUG. This is the full prepare/encode/decode round trip that
# MERGE_BUUN_PLAN §10.38 recorded as the outstanding gap.
#
# The fork splits this file in two with `#ifdef VBR_PROMPT_CACHE_PUBLICATION_TEST`. That half
# reaches the live server (it includes server-context.h / server-task.h and the server module adds
# `server-context`, `mtmd` and tools/mtmd to the target). The macro is NOT defined here, so exactly
# the device-free half compiles -- and the three tools/server modules below are what it links
# against (each byte-identical to the fork's).
llama_build_and_test(
    test-vbr-artifact.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-lease.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-vbr-proof.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-prompt-cache-payload.cpp)
target_include_directories(test-vbr-artifact PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/tools/server)
# The fork's scale arm: the same binary in its --capture-projection-bench mode (it builds a
# million-cell projection union and reports the elapsed time, ~90 ms here).
llama_test_cmd(test-vbr-artifact
    NAME test-vbr-capture-projection-scale
    LABEL "vbr-cache;slow"
    ARGS --capture-projection-bench)

# Format vocabulary the fork's own test does NOT cover: the wire-version gates, the token-block
# codec version, and the portable-domain ordering pinned against a std::tuple oracle. Port-only
# (upstream has no test for llama-vbr-artifact.h at all), and kept after the fork's file landed
# because the fork never asserts any of it. Self-reporting, so no /UNDEBUG.
llama_build_and_test(test-vbr-artifact-vocabulary.cpp)
target_include_directories(test-vbr-artifact-vocabulary PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Segment chain and Merkle range proofs: the fork's tests/test-vbr-artifact-capture.cpp, ported whole
# (fork 9l-d4b9o). Unlike the model-backed fork tests this one has no macro gate -- its only GPU case
# (test_cuda_ring) is dispatched under `argv[1] == "--cuda"`, which ctest never passes, so the default
# run is entirely device-free. Two tools/server modules are what it links against (both byte-identical
# to the fork's); ggml/include/ggml-turbo-meansub.h is on the include path already.
llama_build_and_test(
    test-vbr-artifact-capture.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-vbr-artifact-store.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-prompt-cache-payload.cpp)
target_include_directories(test-vbr-artifact-capture PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/tools/server)
# The fork's scale arm: the same binary building a fragmented range-packing benchmark.
llama_test_cmd(test-vbr-artifact-capture
    NAME test-vbr-range-proof-scale
    LABEL "vbr-cache;slow"
    ARGS --range-pack-bench)

# The adversarial oracle the port wrote for the same header before the fork's file landed. Kept
# because it pins properties the fork's test never names: overlapping / unsorted / out-of-extent
# range lists and "restriction never widens authority" (asserted as status != restricted, not as one
# enum spelling). Self-reporting, so no /UNDEBUG.
llama_build_and_test(test-vbr-artifact-capture-oracle.cpp)
target_include_directories(test-vbr-artifact-capture-oracle PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Artifact adoption: the portable, device-free half of the fork's
# tests/test-vbr-artifact-adopt.cpp. Self-reporting (a failure counter and an
# explicit pass line), so no /UNDEBUG. The other half of that file -- the CUDA
# H2D adapter and the two model-backed drivers, 1831 lines -- is deliberately
# NOT ported: it needs a GPU, and its first statement is an early-out
# `if (!device) return;`, so it would pass here while asserting nothing. The
# model-adoption harness that only that half references is left with it.
#
# No tools/server include directory: the ported range references ZERO `server_*`
# symbols, and the fork's server includes are not free -- server-task.h reaches
# server-common.h which reaches mtmd.h, and tools/mtmd is not on a test's
# include path here (measured: fatal error C1083 on "mtmd.h").
llama_build_and_test(test-vbr-artifact-adopt.cpp)
target_include_directories(test-vbr-artifact-adopt PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Artifact codec admission gates, the first test that reaches the whole artifact strongly-connected
# component (artifact + generation + occupied-replacement + catalog + validate). Self-reporting, so
# no /UNDEBUG. Scope is admission plus the two shared helpers; a full encode/decode round trip needs
# a complete unit-artifact package and is still open (MERGE_BUUN_PLAN §10.38).
llama_build_and_test(test-vbr-artifact-codec.cpp)
target_include_directories(test-vbr-artifact-codec PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Cache-plan / cache-lease / cache-yield / retention unit tests.
#
# Deliberate deviation from the fork: it registers these inside the
# `if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)` block, which would silently create
# no target at all on this Windows + BUILD_SHARED_LIBS=ON machine. They are pure
# logic tests over headers that need no llama internals, and each one compiles
# its own tools/server module .cpp straight into the executable -- so they do
# not depend on the server-context target either, which is what forces the
# placement outside the model-gated block.
llama_build_and_test(
    test-cache-plan-authority.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-plan-authority.cpp)
target_include_directories(test-cache-plan-authority PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/tools/server)

llama_build_and_test(
    test-cache-lease.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-lease.cpp)
target_include_directories(test-cache-lease PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/tools/server)

llama_build_and_test(test-cache-plan-record.cpp)
target_include_directories(test-cache-plan-record PRIVATE ${PROJECT_SOURCE_DIR}/src)

llama_build_and_test(
    test-cache-yield.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-yield.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-lease.cpp)
target_include_directories(test-cache-yield PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/tools/server)

llama_build_and_test(
    test-cache-destruction-quote.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-destruction-quote.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-retention-proof.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-yield.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-lease.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-plan-authority.cpp)
target_include_directories(test-cache-destruction-quote PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/tools/server)

llama_build_and_test(
    test-retention-policy-corpus.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-retention-sidecar.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-yield.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-lease.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-destruction-quote.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-plan-authority.cpp)
target_include_directories(test-retention-policy-corpus PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/tools/server)

llama_build_and_test(test-cache-accounting.cpp)
target_include_directories(test-cache-accounting PRIVATE ${PROJECT_SOURCE_DIR}/src)

llama_build_and_test(test-cache-budget.cpp)
target_include_directories(test-cache-budget PRIVATE ${PROJECT_SOURCE_DIR}/src)

llama_build_and_test(test-cache-authority.cpp)
target_include_directories(test-cache-authority PRIVATE ${PROJECT_SOURCE_DIR}/src)

llama_build_and_test(
    test-retention-sidecar.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-retention-sidecar.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-lease.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-destruction-quote.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-yield.cpp
    ${PROJECT_SOURCE_DIR}/tools/server/server-cache-plan-authority.cpp)
target_include_directories(test-retention-sidecar PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/tools/server)

# Representation-epoch / identity. The epoch binary is llama_build() only because
# its fixture arm needs a real iSWA Gemma-4 model this repo cannot synthesise; the
# three CPU entry points below are self-contained. The fork leaves them unlabelled,
# which would make them invisible to a label-filtered run, so they are given the
# `main` label this tree uses for every other test.
llama_build(test-vbr-representation-epoch.cpp)
target_include_directories(test-vbr-representation-epoch PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(
    NAME test-vbr-controller-identity-cpu
    COMMAND $<TARGET_FILE:test-vbr-representation-epoch> --identity-cpu)
set_property(TEST test-vbr-controller-identity-cpu PROPERTY LABELS main)
add_test(
    NAME test-vbr-generation-tracker-cpu
    COMMAND $<TARGET_FILE:test-vbr-representation-epoch> --generation-cpu)
set_property(TEST test-vbr-generation-tracker-cpu PROPERTY LABELS main)
add_test(
    NAME test-vbr-operation-recovery-cpu
    COMMAND $<TARGET_FILE:test-vbr-representation-epoch> --operation-cpu)
set_property(TEST test-vbr-operation-recovery-cpu PROPERTY LABELS main)

llama_build_and_test(test-vbr-representation-identity.cpp)
target_include_directories(test-vbr-representation-identity PRIVATE ${PROJECT_SOURCE_DIR}/src)

# The SWA-window capture/install gates need a real attention-only iSWA model
# (Gemma4-class), which no CI machine has -- the fork registers this one as
# build-only for exactly that reason. Keep it that way: it is still the only
# link-time consumer of src/llama-vbr-swa-window.cpp, so building it proves the
# new translation unit really lands in libllama.
llama_build(test-vbr-swa-window.cpp)
target_include_directories(test-vbr-swa-window PRIVATE ${PROJECT_SOURCE_DIR}/src)

# MTP carry lifecycle + host-checkpoint codec (slice 9h). Pure logic over a bool
# and a float vector: no model, no context, so unlike the SWA-window binary above
# this one is a genuine runtime consumer -- it pins target_process_mode()'s truth
# table and the codec's wire format. Lives outside the
# `if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)` block on purpose; inside it the
# target would silently not exist on this Windows + shared build.
llama_build_and_test(test-speculative-mtp-carry.cpp)

# MoE expert cache (slice 9l-d5b S8).
#
# Four sources. Three build here; the fourth needs a real CUDA device, so it
# keeps exactly the fork's gates and stays source-only on this machine.
#
# test-moe-cache.cpp and the flat-hits selector are registered at top level
# rather than inside `if (NOT LLAMA_USE_SYSTEM_GGML)` the way the fork does it:
# the fork's block is the tail of a CUDA-instrumentation region this tree does
# not carry, and this tree's two such blocks hold unrelated tests
# (test-backend-meta-replay; test-gguf/test-alloc). All four reach ggml's
# private headers through source-relative paths (`../ggml/src/...`), so unlike
# test-backend-expert-transfer they need no target_include_directories.
#
# The fork gates the fourth source on `TARGET ggml-cuda AND NOT GGML_BACKEND_DL`
# alone. That is not enough here. With BUILD_SHARED_LIBS=ON and DL off,
# ggml_add_backend_library (ggml/src/CMakeLists.txt:396) uses a typeless
# add_library, so ggml-cuda is an ordinary shared library and ggml-cuda.lib is an
# IMPORT library that carries only the symbols marked GGML_BACKEND_API
# (ggml/include/ggml-backend.h:6-18; GGML_BACKEND_SHARED is set at
# ggml/src/CMakeLists.txt:404-406, and ggml-cuda sets no
# WINDOWS_EXPORT_ALL_SYMBOLS). This test calls ggml_cuda_moe_cache_mmv, an
# internal helper declared in mmvq.cuh with no export macro, so against a shared
# ggml-cuda the exe can never link (LNK2019) -- while the same call from
# moe-cache.cu resolves inside ggml-cuda.dll and the server link stays green,
# which is what makes the failure look like a stale library. It is not one.
# Same property as the kv-stream/turbo/vbr-vmm tests above, one step stronger;
# same remedy as their comment gives: skip instead of failing the link.
llama_build(test-moe-cache.cpp)
llama_test(test-moe-cache LABEL "main;backend")
set_tests_properties(test-moe-cache PROPERTIES TIMEOUT 120)

llama_build_and_test(test-cuda-moe-cache-flat-hits-selector.cpp)

llama_build_and_test(test-moe-cache-fit.cpp)

if (TARGET ggml-cuda AND NOT GGML_BACKEND_DL AND NOT BUILD_SHARED_LIBS)
    find_package(CUDAToolkit REQUIRED)
    llama_build(test-cuda-moe-cache-flat-hits.cpp)
    target_link_libraries(test-cuda-moe-cache-flat-hits PRIVATE ggml-cuda CUDA::cudart)
    llama_test(test-cuda-moe-cache-flat-hits LABEL "main;backend")
    set_property(TEST test-cuda-moe-cache-flat-hits PROPERTY ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
    set_tests_properties(test-cuda-moe-cache-flat-hits PROPERTIES TIMEOUT 120)
    if (GGML_CUDA_MOE_CACHE_FLAT_HITS_TEST_INSTRUMENTATION)
        target_compile_definitions(test-cuda-moe-cache-flat-hits
            PRIVATE GGML_CUDA_MOE_CACHE_FLAT_HITS_TEST_INSTRUMENTATION)
    endif()
endif()
if (TARGET ggml-cuda AND GGML_CUDA_MOE_CACHE_FLAT_HITS_TEST_INSTRUMENTATION)
    llama_test(test-moe-cache NAME test-moe-cache-flat-hits-baseline LABEL "main;backend")
    set_property(TEST test-moe-cache-flat-hits-baseline PROPERTY
        ENVIRONMENT "GGML_TEST_MOE_CACHE_FLAT_HITS_EXPECT_BASELINE=1;CUDA_VISIBLE_DEVICES=0")
    set_tests_properties(test-moe-cache-flat-hits-baseline PROPERTIES TIMEOUT 180)
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp  LABEL behavior  LINK ggml)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp  LABEL "behavior backend"  LINK ggml)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()
