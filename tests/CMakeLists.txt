llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE llama llama-common) # TODO: [TAG_TESTS_LLAMA_LINK]
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
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

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
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE llama-common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4           ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

llama_build(test-recurrent-state-rollback.cpp)
llama_build(test-save-load-state.cpp)

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-unicode.cpp)
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-batch-alloc.cpp)
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    llama_build_and_test(test-json-schema-to-grammar.cpp)

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    llama_build(test-llama-archs.cpp)

    set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
    file(MAKE_DIRECTORY "${MODEL_DIR}")

    llama_test(
        test-llama-archs
        NAME test-generate-models
        LABEL main
        ARGS -o "${MODEL_DIR}"
    )
    set_tests_properties(test-generate-models PROPERTIES
        FIXTURES_SETUP generate-models
    )

    llama_test(
        test-recurrent-state-rollback
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback PROPERTIES
        FIXTURES_REQUIRED generate-models
    )

    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-nemotron-h
        LABEL main
        ARGS -m "${MODEL_DIR}/nemotron_h-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-nemotron-h PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-dsv4
        LABEL main
        ARGS -m "${MODEL_DIR}/deepseek4-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-dsv4 PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-kimi-k3
        LABEL main
        ARGS -m "${MODEL_DIR}/kimi-k3-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-kimi-k3 PROPERTIES
        FIXTURES_REQUIRED generate-models
    )

    # Test state save/load functionality across all architectures, using the generated dummy models
    llama_test(
        test-save-load-state
        LABEL main
        ARGS --models "${MODEL_DIR}"
    )
    set_tests_properties(test-save-load-state PROPERTIES FIXTURES_REQUIRED generate-models)

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

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-json-schema.cpp)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
# debug tool for chat template differential analysis (not registered as a test, run it manually)
llama_build(test-chat-analysis.cpp)
llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
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

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp)
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
    # Differential test of the opt-in TurboQuant MMA FlashAttention path (GGML_CUDA_TURBO_MMA=1).
    # Run twice on the device: default = mma mode (counter must advance), "fallback" = env off
    # (counter must stay 0). See BUUN_MERGE_PLAN.md section 10.9.
    llama_build_and_test(test-turbo-mma-fa.cpp LABEL cuda)
endif()
llama_build_and_test(test-model-resolution.cpp)
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()

# TODO: make this test (and others) not link `libllama` as it is not needed [TAG_TESTS_LLAMA_LINK]
llama_build(test-backend-ops.cpp)

llama_build_and_test(test-model-load-cancel.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pattern as test-thread-safety
llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-state-restore-fragmented PROPERTIES FIXTURES_REQUIRED test-download-model)

if (APPLE)
    llama_build(test-rset-release.cpp)
endif()

if (NOT GGML_BACKEND_DL)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp)
    llama_build_and_test(test-quantize-fns.cpp)
    llama_build_and_test(test-quantize-perf.cpp)
    llama_build_and_test(test-rope.cpp)
    llama_build_and_test(test-col2im-1d.cpp)

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
set(LLAMA_TEST_NAME test-mtmd-c-api)
llama_build_and_test(test-mtmd-c-api.c)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
unset(LLAMA_TEST_NAME)

llama_build_and_test(test-mtmd-impl.cpp)
target_link_libraries(test-mtmd-impl PRIVATE mtmd)

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
        llama_test(test-gguf-model-data LABEL "model")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp LABEL "model")
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

# Downward/upward representation algebra: the consumer-free subset of upstream's test-vbr-downward.
# The five upstream cases that need llama-memory.h's preflight types or llama-kv-cache.h's stash
# helpers are stage 2.9 work; the test file's own header comment names them. Unlike the identity and
# hard-seal tests this one *is* assert-based, so it gets /UNDEBUG for the reason the 2.6a group does.
# It includes its header by bare name, hence the src include directory.
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
# need a real device are listed in BUUN_MERGE_PLAN §10.33.
llama_build_and_test(test-vbr-pinned-ring.cpp)
target_include_directories(test-vbr-pinned-ring PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Unit-artifact format vocabulary: the header-only half of llama-vbr-artifact.h. Self-reporting, so
# no /UNDEBUG. The codec (.cpp) is NOT here yet - it links against
# vbr_generation_production_covered_set(), which only llama-vbr-generation.cpp defines, and that
# file cannot land until artifact-validate/occupied-replacement/artifact-capture do. See
# BUUN_MERGE_PLAN §10.35. The portable-domain ordering is pinned against a std::tuple oracle.
llama_build_and_test(test-vbr-artifact.cpp)
target_include_directories(test-vbr-artifact PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Segment chain and Merkle range proofs. Self-reporting, so no /UNDEBUG. The oracle is adversarial
# rather than arithmetic: the tree is a black box that must bind content (one differing byte changes
# the root), stay sound (a proof fails against any other bytes) and admit nothing malformed.
# Restriction is asserted to never widen authority rather than to return one specific status.
llama_build_and_test(test-vbr-artifact-capture.cpp)
target_include_directories(test-vbr-artifact-capture PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Artifact codec admission gates, the first test that reaches the whole artifact strongly-connected
# component (artifact + generation + occupied-replacement + catalog + validate). Self-reporting, so
# no /UNDEBUG. Scope is admission plus the two shared helpers; a full encode/decode round trip needs
# a complete unit-artifact package and is still open (BUUN_MERGE_PLAN §10.38).
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

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()
