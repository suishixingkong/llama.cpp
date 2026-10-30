llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

set(LLAMA_CORE_SOURCES
    llama.cpp
    llama-adapter.cpp
    llama-arch.cpp
    llama-batch.cpp
    llama-chat.cpp
    llama-context.cpp
    llama-cparams.cpp
    llama-grammar.cpp
    llama-graph.cpp
    llama-hparams.cpp
    llama-impl.cpp
    llama-io.cpp
    llama-kv-cache.cpp
    llama-kv-cache-iswa.cpp
    llama-kv-cache-dsa.cpp
    llama-kv-cache-dsa-iswa.cpp
    llama-kv-cache-msa.cpp
    llama-kv-cache-dsv4.cpp
    llama-kv-stream-config.cpp
    llama-kv-stream-config.h
    llama-bit-ops.h
    llama-sha256.h
    llama-vbr-codec.cpp
    llama-vbr-codec.h
    llama-vbr-checkpoint-types.h
    llama-vbr-controller-id.cpp
    llama-vbr-controller-id.h
    llama-vbr-generation-types.h
    llama-vbr-policy.h
    llama-vbr-physical.h
    llama-vbr-transaction.h
    llama-vbr-hard-seal.cpp
    llama-vbr-hard-seal.h
    llama-vbr-identity-digest.h
    llama-vbr-downward.cpp
    llama-vbr-downward.h
    llama-vbr-upward.cpp
    llama-vbr-upward.h
    llama-vbr-operation.cpp
    llama-vbr-operation.h
    llama-vbr-extent.cpp
    llama-vbr-extent.h
    llama-vbr-pinned-ring.cpp
    llama-vbr-pinned-ring.h
    llama-vbr-artifact.h
    llama-vbr-generation.h
    llama-vbr-artifact-capture.cpp
    llama-vbr-artifact-capture.h
    llama-vbr-artifact.cpp
    llama-vbr-generation.cpp
    llama-vbr-occupied-replacement.cpp
    llama-vbr-occupied-replacement.h
    llama-vbr-artifact-catalog.cpp
    llama-vbr-artifact-catalog.h
    llama-vbr-artifact-validate.cpp
    llama-vbr-artifact-validate.h
    llama-vbr-explicit-capture.cpp
    llama-vbr-explicit-capture.h
    llama-vbr-swa-window.cpp
    llama-vbr-swa-window.h
    llama-vbr-qsa-index.cpp
    llama-vbr-qsa-index.h
    llama-vbr-artifact-stage.cpp
    llama-vbr-artifact-stage.h
    llama-vbr-artifact-adopt.cpp
    llama-vbr-artifact-adopt.h
    llama-vram-demand.cpp
    llama-vram-demand.h
    llama-vram-ledger.cpp
    llama-vram-ledger.h
    llama-cache-accounting.cpp
    llama-cache-accounting.h
    llama-cache-budget.cpp
    llama-cache-budget.h
    llama-cache-authority.cpp
    llama-cache-authority.h
    llama-memory-tree.cpp
    llama-memory-tree.h
    llama-kv-stream-plan.cpp
    llama-kv-stream-plan.h
    llama-kv-stream-softmax.cpp
    llama-kv-stream-softmax.h
    llama-memory.cpp
    llama-moe-cache.cpp
    llama-memory-hybrid.cpp
    llama-memory-hybrid-iswa.cpp
    llama-memory-hybrid-idx.cpp
    llama-memory-recurrent.cpp
    llama-mmap.cpp
    llama-model-loader.cpp
    llama-model-saver.cpp
    llama-model.cpp
    llama-quant.cpp
    llama-repack-cache.cpp
    llama-sampler.cpp
    llama-vocab.cpp
    unicode-data.cpp
    unicode.cpp
)

add_library(llama
            ../include/llama.h
            ${LLAMA_CORE_SOURCES}
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 16
)

# exclude non-model sources from unity build
set_source_files_properties(${LLAMA_CORE_SOURCES} ../include/llama.h unicode.h
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

configure_file(llama-version.h.in ${CMAKE_CURRENT_BINARY_DIR}/llama-version.h @ONLY)

# ../vendor is the fork's own addition -- neither the common ancestor bfdc32183 nor upstream
# master has it. src/llama-repack-cache.cpp reaches <nlohmann/json.hpp> through it, so it
# comes in together with that file; nothing else in this tree's src/ needs it.
target_include_directories(llama PRIVATE . ../vendor ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (LLAMA_KVMEM)
    if (LLAMA_KVMEM_ROOT STREQUAL "")
        message(FATAL_ERROR "LLAMA_KVMEM=ON requires LLAMA_KVMEM_ROOT")
    endif()
    target_sources(llama PRIVATE
        ${LLAMA_KVMEM_ROOT}/src/adapter/llama-memory-kvmem.cpp
        ${LLAMA_KVMEM_ROOT}/src/adapter/llama-kvmem-attention.cpp
        ${LLAMA_KVMEM_ROOT}/src/adapter/llama-memory-kvmem-hybrid.cpp
        ${LLAMA_KVMEM_ROOT}/src/adapter/llama-memory-kvmem-mtp.cpp
        ${LLAMA_KVMEM_ROOT}/src/adapter/llama-kvmem-batch.cpp
        ${LLAMA_KVMEM_ROOT}/src/adapter/llama-kvmem-capture.cpp
    )
    if (GGML_CUDA)
        enable_language(CUDA)
        target_sources(llama PRIVATE
            ${LLAMA_KVMEM_ROOT}/src/adapter/llama-kvmem-attention-cuda.cu
            ${LLAMA_KVMEM_ROOT}/src/adapter/llama-kvmem-stagein.cu)
        if (DEFINED CMAKE_CUDA_ARCHITECTURES)
            set_property(TARGET llama PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
        endif()
    endif()
    target_include_directories(llama PRIVATE
        ${LLAMA_KVMEM_ROOT}/kvmem/include
        ${LLAMA_KVMEM_ROOT}/src/adapter
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src
    )
    target_compile_definitions(llama PRIVATE LLAMA_KVMEM=1)
    if (NOT TARGET kvmem)
        # KVMem registers its unit tests from inside that subdirectory, but the root only
        # runs include(CTest) after src/, so add_test() there would be dropped and the
        # tests would build without ever running under ctest. Enable testing one level
        # above the kvmem tree instead of reordering the root file.
        enable_testing()
        add_subdirectory(${LLAMA_KVMEM_ROOT}/kvmem ${CMAKE_BINARY_DIR}/kvmem-from-llama)
    endif()
    if (GGML_CUDA)
        # The retrieval round's page-softmax scorer has a CUDA implementation, and it is
        # opted in by a compile definition on the kvmem target, not by the backend the
        # adapter was given. Skipping it leaves the adapter to score every page on the
        # host, which measured 4195 ms for 3156 rows against 25 ms for the whole round
        # otherwise: the round then dominates the request's prefill.
        kvmem_enable_cuda_recall()
    endif()
    target_link_libraries(llama PRIVATE kvmem)
    if (GGML_CUDA)
        find_package(CUDAToolkit QUIET)
        if (CUDAToolkit_FOUND)
            target_link_libraries(llama PRIVATE CUDA::cudart)
        endif()
    endif()
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if (WIN32)
        # Internal C++ implementation seams are consumed by llama-common and
        # server-context but are not part of the installed C API. Export them
        # from the Windows DLL without annotating them as stable LLAMA_API.
        set_target_properties(llama PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
