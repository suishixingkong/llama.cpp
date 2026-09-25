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
    llama-memory-hybrid.cpp
    llama-memory-hybrid-iswa.cpp
    llama-memory-hybrid-idx.cpp
    llama-memory-recurrent.cpp
    llama-mmap.cpp
    llama-model-loader.cpp
    llama-model-saver.cpp
    llama-model.cpp
    llama-quant.cpp
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

target_include_directories(llama PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
