# llama-bench-impl: benchmark logic, reusable by app

set(TARGET llama-bench-impl)

add_library(${TARGET} llama-bench.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})

if (LLAMA_KVMEM)
    # llama-bench.cpp publishes the KVMem parameters directly, so it needs the same
    # adapter headers llama-common uses (llama-common only exports the define).
    target_include_directories(${TARGET} PRIVATE
        ${LLAMA_KVMEM_ROOT}/src/adapter
        ${LLAMA_KVMEM_ROOT}/kvmem/include)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-bench executable

set(TARGET llama-bench)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-bench-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
