include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-cache-authority.cpp
    server-cache-authority.h
    server-cache-control.cpp
    server-cache-control.h
    server-cache-control-wire.cpp
    server-cache-destruction-quote.cpp
    server-cache-destruction-quote.h
    server-cache-plan-authority.cpp
    server-cache-plan-authority.h
    server-cache-plan-preflight.cpp
    server-cache-plan-preflight.h
    server-cache-plan-preflight-internal.h
    server-cache-lease.cpp
    server-cache-lease.h
    server-cache-lifecycle.h
    server-cache-retention-proof.cpp
    server-cache-retention-proof.h
    server-cache-vbr-proof.cpp
    server-cache-vbr-proof.h
    server-cache-yield.cpp
    server-cache-yield.h
    server-prompt-cache-payload.cpp
    server-prompt-cache-payload.h
    server-recurrent-expansion.h
    server-retention-sidecar.cpp
    server-retention-sidecar.h
    server-vbr-artifact-store.cpp
    server-vbr-artifact-store.h
    server-vbr-capture-readiness.cpp
    server-vbr-capture-readiness.h
    server-vbr-prompt-cache-support.cpp
    server-vbr-prompt-cache-support.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-stream.cpp
    server-stream.h
    server-tools.cpp
    server-tools.h
    server-mcp.cpp
    server-mcp.h
    server-schema.cpp
    server-schema.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})

# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

add_dependencies(${TARGET} llama-ui-assets)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# C++ unit test for the pure cache/VBR decision helpers (slice 7d part 1). Lives here
# rather than tests/ because that subdir is configured before tools/, so the
# server-context target is not yet defined there. Model-free — runs on CPU.
if (LLAMA_BUILD_TESTS)
    add_executable(test-server-vbr-predicates
        ${PROJECT_SOURCE_DIR}/tests/test-server-vbr-predicates.cpp)
    target_link_libraries(test-server-vbr-predicates PRIVATE
        server-context mtmd llama llama-common)
    target_include_directories(test-server-vbr-predicates PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tools/mtmd ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(test-server-vbr-predicates PRIVATE cxx_std_17)
    add_test(NAME test-server-vbr-predicates COMMAND test-server-vbr-predicates)
    set_property(TEST test-server-vbr-predicates PROPERTY LABELS main)
endif()

# C++ unit test for the cache-receipt chain (slice 9i). Same reasoning as above:
# it needs server-common.cpp, which is only reachable through server-context.
# Model-free — runs on CPU.
if (LLAMA_BUILD_TESTS)
    add_executable(test-cache-receipt-chain
        ${PROJECT_SOURCE_DIR}/tests/test-cache-receipt-chain.cpp)
    target_link_libraries(test-cache-receipt-chain PRIVATE
        server-context mtmd llama llama-common)
    target_include_directories(test-cache-receipt-chain PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tools/mtmd ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(test-cache-receipt-chain PRIVATE cxx_std_17)
    add_test(NAME test-cache-receipt-chain COMMAND test-cache-receipt-chain)
    set_property(TEST test-cache-receipt-chain PROPERTY LABELS main)
endif()

# C++ unit test for the retention-reuse predicate (slice 9j). Same reasoning as
# above: server-task.h is only reachable through server-context. Model-free.
if (LLAMA_BUILD_TESTS)
    add_executable(test-retention-reuse-useful
        ${PROJECT_SOURCE_DIR}/tests/test-retention-reuse-useful.cpp)
    target_link_libraries(test-retention-reuse-useful PRIVATE
        server-context mtmd llama llama-common)
    target_include_directories(test-retention-reuse-useful PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tools/mtmd ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(test-retention-reuse-useful PRIVATE cxx_std_17)
    add_test(NAME test-retention-reuse-useful COMMAND test-retention-reuse-useful)
    set_property(TEST test-retention-reuse-useful PROPERTY LABELS main)
endif()

# C++ unit test for server_cache_checkpoint_ring_changed (slice 9k-a). Same
# reasoning as above: the definition lives in server-task.cpp, which is only
# reachable through the server-context target. Model-free.
if (LLAMA_BUILD_TESTS)
    add_executable(test-cache-checkpoint-ring-changed
        ${PROJECT_SOURCE_DIR}/tests/test-cache-checkpoint-ring-changed.cpp)
    target_link_libraries(test-cache-checkpoint-ring-changed PRIVATE
        server-context mtmd llama llama-common)
    target_include_directories(test-cache-checkpoint-ring-changed PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tools/mtmd ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(test-cache-checkpoint-ring-changed PRIVATE cxx_std_17)
    add_test(NAME test-cache-checkpoint-ring-changed COMMAND test-cache-checkpoint-ring-changed)
    set_property(TEST test-cache-checkpoint-ring-changed PROPERTY LABELS main)
endif()
