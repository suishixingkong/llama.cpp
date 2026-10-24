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
    server-decision.cpp
    server-decision.h
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

# C++ unit test for the checkpoint-publication latch trio (slice 9l-d4b8i):
# server_cache_checkpoint_publication_attempt_begin,
# server_cache_checkpoint_refusal_state_changed, and
# server_cache_checkpoint_publication_skipped.  Declared in
# server-cache-authority.h since 8q, defined nowhere until this slice -- the same
# latent dangling declaration as the ring-changed test above, and whose first
# caller is the VBR capture kernel landed by the same slice.  Model-free.
if (LLAMA_BUILD_TESTS)
    add_executable(test-cache-checkpoint-publication-latch
        ${PROJECT_SOURCE_DIR}/tests/test-cache-checkpoint-publication-latch.cpp)
    target_link_libraries(test-cache-checkpoint-publication-latch PRIVATE
        server-context mtmd llama llama-common)
    target_include_directories(test-cache-checkpoint-publication-latch PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tools/mtmd ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(test-cache-checkpoint-publication-latch PRIVATE cxx_std_17)
    add_test(NAME test-cache-checkpoint-publication-latch COMMAND test-cache-checkpoint-publication-latch)
    set_property(TEST test-cache-checkpoint-publication-latch PROPERTY LABELS main)
endif()

# The fork's model-free prompt-cache suite (slice 9l-d4b8d).  5,114 lines, and the
# reason 9l-d4b8b/9l-d4b8c exist: it constructs a `server_prompt_cache` with NO
# llama_context and pins each phase of a host restore, of the retention mirror, and
# of teardown on its own.  It is also the first thing that actually CALLS seven of
# the nine `*_for_test` oracles in this port -- until now they were landed but
# inert.  Model-free; runs on CPU.
if (LLAMA_BUILD_TESTS)
    add_executable(test-server-prompt-cache
        ${PROJECT_SOURCE_DIR}/tests/test-server-prompt-cache.cpp)
    target_link_libraries(test-server-prompt-cache PRIVATE
        server-context mtmd llama llama-common)
    target_include_directories(test-server-prompt-cache PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tools/mtmd ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(test-server-prompt-cache PRIVATE cxx_std_17)
    add_test(NAME test-server-prompt-cache COMMAND test-server-prompt-cache)
    set_property(TEST test-server-prompt-cache PROPERTY LABELS main)
    set_property(TEST test-server-prompt-cache APPEND PROPERTY LABELS vbr-cache)
    # fork 9l-d4b9k: fork tools/server/CMakeLists.txt 130-146.  Two of the three
    # model-free fault modes of the prompt-cache suite.  They pin ordering
    # guarantees that only fire on a failed accounting commit and on a partial
    # checkpoint ownership at restore.  The modes exist in the test's main()
    # either way; without these registrations they were never run.
    #
    # The third mode (`--clone-fault`, LLAMA_SERVER_FAULT=load_clone_fail) is NOT
    # registered: its test asserts the cache-plan OBSERVER row
    # (`rec.selected_row(...) != nullptr`), and this port's load is the
    # fork's UNOBSERVED instantiation flattened into a plain function -- the
    # observed/unobserved dispatch is a separate refactor.  Registering it here
    # would add a permanently red test that measures a feature this port does not
    # carry, not a defect in the restore path.  See the 9l-d4b9k log.
    add_test(NAME test-server-prompt-cache-accounting-fault
        COMMAND test-server-prompt-cache --accounting-fault)
    set_tests_properties(test-server-prompt-cache-accounting-fault PROPERTIES
        LABELS "main;vbr-cache"
        ENVIRONMENT "LLAMA_SERVER_FAULT=cache_lifecycle_after_commit")
    add_test(NAME test-server-prompt-cache-restore-checkpoint-fault
        COMMAND test-server-prompt-cache --restore-checkpoint-fault)
    set_tests_properties(
        test-server-prompt-cache-restore-checkpoint-fault PROPERTIES
        LABELS "main;vbr-cache"
        ENVIRONMENT "LLAMA_SERVER_FAULT=restore_checkpoint_artifact_missing")
endif()
