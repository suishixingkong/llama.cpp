function(ggml_get_flags CCID CCVER)
    set(C_FLAGS "")
    set(CXX_FLAGS "")

    if (CCID MATCHES "Clang")
        set(C_FLAGS   -Wunreachable-code-break -Wunreachable-code-return)
        set(CXX_FLAGS -Wunreachable-code-break -Wunreachable-code-return -Wmissing-prototypes -Wextra-semi)

        if (
            (CCID STREQUAL "Clang"      AND CCVER VERSION_GREATER_EQUAL 3.8.0) OR
            (CCID STREQUAL "AppleClang" AND CCVER VERSION_GREATER_EQUAL 7.3.0)
        )
            list(APPEND C_FLAGS -Wdouble-promotion)
        endif()
    elseif (CCID STREQUAL "GNU")
        set(C_FLAGS   -Wdouble-promotion)
        set(CXX_FLAGS -Wno-array-bounds)

        if (CCVER VERSION_GREATER_EQUAL 8.1.0)
            list(APPEND CXX_FLAGS -Wextra-semi)
        endif()
    endif()

    set(GF_C_FLAGS   ${C_FLAGS}   PARENT_SCOPE)
    set(GF_CXX_FLAGS ${CXX_FLAGS} PARENT_SCOPE)
endfunction()

function(ggml_get_system_arch)
    if (CMAKE_OSX_ARCHITECTURES      STREQUAL "arm64" OR
        CMAKE_GENERATOR_PLATFORM_LWR STREQUAL "arm64" OR
        (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_GENERATOR_PLATFORM_LWR AND
            CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm.*|ARM64)$"))
        set(GGML_SYSTEM_ARCH "ARM" PARENT_SCOPE)
    elseif (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR
            CMAKE_GENERATOR_PLATFORM_LWR MATCHES "^(x86_64|i686|amd64|x64|win32)$" OR
            (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_GENERATOR_PLATFORM_LWR AND
            CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|i686|AMD64|amd64)$"))
        set(GGML_SYSTEM_ARCH "x86" PARENT_SCOPE)
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc|power")
        set(GGML_SYSTEM_ARCH "PowerPC" PARENT_SCOPE)
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
        set(GGML_SYSTEM_ARCH "loongarch64"  PARENT_SCOPE)
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "riscv64")
        set(GGML_SYSTEM_ARCH "riscv64" PARENT_SCOPE)
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
        set(GGML_SYSTEM_ARCH "s390x" PARENT_SCOPE)
    else()
        set(GGML_SYSTEM_ARCH "UNKNOWN" PARENT_SCOPE)
    endif()
endfunction()

# Determines which FlashAttention vector kernel template instances to compile, returns them in OUT_SRCS.
function(ggml_cuda_fattn_vec_instances DIR OUT_SRCS)
    set(FA_TYPES q4_0 q4_1 q5_0 q5_1 q8_0 bf16 f16)
    set(TQ_FA_TYPES turbo2_0 turbo3_0 turbo4_0)
    set(ALL_FA_TYPES ${FA_TYPES} ${TQ_FA_TYPES})
    # Curated default set, used only when -DGGML_CUDA_FA_QUANTS is not given on the command line.
    set(FA_DEFAULT_QUANTS "f16-f16;q4_0-q4_0;q8_0-q8_0;bf16-bf16;q8_0-q4_0;f16-q8_0")

    string(TOLOWER "${GGML_CUDA_FA_QUANTS}" FA_QUANTS)
    string(STRIP   "${FA_QUANTS}" FA_QUANTS)
    set(USE_DEFAULT OFF)
    if (NOT FA_QUANTS)
        set(USE_DEFAULT ON)
    endif()
    if (GGML_CUDA_FA_ALL_QUANTS)
        message(WARNING "GGML_CUDA_FA_ALL_QUANTS is deprecated, use GGML_CUDA_FA_QUANTS=all instead")
        set(FA_QUANTS all)
    endif()
    if (NOT FA_QUANTS)
        # No explicit -DGGML_CUDA_FA_QUANTS was given: fall back to the curated default set.
        set(FA_QUANTS "${FA_DEFAULT_QUANTS}")
    endif()

    if (FA_QUANTS STREQUAL "all")
        # Compile every FlashAttention K-V template instance that exists. This includes the
        # full set of TurboQuant KV-cache combinations, not just the curated subset below.
        file(GLOB FA_INSTANCE_SRCS "${DIR}/template-instances/fattn-vec-instance-*.cu")
        set(FA_COMBINATIONS "")
        foreach (SRC IN LISTS FA_INSTANCE_SRCS)
            get_filename_component(BASENAME "${SRC}" NAME_WE)
            string(REGEX REPLACE "^fattn-vec-instance-" "" COMBINATION "${BASENAME}")
            list(APPEND FA_COMBINATIONS ${COMBINATION})
        endforeach()
    else()
        set(FA_COMBINATIONS f16-f16)

        string(REPLACE "," ";" FA_SELECTED "${FA_QUANTS}")
        foreach (COMBINATION IN LISTS FA_SELECTED)
            string(STRIP   "${COMBINATION}" COMBINATION)
            if (NOT COMBINATION MATCHES "^([a-z0-9_]+)-([a-z0-9_]+)$")
                message(FATAL_ERROR "GGML_CUDA_FA_QUANTS: \"${COMBINATION}\" is not \"all\" or a <type_K>-<type_V> combination")
            endif()
            set(TYPE_K ${CMAKE_MATCH_1})
            set(TYPE_V ${CMAKE_MATCH_2})
            foreach (TYPE ${TYPE_K} ${TYPE_V})
                if (NOT TYPE IN_LIST ALL_FA_TYPES)
                    message(FATAL_ERROR
                        "GGML_CUDA_FA_QUANTS: unknown type \"${TYPE}\" in \"${COMBINATION}\", must be one of: ${ALL_FA_TYPES}")
                endif()
            endforeach()
            list(APPEND FA_COMBINATIONS ${TYPE_K}-${TYPE_V})
        endforeach()

        # TurboQuant KV-cache instances. The turbo types only ever show up as a KV cache,
        # never as a model weight type. They are appended only when no explicit
        # -DGGML_CUDA_FA_QUANTS was given (i.e. the curated default set is in use); an
        # explicit user list is authoritative and is compiled exactly as written. (Has no
        # effect when GGML_CUDA_FA_ALL_QUANTS=ON, which took the "all" branch above and
        # already compiles every available TurboQuant instance.)
        if (USE_DEFAULT)
            set(TQ_FA_COMBINATIONS
                f16-turbo4_0      q8_0-turbo4_0
                q8_0-turbo3_0     q8_0-turbo2_0
                turbo4_0-turbo4_0 turbo4_0-turbo3_0
                turbo4_0-turbo2_0 turbo3_0-turbo3_0
                turbo3_0-turbo2_0 turbo2_0-turbo2_0)
            list(APPEND FA_COMBINATIONS ${TQ_FA_COMBINATIONS})
        endif()
    endif()
    list(REMOVE_DUPLICATES FA_COMBINATIONS)

    string(REPLACE ";" "," FA_QUANTS_DEFINE "${FA_QUANTS}")
    add_compile_definitions(GGML_CUDA_FA_QUANTS="${FA_QUANTS_DEFINE}")
    foreach (TYPE_V IN LISTS ALL_FA_TYPES)
        foreach (TYPE_K IN LISTS ALL_FA_TYPES)
            if ("${TYPE_K}-${TYPE_V}" IN_LIST FA_COMBINATIONS)
                set(COMPILED 1)
            else()
                set(COMPILED 0)
            endif()
            string(TOUPPER "GGML_CUDA_FA_${TYPE_K}_${TYPE_V}" COMBINATION_DEF)
            add_compile_definitions(${COMBINATION_DEF}=${COMPILED})
        endforeach()
    endforeach()

    message(STATUS "FlashAttention K-V type combinations: ${FA_COMBINATIONS}")

    set(SRCS "")
    foreach (COMBINATION IN LISTS FA_COMBINATIONS)
        set(SRC "${DIR}/template-instances/fattn-vec-instance-${COMBINATION}.cu")
        if (NOT EXISTS "${SRC}")
            message(FATAL_ERROR "FlashAttention template instance \"${SRC}\" does not exist")
        endif()
        list(APPEND SRCS "${SRC}")
    endforeach()

    set(${OUT_SRCS} ${SRCS} PARENT_SCOPE)
endfunction()
