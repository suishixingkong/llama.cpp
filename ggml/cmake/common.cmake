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
    set(FA_TYPES q4_0 q4_1 q5_0 q5_1 q8_0 bf16 f16 turbo2_0 turbo3_0 turbo4_0 turbo8_0 turbo3_tcq turbo2_tcq turbo1_tcq)

    string(TOLOWER "${GGML_CUDA_FA_QUANTS}" FA_QUANTS)
    string(STRIP   "${FA_QUANTS}" FA_QUANTS)
    if (GGML_CUDA_FA_ALL_QUANTS)
        message(WARNING "GGML_CUDA_FA_ALL_QUANTS is deprecated, use GGML_CUDA_FA_QUANTS=all instead")
        set(FA_QUANTS all)
    endif()
    if (NOT FA_QUANTS)
        message(FATAL_ERROR "GGML_CUDA_FA_QUANTS must not be empty")
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
                if (NOT TYPE IN_LIST FA_TYPES)
                    message(FATAL_ERROR
                        "GGML_CUDA_FA_QUANTS: unknown type \"${TYPE}\" in \"${COMBINATION}\", must be one of: ${FA_TYPES}")
                endif()
            endforeach()
            list(APPEND FA_COMBINATIONS ${TYPE_K}-${TYPE_V})
        endforeach()
    endif()
    list(REMOVE_DUPLICATES FA_COMBINATIONS)

    string(REPLACE ";" "," FA_QUANTS_DEFINE "${FA_QUANTS}")
    add_compile_definitions(GGML_CUDA_FA_QUANTS="${FA_QUANTS_DEFINE}")
    foreach (TYPE_V IN LISTS FA_TYPES)
        foreach (TYPE_K IN LISTS FA_TYPES)
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

    # fork 9l-d4b9z: this tree only ships instances for pairs with bpv(K) >= bpv(V) -- K is the
    # side dot-producted against Q, so a coarser K would drop precision the V side still has.
    # Report every offender at once: a stale GGML_CUDA_FA_QUANTS cache naming a removed pair
    # otherwise fails one line at a time, which reads like a missing file rather than a
    # configuration that has to be re-derived.
    set(FA_MISSING_INSTANCES "")
    foreach (COMBINATION IN LISTS FA_COMBINATIONS)
        set(SRC "${DIR}/template-instances/fattn-vec-instance-${COMBINATION}.cu")
        if (NOT EXISTS "${SRC}")
            list(APPEND FA_MISSING_INSTANCES "${COMBINATION}")
        endif()
    endforeach()
    if (FA_MISSING_INSTANCES)
        message(FATAL_ERROR
            "GGML_CUDA_FA_QUANTS names ${#FA_MISSING_INSTANCES} combination(s) with no template "
            "instance in this tree: ${FA_MISSING_INSTANCES}\n"
            "Only pairs with bpv(K) >= bpv(V) are instantiated (K is the dot-product side). "
            "Drop them from GGML_CUDA_FA_QUANTS, or rebuild the cache from scratch "
            "(-U GGML_CUDA_FA_QUANTS) to pick up the current default.")
    endif()

    set(SRCS "")
    foreach (COMBINATION IN LISTS FA_COMBINATIONS)
        set(SRC "${DIR}/template-instances/fattn-vec-instance-${COMBINATION}.cu")
        list(APPEND SRCS "${SRC}")
    endforeach()

    set(${OUT_SRCS} ${SRCS} PARENT_SCOPE)
    # Expose the selected K-V combinations (e.g. "q8_0-q8_0", "turbo3_tcq-turbo3_tcq") so other
    # instance families can follow the same GGML_CUDA_FA_QUANTS selection. Currently consumed by
    # ggml_cuda_fattn_mma_turbo_instances() below.
    set(GGML_FA_COMBINATIONS "${FA_COMBINATIONS}" PARENT_SCOPE)
endfunction()

# Determines which non-f16 MMA FlashAttention template instances to compile, returns them in
# OUT_SRCS. These instances follow the vec instances: a <type_K>-<type_V> pair is only compiled
# when its combination was selected via GGML_CUDA_FA_QUANTS (or exists under
# GGML_CUDA_FA_QUANTS=all). Combinations without any instance files are silently skipped, and the
# FA dispatch in fattn.cu guards every turbo case reference with the same GGML_CUDA_FA_<K>_<V>
# macros that this selection defines, so an uncompiled pair falls back to the regular kernels
# instead of failing to link.
#
# This function is only reached when GGML_CUDA_FA_MMA is ON -- it is called from
# ggml_fattn_mma_quant_instances() below, which is the single gate for the whole family. With the
# switch OFF none of these files are compiled and GGML_CUDA_FA_MMA is left undefined, which
# fattn-mma-turbo.cuh reads as 0 and ANDs into every per-pair guard, so a pair selected in
# GGML_CUDA_FA_QUANTS -- whose GGML_CUDA_FA_<K>_<V> macro is still 1, because that macro gates the
# VEC instances too -- is not referenced here. The VEC instances are deliberately left alone:
# GGML_CUDA_FA_MMA is about MMA kernels only.
function(ggml_cuda_fattn_mma_turbo_instances DIR COMBINATIONS OUT_SRCS)
    set(SRCS "")
    foreach (COMBINATION IN LISTS ${COMBINATIONS})
        file(GLOB TURBO_MMA_SRCS "${DIR}/template-instances/fattn-mma-${COMBINATION}-instance-*.cu")
        list(APPEND SRCS ${TURBO_MMA_SRCS})
    endforeach()
    set(${OUT_SRCS} ${SRCS} PARENT_SCOPE)
endfunction()

# Master switch for the non-f16 MMA (tensor-core) FlashAttention kernels: GGML_CUDA_FA_MMA
# (CMake option, default OFF). Shared by the CUDA, HIP and MUSA backends -- they all compile the
# same fattn-mma-<k>-<v>-instance-*.cu sources -- and returns the selected files in OUT_SRCS.
#
# Two halves, and both are required:
#   * the source list: with the switch off, none of those instance files are compiled, leaving
#     fattn-mma-f16-instance-*.cu (f16-f16) as the only MMA kernel in the build;
#   * the compile definition: fattn-mma-turbo.cuh ANDs GGML_CUDA_FA_MMA into every
#     per-pair guard, so with the define absent no translation unit references an instance this
#     selection did not compile. Dropping it makes the ON build silently useless (instances
#     compiled, nothing reachable) and the OFF build a LINK error rather than a fallback:
#     GGML_CUDA_FA_<K>_<V> stays 1 for every GGML_CUDA_FA_QUANTS pair, because that macro gates
#     the VEC instances too and those are compiled either way.
#
# The define is added ONLY in the ON branch, deliberately. add_compile_definitions() is a
# directory property, so adding `-D...=0` for the OFF case would change the compile command of
# every source in the backend and force a full rebuild of it (hundreds of CUDA translation units)
# the first time the option is seen. The header's own `#ifndef ... #define ... 0` supplies the
# same value, so the default OFF build recompiles nothing and only relinks.
function(ggml_fattn_mma_quant_instances DIR COMBINATIONS OUT_SRCS)
    if (GGML_CUDA_FA_MMA)
        add_compile_definitions(GGML_CUDA_FA_MMA=1)
        ggml_cuda_fattn_mma_turbo_instances(${DIR} ${COMBINATIONS} SRCS)
        message(STATUS "FlashAttention MMA K-V instances: ${${COMBINATIONS}}")
    else()
        set(SRCS "")
        message(STATUS "FlashAttention MMA K-V instances: none (f16-f16 MMA only, set GGML_CUDA_FA_MMA=ON to compile them)")
    endif()
    set(${OUT_SRCS} ${SRCS} PARENT_SCOPE)
endfunction()
