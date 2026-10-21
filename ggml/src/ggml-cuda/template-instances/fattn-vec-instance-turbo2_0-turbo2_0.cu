// TurboQuant2 CUDA flash attention vec kernel instantiation

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);

// Head dim 512 is streamed-only: block KV staging calls the partial kernel, never the regular
// vec case, and at D = 512 that case does not fit the register budget (see fattn-vec.cuh).
DECL_FATTN_VEC_PARTIAL_CASE(512, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
