// TurboQuant TCQ CUDA flash attention vec kernel instantiation (K=Q8_0, V=TURBO3_TCQ)

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_TCQ);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_TCQ);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_TCQ);

// Head dim 512 is streamed-only: block KV staging calls the partial kernel, never the regular
// vec case, and at D = 512 that case does not fit the register budget (see fattn-vec.cuh).
DECL_FATTN_VEC_PARTIAL_CASE(512, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_TCQ);
