#include "llama-vbr-codec.h"
#include "llama-bit-ops.h"
#include "llama-sha256.h"

// Anchor TU for the VBR port (buun-llama-cpp -> llama.cpp). The headers landed so far are
// header-only leaves of the subsystem; without a TU that includes them the build never
// checks them. No runtime behaviour is added here - nothing calls the anchor yet.
//
// Staging (BUUN_MERGE_PLAN.md section 3, feature 2 "VBR 动态 KV 量化"):
//   stage 2.1  leaves: bit-ops, sha256, codec ladder                      <- this commit
//   stage 2.2  cache accounting + cache authority + memory tree
//   stage 2.3  artifact core (writer/reader/validate)
//   stage 2.4  downward / upward / explicit-capture / swa-window / hard-seal
//   stage 2.5  CUDA backend interface (ggml-vbr.h + vbr-transcode + vbr-vmm)
//   stage 2.6  kv-cache / context / graph / memory wiring, `-ct vbr`, CLI

extern "C" int llama_vbr_codec_port_anchor(void) {
    const llama_vbr_codec_ladder & ladder = llama_vbr_ladder(LLAMA_VBR_CODEC_TURBO);
    int n = (int) ladder.n_rungs;

    // the codec descriptor and the shared bit helpers must agree on the ladder
    for (size_t i = 0; i < ladder.n_rungs; ++i) {
        n += llama_vbr_codec_contains(LLAMA_VBR_CODEC_TURBO, ladder.rungs[i]) ? 1 : 0;
    }
    return n;
}
