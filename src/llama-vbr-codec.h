#pragma once

#include "llama.h"

#include <cstddef>

// One immutable authority for the representation ladder. The controller owns
// ordering, budgets and transactions; this descriptor owns codec semantics.
struct llama_vbr_codec_ladder {
    const char * name;
    const ggml_type * rungs; // entry-to-bottom, including F16
    size_t n_rungs;
    ggml_type default_floor;
    bool needs_turbo_layout;
};

inline const llama_vbr_codec_ladder & llama_vbr_ladder(llama_vbr_codec codec) {
    static constexpr ggml_type turbo_rungs[] = {
        GGML_TYPE_F16,
        GGML_TYPE_TURBO8_0,
        GGML_TYPE_TURBO4_0,
        GGML_TYPE_TURBO3_TCQ,
        GGML_TYPE_TURBO2_TCQ,
        GGML_TYPE_TURBO1_TCQ,
    };
    // fork 9l-vbr-auto-ladder: `auto` is a ladder in its own right, not a parse-time placeholder
    // that fit later rewrites into turbo or classic. It interleaves the two standard-quant rungs
    // between the turbo ones so a descent step buys the cheapest representation that is still
    // strictly better than the one above it, instead of dropping straight from turbo8 to turbo4.
    // Descending bpv: 16 / 8.125 / 6 / 5 / 4.125 / 3.25.
    static constexpr ggml_type auto_rungs[] = {
        GGML_TYPE_F16,
        GGML_TYPE_TURBO8_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q4_1,
        GGML_TYPE_TURBO4_0,
        GGML_TYPE_TURBO3_TCQ,
    };
    // fork 9l-vbr-classic-ladder: q5_1/q4_1 instead of q4_0 gives classic three bands
    // (16 -> 8.5 -> 6 -> 5) rather than two, so its bottom is 5.0 bpv instead of 4.5.
    static constexpr ggml_type classic_rungs[] = {
        GGML_TYPE_F16,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q4_1,
    };
    static const llama_vbr_codec_ladder turbo = {
        "turbo", turbo_rungs,
        sizeof(turbo_rungs) / sizeof(turbo_rungs[0]), GGML_TYPE_TURBO1_TCQ, true,
    };
    static const llama_vbr_codec_ladder classic = {
        "classic", classic_rungs,
        sizeof(classic_rungs) / sizeof(classic_rungs[0]), GGML_TYPE_Q4_1, false,
    };
    static const llama_vbr_codec_ladder auto_l = {
        "auto", auto_rungs,
        sizeof(auto_rungs) / sizeof(auto_rungs[0]), GGML_TYPE_TURBO3_TCQ, true,
    };
    switch (codec) {
        case LLAMA_VBR_CODEC_TURBO:   return turbo;
        case LLAMA_VBR_CODEC_CLASSIC: return classic;
        case LLAMA_VBR_CODEC_AUTO:    return auto_l;
    }
    GGML_ABORT("invalid VBR codec: %d", int(codec));
}

inline bool llama_vbr_codec_valid(llama_vbr_codec codec) {
    return codec == LLAMA_VBR_CODEC_TURBO ||
           codec == LLAMA_VBR_CODEC_CLASSIC ||
           codec == LLAMA_VBR_CODEC_AUTO;
}

inline bool llama_vbr_codec_contains(llama_vbr_codec codec, ggml_type type) {
    const auto & ladder = llama_vbr_ladder(codec);
    for (size_t i = 0; i < ladder.n_rungs; ++i) {
        if (ladder.rungs[i] == type) {
            return true;
        }
    }
    return false;
}

inline double llama_vbr_type_bits_per_value(ggml_type type) {
    return 8.0 * ggml_type_size(type) / ggml_blck_size(type);
}

inline ggml_type llama_vbr_floor_price_type(llama_vbr_codec codec, double floor_bpv) {
    const auto & ladder = llama_vbr_ladder(codec);
    if (floor_bpv > 0.0) {
        for (size_t i = 0; i < ladder.n_rungs; ++i) {
            const ggml_type type = ladder.rungs[i];
            if (llama_vbr_type_bits_per_value(type) <= floor_bpv + 1e-9) {
                return type;
            }
        }
    }
    return ladder.default_floor;
}

// A rung whose rows live in the truncated "tapped" domain (mean-subtracted) rather than the full
// value domain. On every ladder those are exactly the turbo rungs below turbo8: f16 and turbo8
// carry the full range, the standard-quant rungs (q8_0/q5_1/q4_1) keep a full-domain
// representation, and turbo4 and below do not.
inline bool llama_vbr_codec_rung_is_tapped(ggml_type type) {
    return ggml_is_turbo_kv_type(type) &&
        llama_vbr_type_bits_per_value(type) < llama_vbr_type_bits_per_value(GGML_TYPE_TURBO8_0);
}

inline bool llama_vbr_codec_full_domain(llama_vbr_codec codec, ggml_type type) {
    if (codec == LLAMA_VBR_CODEC_CLASSIC) {
        return llama_vbr_codec_contains(codec, type);
    }
    if (codec == LLAMA_VBR_CODEC_AUTO) {
        // The auto ladder is mixed: its standard-quant rungs decode at full domain exactly as the
        // classic ladder's do, so only the turbo rungs below turbo8 are tapped. Derived from the
        // rung itself, so a rung change cannot leave this behind.
        return llama_vbr_codec_contains(codec, type) && !llama_vbr_codec_rung_is_tapped(type);
    }
    return type == GGML_TYPE_F16 || type == GGML_TYPE_TURBO8_0;
}

// Band index at which the ladder first reaches turbo4 -- the transition the hard seal protects
// ("crossing from the restorable t8 band into t4", VBR_HARD_SEAL_DEFAULT_FLOOR).
//
// A fixed band number cannot express this. On the pure turbo ladder turbo4 is band 1; on the mixed
// auto ladder it is band 3, so passing a literal VBR_TIER_T4 would silently move the seal from the
// 4.125 bpv crossing up to the 6.0 bpv one and seal two bands that need no sealing. Returns
// n_rungs when the ladder has no turbo4 rung (classic), which the caller reads as "nothing is
// sealed" -- the same answer the CLASSIC special case used to hardcode.
inline size_t llama_vbr_codec_hard_seal_tier(llama_vbr_codec codec) {
    const auto & ladder = llama_vbr_ladder(codec);
    for (size_t i = 1; i < ladder.n_rungs; ++i) {
        if (ladder.rungs[i] == GGML_TYPE_TURBO4_0) {
            return i - 1; // band i-1 is the step landing on rungs[i]
        }
    }
    return ladder.n_rungs;
}

// The measured VBR artefacts in this tree -- the baked per-arch degrade orders, --vbr-policy
// ladders, VBR_LAYER_SCHEDULE and the portable capture recipes -- were all measured on the pure
// turbo ladder, and each is keyed by BAND NUMBER. A band means something different on a different
// ladder (turbo band 1 is t8->t4; auto band 1 is t8->q5_1), so consuming them under another codec
// would not fail loudly -- it would silently change what was measured. Only the turbo ladder may
// read them; every other ladder falls back to the conservative generic order.
//
// This is a different axis from `needs_turbo_layout` above, which asks only whether the cache must
// be laid out for turbo rows. The auto ladder answers yes to that and no to this.
inline bool llama_vbr_codec_uses_turbo_artifacts(llama_vbr_codec codec) {
    return codec == LLAMA_VBR_CODEC_TURBO;
}
