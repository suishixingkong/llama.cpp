#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

// Drafting and target verification have shape-dependent costs, and their batch
// kernels need not scale linearly. Measure intermediate depths even when full
// speculation beats target-only.
class common_speculative_dflash_adaptive {
public:
    struct calibration {
        struct sample { int64_t us = 0; int count = 0; };
        std::vector<sample> costs;
        void record(int depth, int64_t us) {
            if (depth >= (int) costs.size()) { costs.resize(depth + 1); }
            auto & s = costs[depth];
            s.us = s.count ? std::min(s.us, us) : us;
            s.count = std::min(s.count + 1, 3);
        }
    };

    int depth(int maximum, int minimum = 1, calibration * timing = nullptr) {
        hardware = timing;
        minimum = std::max(1, minimum);
        if (ceiling != maximum || floor != minimum) {
            ceiling = maximum;
            floor = minimum;
            selected = maximum;
            initial = true;
            start_sweep();
        }
        return cap;
    }

    bool searching() const { return sweep; }

    // Called once per completed cycle, excluding clipped drafts or proposals
    // supplied by another implementation. Emitted includes the target token and
    // any CopySpec extension; accepted counts only our own proposal prefix.
    void observe(int64_t elapsed_us, int drafted, int accepted, int emitted) {
        if (drafted != cap || emitted <= 0 || elapsed_us <= 0) {
            return;
        }
        if (hardware) { hardware->record(cap, elapsed_us); }
        accepts += accepted;
        saturated += cap > 0 && accepted == cap;
        ++cycles;
        for (int i = 1; i <= drafted && i <= accepted + 1; ++i) {
            ++attempts[i];
            matches[i] += accepted >= i;
        }
        if (sweep) {
            auto & sample = costs[candidate];
            // Graph/carry setup inflates the first cycles at a new shape. Rank
            // using its observed steady-cost envelope, not the startup penalty.
            sample.us = sample.cycles ? std::min(sample.us, elapsed_us) : elapsed_us;
            ++sample.cycles;
            sample.extra += std::max(0, emitted - accepted - 1);
            const bool timed = hardware && cap < (int) hardware->costs.size() && hardware->costs[cap].count >= 3;
            const int required = recovering && timed ? 1 : candidate == 0 ? 4 : 3;
            if (cycles < required) {
                return;
            }
            // High-acceptance full-depth drafting already amortizes verification
            // well. Avoid spending a short code response on exploration. This
            // exemption is only for the first window, not periodic rechecks or
            // a later phase change.
            if (initial && candidate == 0 && cap == ceiling && cap >= 7 &&
                    accepts >= cycles * cap * 0.8) {
                selected_acceptance = double(accepts) / (cycles * cap);
                initial = sweep = false;
                hold = 256;
                recovery_wait = 16;
                clear_sample();
                return;
            }
            // Once the middle probe yields at least five tokens per cycle,
            // don't spend a short response exploring tiny batches (whose best
            // possible yield is three). Later rechecks still test every shape.
            if (cap == 7 && ceiling > 7) {
                saturated_native = initial && accepts >= cycles * 4;
            }
            clear_sample();
            ++candidate;
            while (candidate < n_candidates && costs[candidate].cycles) {
                ++candidate;
            }
            if (saturated_native) {
                while (candidate < n_candidates && candidates[candidate] < 7) {
                    ++candidate;
                }
            }
            if (candidate < n_candidates) {
                cap = candidates[candidate];
            } else {
                choose();
                cap = selected;
                sweep = false;
                initial = false;
                hold = cap == 0 ? 32 : 256;
                recovery_wait = recovering && selected == candidates[0] ? 256 : cap >= 7 ? 8 : 16;
            }
            return;
        }

        --hold;
        if (recovery_wait > 0) { --recovery_wait; }
        // Phase changes deserve a fresh comparison, not an unconditional jump
        // to full depth. Periodic exploration also recovers from target-only.
        bool changed = false;
        bool probe_up = false;
        if (cycles >= 8) {
            const double acceptance = cap ? double(accepts) / (cycles * cap) : 0.0;
            changed = cap > 0 && acceptance <= 0.50 && selected_acceptance > 0.80;
            // A shallow cap censors the later positions: even unchanged perfect
            // prefix acceptance can conceal a newly useful longer proposal.
            const bool newly_high_match = acceptance >= 0.90 && selected_acceptance < 0.80;
            if (cap > 0 && cap < ceiling && saturated > 0 && (recovery_wait == 0 || newly_high_match)) {
                const int next = cap < 7 && ceiling > 7 && floor <= 7 ? 7 : ceiling;
                const double current_us = measured_us(cap);
                const double next_us = measured_us(next);
                const double emitted = 1.0 + double(accepts) / cycles;
                // Only fully accepted prefixes can gain from an unseen tail.
                // Probe when even this optimistic bound can repay the wider
                // verification. Known shape costs need only an acceptance probe.
                const double upper_emitted = emitted + double(saturated) / cycles * (next - cap);
                probe_up = next_us == 0 || current_us == 0 ||
                    next_us / upper_emitted < 0.97 * current_us / emitted;
            }
            choose();
            cap = selected;
            if (cap == 0) { hold = std::min(hold, 32); }
            for (int i = 1; i <= ceiling; ++i) {
                attempts[i] *= 0.5;
                matches[i] *= 0.5;
            }
            prior_decay *= 0.5;
            clear_sample();
        }
        if (hold <= 0 || changed) {
            start_sweep();
        } else if (probe_up && selected < ceiling) {
            start_sweep(true);
        }
    }

private:
    double measured_us(int depth) const {
        if (hardware && depth < (int) hardware->costs.size() && hardware->costs[depth].count >= 3) {
            return hardware->costs[depth].us;
        }
        for (int j = 0; j < n_candidates; ++j) {
            if (candidates[j] == depth && costs[j].cycles) { return costs[j].us; }
        }
        return 0;
    }

    void choose() {
        const int incumbent = selected;
        const int middle = ceiling > 7 && floor <= 7 ? 7 : ceiling > 2 && floor <= 2 ? 2 : ceiling;
        const double full_us = measured_us(ceiling);
        const double middle_us = measured_us(middle);
        double prior = 0;
        if (middle < ceiling && full_us > 0 && middle_us > 0) {
            // Conservatism should reflect the cost of a wrong short-depth
            // decision. If middle is faster even with perfect full acceptance,
            // an unseen long tail cannot rescue full depth. As that worst-case
            // penalty grows, give the unseen tail a weak optimistic allowance.
            const double dominance = double(middle + 1) / (ceiling + 1);
            const double risk = (middle_us / full_us - dominance) / (1.0 - dominance);
            prior = 2 * prior_decay * std::clamp(risk, 0.0, 1.0);
        }
        const int common_prefix = floor <= 2 ? 2 : floor <= 7 ? 7 : ceiling;
        double best_cost = std::numeric_limits<double>::infinity();
        double best_acceptance = 0.0;
        auto consider = [&](int j) {
            const auto & sample = costs[j];
            if (sample.cycles == 0) { return; }
            double accepted = 0.0;
            double survival = 1.0;
            // Estimate conditional survival at each position. A shallow probe
            // updates the common prefix without treating an unobserved tail as
            // a rejection. Later-position observations remain conditional on
            // reaching that position, so improving prefixes can lift the tail.
            for (int i = 1; i <= candidates[j]; ++i) {
                const double allowance = i > common_prefix ? prior : 0;
                const double n = attempts[i] + allowance;
                survival *= n ? (matches[i] + allowance) / n : 0.0;
                accepted += survival;
            }
            const double emitted = 1.0 + accepted + double(sample.extra) / sample.cycles;
            const double cost = double(sample.us) / emitted;
            if (cost < best_cost * 0.97) {
                best_cost = cost;
                selected = candidates[j];
                best_acceptance = selected ? accepted / selected : 0.0;
            }
        };
        // Prefer the current choice inside the noise band even after a rescore.
        for (int j = 0; j < n_candidates; ++j) {
            if (candidates[j] == incumbent) { consider(j); }
        }
        for (int j = 0; j < n_candidates; ++j) { consider(j); }
        if (sweep || selected != incumbent) {
            selected_acceptance = best_acceptance;
        }
    }

    void clear_sample() {
        cycles = accepts = saturated = 0;
    }

    void start_sweep(bool upwards = false) {
        recovering = upwards;
        cost_sample incumbent_cost{};
        if (upwards) {
            for (int j = 0; j < n_candidates; ++j) {
                if (candidates[j] == selected) { incumbent_cost = costs[j]; }
            }
        }
        n_candidates = 0;
        auto add = [&](int value) {
            if (value < 0 || value > ceiling || (value > 0 && value < floor) ||
                    std::find(candidates.begin(), candidates.begin() + n_candidates, value) !=
                    candidates.begin() + n_candidates) {
                return;
            }
            candidates[n_candidates++] = value;
        };
        // Seven proposals is the native eight-position DFlash2 block. Two
        // covers low-match verification; the caller's ceiling is always
        // measured, including when it is below those landmarks.
        add(selected);
        if (upwards) {
            add(selected < 7 && ceiling > 7 && floor <= 7 ? 7 : ceiling);
        } else {
            add(ceiling);
            add(7);
            add(2);
            // Preserve the existing persistent choices: target-only was a
            // calibration arm, not an off-ramp. Zero is only for an explicit
            // zero ceiling or incompatible minimum/maximum constraints.
            if (n_candidates == 0) { add(0); }
        }
        // The hold window already supplied observations of the incumbent. An
        // upward probe should measure the unseen tail immediately, not spend
        // another four cycles at the depth we are trying to leave.
        candidate = upwards ? 1 : 0;
        cap = candidates[candidate];
        saturated_native = false;
        sweep = true;
        costs = {};
        if (upwards) {
            costs[0] = incumbent_cost;
        } else {
            attempts.assign(ceiling + 1, 0);
            matches.assign(ceiling + 1, 0);
            prior_decay = 1;
            if (hardware) {
                for (int j = 0; j < n_candidates; ++j) {
                    const int d = candidates[j];
                    if (d < (int) hardware->costs.size() && hardware->costs[d].count >= 3) {
                        costs[j].us = hardware->costs[d].us;
                        costs[j].cycles = 1;
                    }
                }
            }
        }
        clear_sample();
    }

    std::array<int, 6> candidates{};
    calibration * hardware = nullptr;
    struct cost_sample { int64_t us = 0; int cycles = 0, extra = 0; };
    std::array<cost_sample, 6> costs{};
    std::vector<double> attempts, matches;
    int ceiling = -1, floor = 1, cap = 0, selected = 0;
    int candidate = 0, n_candidates = 0, hold = 0;
    bool sweep = false;
    bool initial = true;
    bool recovering = false;
    int recovery_wait = 0;
    bool saturated_native = false;
    int cycles = 0, accepts = 0, saturated = 0;
    double selected_acceptance = 0.0;
    double prior_decay = 1.0;
};
