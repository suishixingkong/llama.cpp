#pragma once

#include "ggml-backend.h"

// Order every consumer on this device without blocking host preparation.
inline bool kvmem_order_completion(ggml_backend_sched_t sched,
        ggml_backend_dev_t device, ggml_backend_event_t completion) {
    if (!sched || !device || !completion) return false;
    bool ordered = false;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        auto * backend = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_get_device(backend) == device) {
            ggml_backend_event_wait(backend, completion);
            ordered = true;
        }
    }
    return ordered;
}
