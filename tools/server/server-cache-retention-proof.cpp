#include "server-cache-retention-proof.h"
#include "server-cache-destruction-quote.h"

#include <memory>
#include <utility>

server_cache_durable_fallback_proof
server_cache_retention_fallback_proof(
        server_cache_recovery_pin && pin) noexcept {
    if (!pin.valid()) {
        return server_cache_durable_fallback_proof_for_test(
            server_cache_lease_fallback_state::invalid, {});
    }
    try {
        auto owner = std::make_shared<server_cache_recovery_pin>(std::move(pin));
        return server_cache_durable_fallback_proof_for_test(
            server_cache_lease_fallback_state::available,
            std::static_pointer_cast<void>(std::move(owner)));
    } catch (...) {
        return {};
    }
}

server_cache_durable_fallback_proof
server_cache_retention_fallback_proof_for_test(
        server_cache_recovery_pin && pin) noexcept {
    return server_cache_retention_fallback_proof(std::move(pin));
}
