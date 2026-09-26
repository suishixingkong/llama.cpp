#include "server-cache-vbr-proof.h"
#include "../../src/llama-vbr-artifact-catalog.h"

#include <memory>
#include <utility>

server_cache_durable_fallback_proof
server_cache_vbr_fallback_proof(
        vbr_artifact_package_view && package) noexcept {
    if (!package || package.validate() != vbr_artifact_status::ok) {
        return server_cache_durable_fallback_proof_for_test(
            server_cache_lease_fallback_state::invalid, {});
    }
    try {
        auto owner = std::make_shared<vbr_artifact_package_view>(
            std::move(package));
        return server_cache_durable_fallback_proof_for_test(
            server_cache_lease_fallback_state::available,
            std::static_pointer_cast<void>(std::move(owner)));
    } catch (...) {
        return {};
    }
}

server_cache_durable_fallback_proof
server_cache_vbr_fallback_proof_for_test(
        vbr_artifact_package_view && package) noexcept {
    return server_cache_vbr_fallback_proof(std::move(package));
}
