#pragma once

#include "server-cache-control.h"

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <cstdint>
#include <unordered_map>

struct common_params;

// fork 9l-d4b9d: the reviewed-route boundary for the GCP /predict dispatcher
// (fork server-http.h 18-24).
// Vertex /predict is a compatibility dispatcher, not an alternate security
// boundary. Cache-plan preflight is intentionally reachable only through its
// reviewed route so its no-store and body-redaction contracts cannot be lost.
inline bool server_http_gcp_predict_dispatch_allowed(
        std::string_view path) noexcept {
    return path != "/cache/plan" && !server_cache_control_is_route(path);
}

// generator-like API for HTTP response generation
// this object response with one of the 2 modes:
// 1) normal response: `data` contains the full response body
// 2) streaming response: each call to next(output) generates the next chunk
//    when next(output) returns false, no more data after the current chunk
//    note: some chunks can be empty, in which case no data is sent for that chunk
struct server_http_res {
    std::string content_type = "application/json; charset=utf-8";
    int status = 200;
    std::string data;
    std::map<std::string, std::string> headers;

    std::function<bool(std::string &)> next = nullptr;
    bool is_stream() const {
        return next != nullptr;
    }

    // fired before req and res are destroyed
    virtual void on_complete() {}

    virtual ~server_http_res() = default;
};

// unique pointer, used by set_chunked_content_provider
// httplib requires the stream provider to be stored in heap
using server_http_res_ptr = std::unique_ptr<server_http_res>;
using raw_buffer = std::vector<uint8_t>;

struct uploaded_file {
    raw_buffer data;
    std::string filename;
    std::string content_type;
};

struct server_http_req {
    std::map<std::string, std::string> params; // path_params + query_params
    std::map<std::string, std::string> headers; // used by MCP proxy
    std::string path;
    std::string query_string; // query parameters string (e.g. "action=save")
    std::string body;
    std::map<std::string, uploaded_file> files; // used for file uploads (form data)
    const std::function<bool()> & should_stop;

    std::string get_param(const std::string & key, const std::string & def = "") const {
        auto it = params.find(key);
        if (it != params.end()) {
            return it->second;
        }
        return def;
    }

    // fork 9l-d4b7e: case-insensitive header lookup on the lowered request
    // (fork server-http.h 77-100).
    // Header field names are case-insensitive (RFC 9110). Keep this behavior
    // aligned with httplib::Request::get_header_value after the request is
    // lowered into the server-owned map.
    std::string get_header_value(const std::string & key) const {
        const auto ascii_lower = [](unsigned char c) {
            return c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c;
        };
        for (const auto & [name, value] : headers) {
            if (name.size() != key.size()) {
                continue;
            }
            bool equal = true;
            for (size_t i = 0; i < name.size(); ++i) {
                if (ascii_lower(name[i]) != ascii_lower(key[i])) {
                    equal = false;
                    break;
                }
            }
            if (equal) {
                return value;
            }
        }
        return {};
    }
};

struct server_http_context {
    class Impl;
    std::unique_ptr<Impl> pimpl;

    std::thread thread; // server thread
    std::atomic<bool> is_ready = false;

    // note: the handler should never throw exceptions
    using handler_t = std::function<server_http_res_ptr(const server_http_req & req)>;
    mutable std::unordered_map<std::string, handler_t> handlers;

    std::string path_prefix;
    std::string hostname;
    int port    = 8080;
    bool is_ssl = false;

    server_http_context();
    ~server_http_context();

    bool init(const common_params & params);
    bool start();
    void stop() const;

    void get(const std::string & path, const handler_t & handler) const;
    void post(const std::string & path, const handler_t & handler) const;
    void del(const std::string & path, const handler_t & handler) const;

    // Register the Google Cloud Platform (Vertex AI) compat (AIP_PREDICT_ROUTE env var, or /predict)
    // Must be called AFTER all other API routes are registered
    void register_gcp_compat() const;

    // for debugging
    std::string listening_address;
};
