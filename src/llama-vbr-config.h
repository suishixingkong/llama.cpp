#pragma once

#include <string>

// Authoritative VBR_LAYER_SCHEDULE resolver shared by the runtime and calibration
// identity. Trims the environment value, accepts inline schedules plus @path/bare-path
// forms, and returns false only when a named file cannot be read.
bool llama_vbr_resolve_layer_schedule(
        const char * env,
        std::string & schedule,
        std::string & source);
