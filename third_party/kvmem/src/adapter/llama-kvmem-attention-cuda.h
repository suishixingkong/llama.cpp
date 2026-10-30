#pragma once
#include "ggml.h"
#include <cstdint>
#include <memory>
#include <vector>

// Engine-owned sampled attention workspace. Uses represented Q and native K,
// without quantizing Q or materializing the token attention matrix.
class llama_kvmem_attention_cuda {
public:
    llama_kvmem_attention_cuda();
    ~llama_kvmem_attention_cuda();
    bool score(ggml_tensor* q,ggml_tensor* k,ggml_tensor* mask,float scale,float softcap,
        uint32_t block,const std::vector<uint32_t>& rows,std::vector<std::vector<float>>& output);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
