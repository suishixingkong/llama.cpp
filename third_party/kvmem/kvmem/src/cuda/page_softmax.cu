#include "kvmem/page_softmax_cuda.hpp"
#include "kvmem/raw_kv_store.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace kvmem {
namespace {
void checked(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("page softmax CUDA: ")+cudaGetErrorString(error));
}
struct DeviceBinding {
    int previous;
    explicit DeviceBinding(int device) { checked(cudaGetDevice(&previous)); checked(cudaSetDevice(device)); }
    ~DeviceBinding() { cudaSetDevice(previous); }
};
struct Buffer {
    void* data = nullptr;
    size_t capacity = 0;
    bool borrowed = false;
    void reserve(size_t bytes) {
        if (bytes <= capacity) return;
        if(borrowed) throw std::length_error("page softmax exceeds its planned device workspace");
        void* next = nullptr; checked(cudaMalloc(&next, bytes));
        if (data) cudaFree(data);
        data = next; capacity = bytes;
    }
    ~Buffer() { if (data && !borrowed) cudaFree(data); }
};
constexpr size_t logits_limit=64u<<20;
std::array<size_t,6> planned_sizes(uint32_t blocks,uint32_t heads,uint32_t kvheads,uint32_t dim) {
    if(!blocks || !heads || !kvheads || !dim || heads%kvheads ||
       uint64_t(blocks)*heads*sizeof(float)>logits_limit)
        throw std::invalid_argument("invalid planned page softmax geometry");
    // 512 rows a batch keeps the stream cadence low without pressing the logits
    // budget: at 9.5k rows the old 64-row batch meant 148 sync points per layer.
    const auto batch=std::min<size_t>(512,logits_limit/(size_t(blocks)*heads*sizeof(float)));
    // The Query is staged as float: the device does the FP16 conversion (see
    // page_logits), and a 2x upload is cheaper than converting it on the host.
    return {batch*heads*dim*sizeof(float),size_t(blocks)*kvheads*dim*sizeof(half),
        batch*blocks*heads*sizeof(float),batch*heads*2*sizeof(float),size_t(blocks)*sizeof(float),
        sizeof(int)};
}
size_t aligned(size_t bytes) {
    if(bytes>SIZE_MAX-255) throw std::length_error("page softmax workspace overflow");
    return (bytes+255)/256*256;
}
__device__ float warp_sum(float value) {
    for (int offset=16;offset;offset/=2) value += __shfl_down_sync(0xffffffff,value,offset);
    return value;
}
// One warp per (row, head) distribution, looping the pages. The previous shape gave
// every warp a single page, so a 4k-token Query at a 19-page budget launched
// rows*heads*ceil(pages/4) = 20k blocks per layer that each did four FMAs and a
// shuffle reduction -- launch-bound, and it re-read the row's Query vector once per
// page group. Here the Query is read once per distribution and each page costs the
// same dot it did before, so the logits are bit-identical.
//
// The Query arrives as float and is converted here. Converting it on the host cost
// 70 ms of a 77 ms scoring pass on a 4058-token prompt: MSVC compiles
// __float2half_rn to a software routine when the translation unit has no F16C, and
// the pass ran over rows*heads*dim = 14.7M elements twice (a range scan, then the
// conversion). The device converts the same values with the same rounding, one
// register at a time, and `bad` keeps the range rejection that the host pass used
// to perform -- recall_fp16() rounds to nearest even too, and its only other job was
// to throw on operands a bare conversion cannot represent.
__global__ void page_logits(const float* queries, const half* means, float* logits,
                            int* bad, int distributions, int blocks, int qheads,
                            int kvheads, int dim) {
    const int distribution=int(blockIdx.x)*4+int(threadIdx.x)/32;
    const int lane=int(threadIdx.x)%32;
    if (distribution >= distributions) return;
    const int head=distribution%qheads;
    const int kvhead=head/(qheads/kvheads);
    const auto* q=queries+size_t(distribution)*dim;
    for (int d=lane;d<dim;d+=32) {
        if (!(fabsf(q[d]) <= 65504.0f)) {
            // NaN and anything past the FP16 maximum. Any lane may report it; a
            // plain store is enough because the host only tests "not zero".
            *bad = 1;
        }
    }
    for (int page=0;page<blocks;++page) {
        const auto* k=means+(size_t(page)*kvheads+kvhead)*dim;
        float dot=0;
        for (int d=lane;d<dim;d+=32) dot=fmaf(__half2float(__float2half_rn(q[d])),__half2float(k[d]),dot);
        dot=warp_sum(dot);
        if (lane==0) logits[size_t(distribution)*blocks+page]=dot*rsqrtf(float(dim));
    }
}
__global__ void normalize_pages(const float* logits, float* norms, int blocks) {
    __shared__ float reduction[128];
    const int tid=int(threadIdx.x), row=int(blockIdx.x);
    float maximum=-CUDART_INF_F;
    for (int b=tid;b<blocks;b+=128) maximum=fmaxf(maximum,logits[size_t(row)*blocks+b]);
    reduction[tid]=maximum; __syncthreads();
    for (int s=64;s;s/=2) { if(tid<s) reduction[tid]=fmaxf(reduction[tid],reduction[tid+s]); __syncthreads(); }
    maximum=reduction[0];
    __syncthreads();
    float sum=0;
    for (int b=tid;b<blocks;b+=128) sum+=expf(logits[size_t(row)*blocks+b]-maximum);
    reduction[tid]=sum; __syncthreads();
    for (int s=64;s;s/=2) { if(tid<s) reduction[tid]+=reduction[tid+s]; __syncthreads(); }
    if(tid==0) { norms[2*row]=maximum; norms[2*row+1]=1/reduction[0]; }
}
__global__ void accumulate_pages(const float* logits, const float* norms, float* output,
                                 int blocks, int distributions, float weight) {
    const int b=int(blockIdx.x)*128+int(threadIdx.x);
    if(b>=blocks) return;
    float sum=0;
    for(int row=0;row<distributions;++row)
        sum+=expf(logits[size_t(row)*blocks+b]-norms[2*row])*norms[2*row+1];
    output[b]+=sum*weight;
}
// Independent groups own disjoint original head rows. All head probabilities
// are read before their group's first row is overwritten with max probabilities.
// This reuses the existing logits workspace without a second full tensor.
__global__ void collapse_gqa_max(float* logits, float* norms, int blocks, int group) {
    const int base = int(blockIdx.x) * group;
    const int tid = int(threadIdx.x);
    __shared__ float reduction[128];
    float sum = 0;
    for (int page = tid; page < blocks; page += 128) {
        float maximum = 0;
        for (int h = 0; h < group; ++h) {
            const int row = base + h;
            maximum = fmaxf(maximum, expf(logits[size_t(row)*blocks+page]-norms[2*row])*norms[2*row+1]);
        }
        logits[size_t(base)*blocks+page] = maximum;
        sum += maximum;
    }
    reduction[tid] = sum;
    __syncthreads();
    for (int s=64; s; s/=2) {
        if (tid<s) reduction[tid] += reduction[tid+s];
        __syncthreads();
    }
    // All reads of the old softmax norms in this group have now completed.
    if (tid==0) norms[2*base] = 1/reduction[0];
}
__global__ void accumulate_gqa_max(const float* logits, const float* norms, float* output,
                                  int blocks, int groups, int group, float weight) {
    const int page = int(blockIdx.x)*128+int(threadIdx.x);
    if (page>=blocks) return;
    float sum = 0;
    for (int g=0; g<groups; ++g) {
        const int base = g*group;
        sum += logits[size_t(base)*blocks+page]*norms[2*base];
    }
    output[page] += sum*weight;
}

} // namespace

struct PageSoftmaxCuda::Impl {
    int device;
    cudaStream_t stream=nullptr;
    Buffer queries,means,logits,norms,output,bad;
    uint32_t max_batch=512;
    explicit Impl(int device):device(device) {
        DeviceBinding binding(device); checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    }
    ~Impl() {
        DeviceBinding binding(device);
        cudaStreamSynchronize(stream);
        // Release in the owning context, before member destruction restores it.
        for(auto* buffer:{&queries,&means,&logits,&norms,&output,&bad}) {
            if(buffer->data && !buffer->borrowed) cudaFree(buffer->data);
            buffer->data=nullptr; buffer->capacity=0;
        }
        cudaStreamDestroy(stream);
    }
};
PageSoftmaxCuda::PageSoftmaxCuda(int device):impl_(std::make_unique<Impl>(device)) {}
size_t PageSoftmaxCuda::workspace_bytes(uint32_t blocks,uint32_t heads,uint32_t kvheads,uint32_t dim) {
    size_t bytes=0;
    for(auto size:planned_sizes(blocks,heads,kvheads,dim)) {
        if(bytes>SIZE_MAX-aligned(size)) throw std::length_error("page softmax workspace overflow");
        bytes+=aligned(size);
    }
    return bytes;
}
PageSoftmaxCuda::PageSoftmaxCuda(int device,void* workspace,size_t bytes,uint32_t blocks,
    uint32_t heads,uint32_t kvheads,uint32_t dim):PageSoftmaxCuda(device) {
    const auto sizes=planned_sizes(blocks,heads,kvheads,dim);
    if(!workspace || reinterpret_cast<uintptr_t>(workspace)%256 || bytes<workspace_bytes(blocks,heads,kvheads,dim))
        throw std::invalid_argument("page softmax borrowed workspace is missing, unaligned or too small");
    auto* begin=static_cast<uint8_t*>(workspace);size_t at=0;size_t index=0;
    for(auto* buffer:{&impl_->queries,&impl_->means,&impl_->logits,&impl_->norms,&impl_->output,&impl_->bad}) {
        buffer->data=begin+at;buffer->capacity=sizes[index++];buffer->borrowed=true;
        at+=aligned(buffer->capacity);
    }
    impl_->max_batch=uint32_t(std::min<size_t>(512,logits_limit/(size_t(blocks)*heads*sizeof(float))));
}
PageSoftmaxCuda::~PageSoftmaxCuda()=default;
size_t PageSoftmaxCuda::device_bytes() const {
    const auto& w=*impl_;
    return w.queries.capacity+w.means.capacity+w.logits.capacity+w.norms.capacity+w.output.capacity+
        w.bad.capacity;
}

std::vector<double> PageSoftmaxCuda::score(const RawKvStore& keys, PageQueryView q,
                                         const QueryScoreFrame& frame, GqaScoreAggregation aggregation) {
    if (aggregation != GqaScoreAggregation::Mean && aggregation != GqaScoreAggregation::Max)
        throw std::invalid_argument("invalid page softmax CUDA aggregation");
    if ((!q.data && !q.layer_data) || !q.layer_rows || !q.rows || !q.layers || !q.head_dim || !q.kv_heads ||
        !q.query_heads || q.query_heads%q.kv_heads || q.layers!=keys.config().n_layer ||
        uint64_t(q.kv_heads)*q.head_dim!=keys.config().n_embd_k ||
        frame.starts.size()!=frame.counts.size() || frame.starts.size()!=frame.score_mask.size())
        throw std::invalid_argument("page softmax CUDA geometry mismatch");
    std::vector<size_t> enabled;
    size_t masked=0;
    const auto block_tokens=keys.config().block_tokens;
    for(size_t i=0;i<frame.starts.size();++i) {
        if(!frame.score_mask[i]) continue;
        ++masked;
        const auto begin=frame.starts[i];
        if(!block_tokens || begin%block_tokens || begin>=frame.accepted_through ||
            !frame.counts[i] || frame.counts[i]>frame.accepted_through-begin ||
            begin/block_tokens>UINT32_MAX)
            throw std::invalid_argument("page softmax CUDA stale index extent");
        // A page the index never accepted has no evidence to score, and the host
        // scorers treat exactly that as "not a candidate" (mean_key_retrieval.cpp
        // continues on a missing block). Failing the round instead costs the whole
        // retrieval pass for one hole: the 2026-11-09 field log threw on all 318
        // rounds of a 160k conversation and silently fell back to recency. Such a
        // page keeps its 0 and stays out of the softmax denominator.
        if(!keys.has_block(uint32_t(begin/block_tokens))) continue;
        enabled.push_back(i);
    }
    std::vector<double> result(frame.starts.size(),0);
    uint32_t active=0;
    for(uint32_t l=0;l<q.layers;++l) {
        if(q.layer_rows[l]>q.rows) throw std::invalid_argument("page softmax CUDA row coverage mismatch");
        active+=q.layer_rows[l]!=0;
    }
    if(enabled.empty() && masked)
        throw std::invalid_argument("page softmax CUDA found no page with an accepted block");
    if(enabled.empty() || !active) return result;
    if(enabled.size()>INT_MAX || q.query_heads>INT_MAX || q.head_dim>INT_MAX)
        throw std::length_error("page softmax CUDA geometry exceeds launch addressing");
    const size_t blocks=enabled.size(),kw=size_t(q.kv_heads)*q.head_dim,qw=size_t(q.query_heads)*q.head_dim;
    constexpr size_t max_logits=64u<<20;
    const size_t row_bytes=blocks*q.query_heads*sizeof(float);
    if(row_bytes>max_logits) throw std::length_error("page softmax CUDA needs a larger per-row logits budget");
    const auto batch=uint32_t(std::min<size_t>(impl_->max_batch,max_logits/row_bytes));
    auto& w=*impl_; DeviceBinding binding(w.device);
    checked(cudaStreamSynchronize(w.stream));
    w.queries.reserve(size_t(batch)*qw*sizeof(float));
    w.means.reserve(blocks*kw*sizeof(half));
    w.logits.reserve(size_t(batch)*row_bytes);
    w.norms.reserve(size_t(batch)*q.query_heads*2*sizeof(float));
    w.output.reserve(blocks*sizeof(float));
    w.bad.reserve(sizeof(int));
    checked(cudaMemsetAsync(w.output.data,0,blocks*sizeof(float),w.stream));
    std::vector<half> host_means(blocks*kw);
    std::vector<float> key(kw),scores(blocks);
    // One flag for the whole pass: any Query operand that a half cannot hold.
    checked(cudaMemsetAsync(w.bad.data,0,sizeof(int),w.stream));
    for(uint32_t layer=0;layer<q.layers;++layer) {
        if(!q.layer_rows[layer]) continue;
        for(size_t j=0;j<blocks;++j) {
            const auto i=enabled[j];
            keys.mean_k(uint32_t(frame.starts[i]/block_tokens),layer,key.data());
            for(size_t d=0;d<kw;++d) host_means[j*kw+d]=__float2half_rn(recall_fp16(key[d]));
        }
        checked(cudaMemcpyAsync(w.means.data,host_means.data(),blocks*kw*sizeof(half),cudaMemcpyHostToDevice,w.stream));
        for(uint32_t first=0;first<q.layer_rows[layer];first+=batch) {
            const auto rows=std::min(batch,q.layer_rows[layer]-first);
            const auto* base=q.layer_data ? q.layer_data[layer] : q.data+size_t(layer)*q.rows*qw;
            if(!base) throw std::invalid_argument("page softmax CUDA missing active layer storage");
            const auto* input=base+size_t(first)*qw;
            const int distributions=int(rows*q.query_heads);
            checked(cudaMemcpyAsync(w.queries.data,input,size_t(rows)*qw*sizeof(float),cudaMemcpyHostToDevice,w.stream));
            page_logits<<<unsigned((distributions+3)/4),128,0,w.stream>>>(
                static_cast<const float*>(w.queries.data),static_cast<const half*>(w.means.data),
                static_cast<float*>(w.logits.data),static_cast<int*>(w.bad.data),
                distributions,int(blocks),int(q.query_heads),int(q.kv_heads),int(q.head_dim));
            normalize_pages<<<distributions,128,0,w.stream>>>(static_cast<float*>(w.logits.data),static_cast<float*>(w.norms.data),int(blocks));
            if (aggregation == GqaScoreAggregation::Mean) {
                accumulate_pages<<<unsigned((blocks+127)/128),128,0,w.stream>>>(static_cast<float*>(w.logits.data),
                    static_cast<float*>(w.norms.data),static_cast<float*>(w.output.data),int(blocks),distributions,1.f/(active*q.query_heads));
            } else {
                const int groups=int(rows*q.kv_heads),group=int(q.query_heads/q.kv_heads);
                collapse_gqa_max<<<groups,128,0,w.stream>>>(static_cast<float*>(w.logits.data),
                    static_cast<float*>(w.norms.data),int(blocks),group);
                accumulate_gqa_max<<<unsigned((blocks+127)/128),128,0,w.stream>>>(static_cast<float*>(w.logits.data),
                    static_cast<float*>(w.norms.data),static_cast<float*>(w.output.data),int(blocks),groups,group,1.f/(active*q.kv_heads));
            }
            checked(cudaGetLastError());
            // Staging arrays are reused only after their asynchronous reads finish.
            checked(cudaStreamSynchronize(w.stream));
        }
    }
    checked(cudaMemcpyAsync(scores.data(),w.output.data,blocks*sizeof(float),cudaMemcpyDeviceToHost,w.stream));
    checked(cudaStreamSynchronize(w.stream));
    int bad=0;
    checked(cudaMemcpy(&bad,w.bad.data,sizeof(int),cudaMemcpyDeviceToHost));
    if(bad) throw std::runtime_error(
        "page softmax CUDA query operand is nonfinite or exceeds the FP16 range");
    for(size_t j=0;j<blocks;++j) result[enabled[j]]=scores[j];
    return result;
}
} // namespace kvmem
