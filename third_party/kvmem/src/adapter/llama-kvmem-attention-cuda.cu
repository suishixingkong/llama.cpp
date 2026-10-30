#include "llama-kvmem-attention-cuda.h"
#include "ggml-backend.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace kvmem_attention_cuda_detail {
void checked(cudaError_t error) {
    if(error!=cudaSuccess) throw std::runtime_error(std::string("KVMem attention CUDA: ")+cudaGetErrorString(error));
}
struct DeviceGuard {
    int previous;
    explicit DeviceGuard(int device) {checked(cudaGetDevice(&previous));checked(cudaSetDevice(device));}
    ~DeviceGuard(){cudaSetDevice(previous);}
};
struct Buffer {
    void* data=nullptr;size_t capacity=0;
    ~Buffer(){if(data) cudaFree(data);}
    void reserve(size_t bytes) {
        if(bytes<=capacity) return;
        void* next=nullptr;checked(cudaMalloc(&next,bytes));
        if(data) checked(cudaFree(data));
        data=next;capacity=bytes;
    }
};
struct Q8 {half scale;int8_t codes[32];};
struct Q4 {half scale;uint8_t codes[16];};
struct Q5 {half scale;uint8_t high[4],codes[16];};
static_assert(sizeof(Q8)==34 && sizeof(Q4)==18 && sizeof(Q5)==22,"native ggml block layout");
__device__ float scalar(const void* data,int type,int d) {
    if(type==GGML_TYPE_F32) return static_cast<const float*>(data)[d];
    if(type==GGML_TYPE_F16) return __half2float(static_cast<const half*>(data)[d]);
    return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[d]);
}
__device__ float key(const uint8_t* data,int type,int d) {
    const int at=d%32;
    if(type==GGML_TYPE_Q8_0) {
        const auto& block=reinterpret_cast<const Q8*>(data)[d/32];
        return float(block.codes[at])*__half2float(block.scale);
    }
    if(type==GGML_TYPE_Q4_0) {
        const auto& block=reinterpret_cast<const Q4*>(data)[d/32];
        const int code=(block.codes[at%16]>>(at<16?0:4))&15;
        return float(code-8)*__half2float(block.scale);
    }
    if(type==GGML_TYPE_Q5_0) {
        const auto& block=reinterpret_cast<const Q5*>(data)[d/32];
        const int low=(block.codes[at%16]>>(at<16?0:4))&15;
        const int high=(block.high[at/8]>>(at%8))&1;
        return float((low|(high<<4))-16)*__half2float(block.scale);
    }
    return scalar(data,type,d);
}
__global__ void page_lse(const uint8_t* q,int qtype,size_t qrow,size_t qhead,
    const uint8_t* k,int ktype,size_t krow,size_t khead,const uint8_t* mask,int mtype,
    size_t mrow,const uint32_t* rows,bool staged_mask,int dim,int heads,int kvheads,
    int tokens,int pages,int block,float scale,float softcap,float* output) {
    __shared__ float maxima[4],sums[4];
    const int tid=int(threadIdx.x),lane=tid%32,warp=tid/32;
    const int page=int(blockIdx.x),distribution=int(blockIdx.y),head=distribution%heads,sample=distribution/heads;
    const auto* query=q+size_t(rows[sample])*qrow+size_t(head)*qhead;
    const auto* causal=mask+size_t(staged_mask?sample:rows[sample])*mrow;
    float query_values[8];
    for(int i=0;i<8;++i) query_values[i]=lane+32*i<dim ? scalar(query,qtype,lane+32*i):0;
    float maximum=-CUDART_INF_F,sum=0;
    for(int offset=warp;offset<block;offset+=4) {
        const int token=page*block+offset;
        const float bias=token<tokens ? scalar(causal,mtype,token):-CUDART_INF_F;
        float dot=0;
        if(isfinite(bias)) {
            const auto* native=k+size_t(token)*krow+size_t(head/(heads/kvheads))*khead;
            for(int i=0;i<8;++i) if(lane+32*i<dim)
                dot=fmaf(query_values[i],key(native,ktype,lane+32*i),dot);
            for(int s=16;s;s/=2) dot+=__shfl_down_sync(0xffffffff,dot,s);
        }
        if(lane==0 && isfinite(bias)) {
            dot*=scale;
            if(softcap>0) dot=softcap*tanhf(dot/softcap);
            dot+=bias;
            const float next=fmaxf(maximum,dot);
            sum=sum*expf(maximum-next)+expf(dot-next);maximum=next;
        }
    }
    if(lane==0) {maxima[warp]=maximum;sums[warp]=sum;}
    __syncthreads();
    if(tid==0) {
        float maximum=-CUDART_INF_F,total=0;
        for(int w=0;w<4;++w) maximum=fmaxf(maximum,maxima[w]);
        for(int w=0;w<4;++w) if(sums[w]>0) total+=sums[w]*expf(maxima[w]-maximum);
        output[size_t(distribution)*pages+page]=total>0 ? maximum+logf(total):-CUDART_INF_F;
    }
}
__global__ void normalize(const float* lse,int pages,float* norms) {
    __shared__ float values[128];
    const int tid=int(threadIdx.x),row=int(blockIdx.x);
    float maximum=-CUDART_INF_F;
    for(int p=tid;p<pages;p+=128) maximum=fmaxf(maximum,lse[size_t(row)*pages+p]);
    values[tid]=maximum;__syncthreads();
    for(int s=64;s;s/=2) {if(tid<s) values[tid]=fmaxf(values[tid],values[tid+s]);__syncthreads();}
    maximum=values[0];__syncthreads();
    float sum=0;
    for(int p=tid;p<pages;p+=128) sum+=expf(lse[size_t(row)*pages+p]-maximum);
    values[tid]=sum;__syncthreads();
    for(int s=64;s;s/=2) {if(tid<s) values[tid]+=values[tid+s];__syncthreads();}
    if(tid==0) {norms[2*row]=maximum;norms[2*row+1]=1/values[0];}
}
__global__ void average(const float* lse,const float* norms,int pages,int heads,float* output) {
    const int page=int(blockIdx.x)*128+int(threadIdx.x),sample=int(blockIdx.y);
    if(page>=pages) return;
    float sum=0;
    for(int h=0;h<heads;++h) {
        const int row=sample*heads+h;
        sum+=expf(lse[size_t(row)*pages+page]-norms[2*row])*norms[2*row+1];
    }
    output[size_t(sample)*pages+page]=sum/heads;
}
bool scalar_type(ggml_type type) {return type==GGML_TYPE_F32 || type==GGML_TYPE_F16 || type==GGML_TYPE_BF16;}
bool device_pointer(const void* data,int& device) {
    cudaPointerAttributes attr{};
    if(cudaPointerGetAttributes(&attr,data)!=cudaSuccess) {cudaGetLastError();return false;}
    if(attr.type!=cudaMemoryTypeDevice && attr.type!=cudaMemoryTypeManaged) return false;
    device=attr.device;return true;
}
} // namespace kvmem_attention_cuda_detail

struct llama_kvmem_attention_cuda::Impl {
    int device=-1;cudaStream_t stream=nullptr;
    kvmem_attention_cuda_detail::Buffer indices,mask,lse,norms,output;
    ~Impl() {
        if(device<0) return;
        kvmem_attention_cuda_detail::DeviceGuard guard(device);
        if(stream) cudaStreamDestroy(stream);
        // Buffers are destroyed after this body; their device must remain selected.
        for(auto* buffer:{&indices,&mask,&lse,&norms,&output}) {
            if(buffer->data) cudaFree(buffer->data);buffer->data=nullptr;
        }
    }
};
llama_kvmem_attention_cuda::llama_kvmem_attention_cuda():impl(std::make_unique<Impl>()){}
llama_kvmem_attention_cuda::~llama_kvmem_attention_cuda()=default;
bool llama_kvmem_attention_cuda::score(ggml_tensor* q,ggml_tensor* k,ggml_tensor* mask,
    float scale,float softcap,uint32_t block,const std::vector<uint32_t>& rows,
    std::vector<std::vector<float>>& output) {
    using namespace kvmem_attention_cuda_detail;
    if(!scalar_type(q->type) || !scalar_type(mask->type) || q->ne[0]>256 || q->ne[0]%32 ||
        !(scalar_type(k->type) || k->type==GGML_TYPE_Q8_0 || k->type==GGML_TYPE_Q5_0 || k->type==GGML_TYPE_Q4_0)) return false;
    int device=-1,key_device=-1,mask_device=-1;
    if(!device_pointer(q->data,device) || !device_pointer(k->data,key_device) || device!=key_device) return false;
    DeviceGuard guard(device);
    if(impl->device<0) {impl->device=device;checked(cudaStreamCreateWithFlags(&impl->stream,cudaStreamNonBlocking));}
    if(impl->device!=device) throw std::runtime_error("KVMem attention workspace changed device");
    const int pages=int((k->ne[1]+block-1)/block),heads=int(q->ne[2]);
    const size_t mask_row=ggml_row_size(mask->type,mask->ne[0]);
    const bool staged=!device_pointer(mask->data,mask_device) || mask_device!=device;
    const size_t per_sample=size_t(pages)*heads*sizeof(float)+(staged?mask_row:0)+size_t(heads)*2*sizeof(float)+size_t(pages)*sizeof(float)+sizeof(uint32_t);
    if(per_sample>(32u<<20)) throw std::length_error("KVMem attention sample exceeds its workspace budget");
    const auto batch=std::max<size_t>(1,std::min<size_t>(64,(32u<<20)/per_sample));
    output.assign(rows.size(),std::vector<float>(pages));
    for(size_t first=0;first<rows.size();first+=batch) {
        const auto count=std::min(batch,rows.size()-first);
        impl->indices.reserve(count*sizeof(uint32_t));
        impl->lse.reserve(count*heads*pages*sizeof(float));
        impl->norms.reserve(count*heads*2*sizeof(float));impl->output.reserve(count*pages*sizeof(float));
        checked(cudaMemcpyAsync(impl->indices.data,rows.data()+first,count*sizeof(uint32_t),cudaMemcpyHostToDevice,impl->stream));
        const uint8_t* mask_data=static_cast<const uint8_t*>(mask->data);size_t mask_stride=mask->nb[1];
        std::vector<uint8_t> mask_host;
        if(staged) {
            impl->mask.reserve(count*mask_row);mask_host.resize(count*mask_row);
            for(size_t i=0;i<count;++i) ggml_backend_tensor_get(mask,mask_host.data()+i*mask_row,
                size_t(rows[first+i])*mask->nb[1],mask_row);
            checked(cudaMemcpyAsync(impl->mask.data,mask_host.data(),mask_host.size(),cudaMemcpyHostToDevice,impl->stream));
            mask_data=static_cast<const uint8_t*>(impl->mask.data);mask_stride=mask_row;
        }
        page_lse<<<dim3(unsigned(pages),unsigned(count*heads)),128,0,impl->stream>>>(
            static_cast<const uint8_t*>(q->data),q->type,q->nb[1],q->nb[2],static_cast<const uint8_t*>(k->data),
            k->type,k->nb[1],k->nb[2],mask_data,mask->type,mask_stride,static_cast<const uint32_t*>(impl->indices.data),
            staged,int(q->ne[0]),heads,int(k->ne[2]),int(k->ne[1]),pages,int(block),scale,softcap,static_cast<float*>(impl->lse.data));
        normalize<<<unsigned(count*heads),128,0,impl->stream>>>(static_cast<const float*>(impl->lse.data),pages,static_cast<float*>(impl->norms.data));
        average<<<dim3(unsigned((pages+127)/128),unsigned(count)),128,0,impl->stream>>>(
            static_cast<const float*>(impl->lse.data),static_cast<const float*>(impl->norms.data),pages,heads,static_cast<float*>(impl->output.data));
        checked(cudaGetLastError());
        std::vector<float> host(count*pages);
        checked(cudaMemcpyAsync(host.data(),impl->output.data,host.size()*sizeof(float),cudaMemcpyDeviceToHost,impl->stream));
        checked(cudaStreamSynchronize(impl->stream));
        for(size_t i=0;i<count;++i) std::copy_n(host.data()+i*pages,pages,output[first+i].data());
    }
    return true;
}
