/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The code that registers a PyTorch custom operation.
*/
#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
    // xzl: cast pointer only? 
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

void forward(
    const int B, 
    const int T, 
    const int C, 
    const int H, 
    torch::Tensor& r, 
    torch::Tensor& k, 
    torch::Tensor& v, 
    torch::Tensor& w, 
    torch::Tensor& u, 
    torch::Tensor& y) {
    @autoreleasepool {
        NSString * sourcePath = @"wkv5_metal.metal";

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        // xzl: below, 1st argument name omitted
        NSString * src = [NSString stringWithContentsOfFile:sourcePath encoding:NSUTF8StringEncoding error:&error]; 
        if (error) printf("%s: error: %s\n", __func__, [[error description] UTF8String]);

        // Load shader
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_1;  // needed for bfloat
        id<MTLLibrary> myKernLib = [device newLibraryWithSource:src options:options error:&error];
        TORCH_CHECK(myKernLib, "Failed to cr kernel library ", error.localizedDescription.UTF8String);

        id<MTLFunction> myMetalfunc = [myKernLib newFunctionWithName:[NSString stringWithUTF8String:"metal_forward"]];
        TORCH_CHECK(myMetalfunc, "Failed to cr func");

        // Create a compute pipeline state object for the soft shrink kernel.
        id<MTLComputePipelineState> myPipe = [device newComputePipelineStateWithFunction:myMetalfunc error:&error];
        TORCH_CHECK(myPipe, error.localizedDescription.UTF8String);

        // Get a reference to the command buffer for the MPS stream.
        id<MTLCommandBuffer> cmdbuf = torch::mps::get_command_buffer(); TORCH_CHECK(cmdbuf, "cmdbuf failed");

        // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
        dispatch_queue_t myQueue = torch::mps::get_dispatch_queue();

        // xzl 
        dispatch_sync(myQueue, ^(){
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            TORCH_CHECK(enc, "enc");

            [enc setComputePipelineState:myPipe];
            
            [enc setBytes:&B length:sizeof(B) atIndex:0];
            [enc setBytes:&T length:sizeof(T) atIndex:1];
            [enc setBytes:&C length:sizeof(C) atIndex:2];
            [enc setBytes:&H length:sizeof(H) atIndex:3];

            [enc setBuffer:getMTLBufferStorage(r) offset:r.storage_offset() * r.element_size() atIndex:4];
            [enc setBuffer:getMTLBufferStorage(k) offset:k.storage_offset() * k.element_size() atIndex:5];
            [enc setBuffer:getMTLBufferStorage(v) offset:v.storage_offset() * v.element_size() atIndex:6];
            [enc setBuffer:getMTLBufferStorage(w) offset:w.storage_offset() * w.element_size() atIndex:7];
            [enc setBuffer:getMTLBufferStorage(u) offset:u.storage_offset() * u.element_size() atIndex:8];
            [enc setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:9];

            [enc dispatchThreadgroups:MTLSizeMake(B*H, 1, 1) threadsPerThreadgroup:MTLSizeMake(_N_, 1, 1)];
            // [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [enc endEncoding];

            torch::mps::commit();
        });
    }
}

// C++ op 
void metal_forward(    
    const int B, 
    const int T, 
    const int C, 
    const int H, 
    torch::Tensor& r, 
    torch::Tensor& k, 
    torch::Tensor& v, 
    torch::Tensor& w, 
    torch::Tensor& u, 
    torch::Tensor& y) {
    return forward(B,T,C,H,r,k,v,w,u,y);
}

void metal_backward(
    const int B, 
    const int T, 
    const int C, 
    const int H, 
    torch::Tensor& r, 
    torch::Tensor& k, 
    torch::Tensor& v, 
    torch::Tensor& w /*float*/, 
    torch::Tensor& ww /*float*/, 
    torch::Tensor& u, 
    torch::Tensor& gy, /*from downstream*/
    torch::Tensor& gr,
    torch::Tensor& gk,
    torch::Tensor& gv,
    torch::Tensor& gw,
    torch::Tensor& gu
    ) {
    @autoreleasepool {
        NSString * sourcePath = @"wkv5_metal.metal";

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        NSString * src = [NSString stringWithContentsOfFile:sourcePath encoding:NSUTF8StringEncoding error:&error]; 
        if (error) printf("%s: error: %s\n", __func__, [[error description] UTF8String]);

        // shader
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_1;  // needed for bfloat
        id<MTLLibrary> myKernLib = [device newLibraryWithSource:src options:options error:&error];
        TORCH_CHECK(myKernLib, "Failed to cr kernel library ", error.localizedDescription.UTF8String);

        // kern func 
        id<MTLFunction> myMetalfunc = [myKernLib newFunctionWithName:[NSString stringWithUTF8String:"metal_backward"]];
        TORCH_CHECK(myMetalfunc, "Failed to cr func");

        // pipeline
        id<MTLComputePipelineState> myPipe = [device newComputePipelineStateWithFunction:myMetalfunc error:&error];
        TORCH_CHECK(myPipe, error.localizedDescription.UTF8String);
        
        // cmdbuf, queue....
        id<MTLCommandBuffer> cmdbuf = torch::mps::get_command_buffer(); TORCH_CHECK(cmdbuf, "cmdbuf failed");
        dispatch_queue_t myQueue = torch::mps::get_dispatch_queue();

        dispatch_sync(myQueue, ^(){
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            TORCH_CHECK(enc, "enc");

            [enc setComputePipelineState:myPipe];
            
            [enc setBytes:&B length:sizeof(B) atIndex:0];
            [enc setBytes:&T length:sizeof(T) atIndex:1];
            [enc setBytes:&C length:sizeof(C) atIndex:2];
            [enc setBytes:&H length:sizeof(H) atIndex:3];

            [enc setBuffer:getMTLBufferStorage(r) offset:r.storage_offset() * r.element_size() atIndex:4];
            [enc setBuffer:getMTLBufferStorage(k) offset:k.storage_offset() * k.element_size() atIndex:5];
            [enc setBuffer:getMTLBufferStorage(v) offset:v.storage_offset() * v.element_size() atIndex:6];

            [enc setBuffer:getMTLBufferStorage(w) offset:w.storage_offset() * w.element_size() atIndex:7];
            [enc setBuffer:getMTLBufferStorage(ww) offset:ww.storage_offset() * ww.element_size() atIndex:8];
            [enc setBuffer:getMTLBufferStorage(u) offset:u.storage_offset() * u.element_size() atIndex:9];

            [enc setBuffer:getMTLBufferStorage(gy) offset:gy.storage_offset() * gy.element_size() atIndex:10];
            [enc setBuffer:getMTLBufferStorage(gr) offset:gr.storage_offset() * gr.element_size() atIndex:11];
            [enc setBuffer:getMTLBufferStorage(gk) offset:gk.storage_offset() * gk.element_size() atIndex:12];
            [enc setBuffer:getMTLBufferStorage(gv) offset:gv.storage_offset() * gv.element_size() atIndex:13];
            [enc setBuffer:getMTLBufferStorage(gw) offset:gw.storage_offset() * gw.element_size() atIndex:14];
            [enc setBuffer:getMTLBufferStorage(gu) offset:gu.storage_offset() * gu.element_size() atIndex:15];

            [enc dispatchThreadgroups:MTLSizeMake(B*H, 1, 1) threadsPerThreadgroup:MTLSizeMake(_N_, 1, 1)];
            // [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [enc endEncoding];

            torch::mps::commit();
        });
    }
}

// Create Python bindings for the Objective-C++ code.
// xzl ... so that python can find these funcs in "compiled" metal
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &metal_forward);
    m.def("backward", &metal_backward);
}
