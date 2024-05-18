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
    @autoreleasepool {      // xzl: ???

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        // Set the number of threads equal to the number of elements within the input tensor.
        int numThreads = input.numel();

        // xzl
        NSString * sourcePath = @"wkv5.metal";
        // xzl: below, 1st argument name omitted
        NSString * src = [NSString stringWithContentsOfFile:sourcePath encoding:NSUTF8StringEncoding error:&error]; 
        if (error)
            printf("%s: error: %s\n", __func__, [[error description] UTF8String]);

        // Load shader.
        id<MTLLibrary> myKernelLibrary = [device newLibraryWithSource:src
                                                                  options:nil
                                                                    error:&error];
        TORCH_CHECK(myKernelLibrary, 
            "Failed to to create my kernel library, error: ", error.localizedDescription.UTF8String);                                                                    

        id<MTLFunction> mySoftShrinkFunction = [myKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:"metal_forward"]];
        TORCH_CHECK(mySoftShrinkFunction, "Failed to create function state object forward");

        // Create a compute pipeline state object for the soft shrink kernel.
        id<MTLComputePipelineState> mysoftShrinkPSO = [device newComputePipelineStateWithFunction:mySoftShrinkFunction error:&error];
        TORCH_CHECK(mysoftShrinkPSO, error.localizedDescription.UTF8String);

        // Get a reference to the command buffer for the MPS stream.
        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

        // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        // xzl 
        dispatch_sync(serialQueue, ^(){
            // Start a compute pass.
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

            [computeEncoder setComputePipelineState:mysoftShrinkPSO];
            
            [encoder setBytes:&B length:sizeof(B) atIndex:0];
            [encoder setBytes:&T length:sizeof(T) atIndex:1];
            [encoder setBytes:&C length:sizeof(C) atIndex:2];
            [encoder setBytes:&H length:sizeof(H) atIndex:3];

            [computeEncoder setBuffer:getMTLBufferStorage(r) offset:r.storage_offset() * r.element_size() atIndex:4];
            [computeEncoder setBuffer:getMTLBufferStorage(k) offset:k.storage_offset() * k.element_size() atIndex:5];
            [computeEncoder setBuffer:getMTLBufferStorage(v) offset:v.storage_offset() * v.element_size() atIndex:6];
            [computeEncoder setBuffer:getMTLBufferStorage(w) offset:w.storage_offset() * w.element_size() atIndex:7];
            [computeEncoder setBuffer:getMTLBufferStorage(u) offset:u.storage_offset() * u.element_size() atIndex:8];
            [computeEncoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:9];

            MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

            [computeEncoder dispatchThreadgroups:MTLSizeMake(B*H, 1, 1) threadsPerThreadgroup:MTLSizeMake(_N_, 1, 1)];
            // [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [computeEncoder endEncoding];

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

// Create Python bindings for the Objective-C++ code.
// xzl ... so that python can find these funcs in "compiled" metal
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("metal_forward", &metal_forward);
}
