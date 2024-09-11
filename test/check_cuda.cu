// filename: check_cuda.cu
#include <cuda_runtime.h>
#include <stdio.h>
 
__global__ void hello_from_gpu(void) {
    printf("Hello, World from GPU!\n");
}
 
int main() {
    int device_count;
    cudaGetDeviceCount(&device_count);
 
    if (device_count == 0) {
        fprintf(stderr, "Did not detect any CUDA-capable devices.\n");
        return 1;
    }
 
    int device;
    for (device = 0; device < device_count; ++device) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            printf("Device %d: %s\n", device, prop.name);
            hello_from_gpu<<<1, 1>>>();
            cudaDeviceSynchronize();
        }
    }
 
    return 0;
}