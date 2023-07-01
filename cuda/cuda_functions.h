#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

void xor_gpu(unsigned char* c, unsigned char* a, unsigned char* b, size_t size);

__global__
void xor_kernel(unsigned char* c, unsigned char* a, unsigned char* b, size_t size);


#ifdef __cplusplus
}
#endif

