#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void upsweep_kernel(int* data, int twod, int twod1, int activeThreads) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= activeThreads) {
        return;
    }

    int i = index * twod1;
    data[i + twod1 - 1] += data[i + twod - 1];
}

__global__ void downsweep_kernel(int* data, int twod, int twod1, int activeThreads) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= activeThreads) {
        return;
    }

    int i = index * twod1;
    int t = data[i + twod - 1];
    data[i + twod - 1] = data[i + twod1 - 1];
    data[i + twod1 - 1] += t;
}

__global__ void clear_last_kernel(int* data, int index) {
    data[index] = 0;
}

__global__ void make_repeats_flags_kernel(int* input, int length, int roundedLength, int* flags) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= roundedLength) {
        return;
    }

    flags[index] = (index < length - 1 && input[index] == input[index + 1]) ? 1 : 0;
}

__global__ void scatter_repeats_kernel(int* flags, int* offsets, int length, int* output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) {
        return;
    }

    if (flags[index]) {
        output[offsets[index]] = index;
    }
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result)
{
    int roundedLength = nextPow2(N);
    size_t inputBytes = sizeof(int) * N;

    if (input != result) {
        cudaMemcpy(result, input, inputBytes, cudaMemcpyDeviceToDevice);
    }
    if (roundedLength > N) {
        cudaMemset(result + N, 0, sizeof(int) * (roundedLength - N));
    }

    for (int twod = 1; twod < roundedLength; twod *= 2) {
        int twod1 = 2 * twod;
        int activeThreads = roundedLength / twod1;
        int blocks = (activeThreads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        upsweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(result, twod, twod1, activeThreads);
    }

    clear_last_kernel<<<1, 1>>>(result, roundedLength - 1);

    for (int twod = roundedLength / 2; twod >= 1; twod /= 2) {
        int twod1 = 2 * twod;
        int activeThreads = roundedLength / twod1;
        int blocks = (activeThreads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        downsweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(result, twod, twod1, activeThreads);
    }
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    if (length <= 1) {
        return 0;
    }

    int roundedLength = nextPow2(length);
    int* flags;
    int* offsets;

    cudaMalloc((void **)&flags, roundedLength * sizeof(int));
    cudaMalloc((void **)&offsets, roundedLength * sizeof(int));

    int blocks = (roundedLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    make_repeats_flags_kernel<<<blocks, THREADS_PER_BLOCK>>>(device_input, length, roundedLength, flags);

    exclusive_scan(flags, length, offsets);

    int lastFlag = 0;
    int lastOffset = 0;
    cudaMemcpy(&lastFlag, flags + length - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lastOffset, offsets + length - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int count = lastOffset + lastFlag;

    blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    scatter_repeats_kernel<<<blocks, THREADS_PER_BLOCK>>>(flags, offsets, length, device_output);

    cudaFree(flags);
    cudaFree(offsets);

    return count; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
