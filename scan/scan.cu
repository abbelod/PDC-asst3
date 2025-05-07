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
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
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

__global__ void exclusive_scan_kernel_upsweep(int* array, int N, int two_d){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = idx * (2 * two_d);
    if(k+2*two_d -1 < N){
        array[k + 2 * two_d - 1] += array[k+two_d-1];
    }
}

__global__ void exclusive_scan_kernel_downsweep(int* array, int N, int two_d){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = idx * (2 * two_d);
    if(k+2*two_d -1 < N){
        int temp = array[k + two_d -1];
        array[k + two_d - 1] = array[k + 2 * two_d - 1];
        array[k + 2 * two_d -1] += temp;
    }

}

void exclusive_scan(int *input, int rounded_length, int *result) {

    int* device_input = nullptr;
    cudaMalloc(&device_input, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, rounded_length * sizeof(int), cudaMemcpyHostToDevice);

    // Upsweep phase
    for (int two_d = 1; two_d < rounded_length; two_d *= 2) {
        int threads = (rounded_length + 2 * two_d - 1) / (2 * two_d);
        int blocks = (threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        exclusive_scan_kernel_upsweep<<<blocks, THREADS_PER_BLOCK>>>(device_input, rounded_length, two_d);
        cudaDeviceSynchronize();
    }

    // Set root to 0
    int last_element = 0;
    cudaMemcpy(&device_input[rounded_length - 1], &last_element, sizeof(int), cudaMemcpyHostToDevice);

    // Downsweep phase
    for (int two_d = rounded_length / 2; two_d >= 1; two_d /= 2) {
        int threads = (rounded_length + 2 * two_d - 1) / (2 * two_d);
        int blocks = (threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        exclusive_scan_kernel_downsweep<<<blocks, THREADS_PER_BLOCK>>>(device_input, rounded_length, two_d);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(result, device_input, rounded_length * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_input);
}
//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int *inarray, int *end, int *resultarray) {
    int N = end - inarray;
    int rounded_length = nextPow2(N);
    int* device_input;
    int* device_result;

    // Allocate device_input and device_result with rounded_length
    cudaMalloc(&device_input, sizeof(int) * rounded_length);
    cudaMalloc(&device_result, sizeof(int) * rounded_length);

    // Initialize device_input and device_result (including padding)
    cudaMemcpy(device_input, inarray, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(device_input + N, 0, (rounded_length - N) * sizeof(int)); // Initialize padding to 0

    double startTime = CycleTimer::currentSeconds();
    // Call exclusive_scan with rounded_length
    exclusive_scan(device_input, rounded_length, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // Copy only the first N elements back to resultarray
    cudaMemcpy(resultarray, device_result, N * sizeof(int), cudaMemcpyDeviceToHost);

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
double cudaScanThrust(int *inarray, int *end, int *resultarray)
{

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
int find_repeats(int *device_input, int length, int *device_output)
{

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    return 0;
}

//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length)
{

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

    for (int i = 0; i < deviceCount; i++)
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
