/*
This kernel assumes that you have an array of floats in global memory that you want to sum up, and that you want to store the result in an output array.
 The array_length parameter tells the kernel how many elements are in the input array.

The kernel works by copying data from global memory to shared memory, performing a parallel reduction in shared memory, 
and then writing the result to global memory.
 The blockDim.x parameter tells the kernel how many threads are in the thread block, 
and the blockIdx.x parameter tells the kernel which block it is in.

Note that this kernel assumes that the input array length is a multiple of the block size, and that the block size is less than or equal to 256. 
If your input array length is not a multiple of the block size, you will need to handle the leftover elements separately.


*/
__global__ void sum_reduction_kernel(float* input_array, float* output_sum, int array_length)
{
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int index = blockIdx.x * block_size + tid;

    __shared__ float shared_array[256];

    // copy data from global memory to shared memory
    if (index < array_length)
    {
        shared_array[tid] = input_array[index];
    }
    else
    {
        shared_array[tid] = 0;
    }

    // synchronize to ensure all threads have finished copying data
    __syncthreads();

    // perform reduction in shared memory
    for (int stride = block_size / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared_array[tid] += shared_array[tid + stride];
        }
        __syncthreads();
    }

    // write result to output array
    if (tid == 0)
    {
        output_sum[blockIdx.x] = shared_array[0];
    }
}
