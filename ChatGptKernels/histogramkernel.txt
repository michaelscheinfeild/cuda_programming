/*
This kernel assumes that you have an 8-bit grayscale image stored as a linear array of uchar values in global memory.
 The num_pixels parameter tells the kernel how many pixels are in the image. The kernel computes a histogram of the pixel intensities and stores it in an array of
 int values in global memory.

The kernel works by dividing the image into multiple thread blocks, with each thread block computing a portion of the histogram. 
Each thread block uses shared memory to accumulate a partial histogram, and then updates the global histogram using atomic operations to avoid race conditions. 
The kernel assumes that the image size is large enough to fully utilize the available thread blocks.

Note that this kernel assumes that the image size is a multiple of the block size, and that the block size is less than or equal to 1024. 
If your image size is not a multiple of the block size, you will need to handle the leftover pixels separately. Additionally, 
if your image is in color or has a different bit depth, you will need to modify the kernel accordingly.



The line index += gridDim.x * block_size in the while loop of the histogram_kernel is used to ensure that each thread processes a different pixel in the image.

In CUDA, a kernel is executed by a grid of thread blocks, where each block contains a set of threads that execute the same code in parallel.
 The gridDim.x variable gives the number of thread blocks in the x-direction of the grid. Each thread block can process a different portion of the image.

The index variable in the kernel is used to keep track of the index of the pixel being processed. Each thread block computes a partial histogram for a subset of the image,
 and each thread in the block processes a different pixel. The while loop processes pixels in the image until all pixels have been processed by the kernel. 
The index variable is incremented by gridDim.x * block_size at the end of each iteration of the while loop to ensure that each thread processes a different pixel.

The expression gridDim.x * block_size calculates the total number of threads in the grid, which is equal to the total number of pixels being processed by the kernel. 
By incrementing index by this value at each iteration of the while loop, we ensure that each thread processes a different pixel until all pixels have been processed.

*/
__global__ void histogram_kernel(const uchar* image, int* histogram, int num_pixels)
{
    // determine thread and block index
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int index = blockIdx.x * block_size + tid;

    // initialize shared memory histogram
    __shared__ int shared_histogram[256];
    for (int i = tid; i < 256; i += block_size)
    {
        shared_histogram[i] = 0;
    }

    // update shared memory histogram
    while (index < num_pixels)
    {
        uchar pixel_value = image[index];
        atomicAdd(&(shared_histogram[pixel_value]), 1);
        index += gridDim.x * block_size;
    }

    // synchronize threads to ensure all updates are complete
    __syncthreads();

    // update global memory histogram
    for (int i = tid; i < 256; i += block_size)
    {
        atomicAdd(&(histogram[i]), shared_histogram[i]);
    }
}
