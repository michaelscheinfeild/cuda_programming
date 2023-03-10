/*
This kernel assumes that you have a grayscale image stored as a linear array of uchar values in global memory, and that you want to perform a
 separable convolution with a Gaussian kernel. The kernel parameter is an array of float values containing the Gaussian kernel coefficients, 
and the kernel_size parameter gives the size of the kernel. The kernel computes the convolved image and stores it in output_image.

The kernel works by dividing the image into multiple thread blocks, with each block computing a portion of the convolved image. 
The kernel first convolves the image with the Gaussian kernel in the x-direction, and then convolves the resulting image in the y-direction.
 The Gaussian kernel is assumed to be separable, so it can be convolved in the x- and y-directions separately. The kernel uses shared memory
 to store the kernel coefficients and avoid redundant memory accesses.

*/
__global__ void convolve_separable_kernel(const uchar* input_image, uchar* output_image,
    int image_width, int image_height, const float* kernel, int kernel_size)
{
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int x = blockIdx.x * block_size + tid;
    int y = blockIdx.y;

    __shared__ float shared_kernel[32];

    // copy kernel to shared memory
    if (tid < kernel_size)
    {
        shared_kernel[tid] = kernel[tid];
    }

    // synchronize threads to ensure all kernel data has been copied
    __syncthreads();

    // convolve kernel with input image in x-direction
    float sum_x = 0;
    for (int i = 0; i < kernel_size; i++)
    {
        int index = y * image_width + x - kernel_size / 2 + i;
        if (index >= 0 && index < image_width)
        {
            sum_x += input_image[index] * shared_kernel[i];
        }
    }

    // write result to output image
    if (x < image_width)
    {
        output_image[y * image_width + x] = (uchar)sum_x;
    }

    // synchronize threads to ensure all writes to output image are complete
    __syncthreads();

    // convolve kernel with output image in y-direction
    float sum_y = 0;
    for (int i = 0; i < kernel_size; i++)
    {
        int index = (y - kernel_size / 2 + i) * image_width + x;
        if (index >= 0 && index < image_width * image_height)
        {
            sum_y += output_image[index] * shared_kernel[i];
        }
    }

    // write result to output image
    if (x < image_width && y < image_height)
    {
        output_image[y * image_width + x] = (uchar)sum_y;
    }
}

int main()
{
    // define input and output images
    uchar* input_image, *output_image;
    int image_width = 512;
    int image_height = 512;
    int image_size = image_width * image_height * sizeof(uchar);
    input_image = new uchar[image_size];
    output_image = new uchar[image_size];

    // initialize input image
    for (int i = 0; i < image_size; i++)
    {
        input_image[i] = rand() % 256;
    }

    // define Gaussian kernel
    float sigma = 1.0f;
    int kernel_size = 2 * ceil(2 * sigma) + 1;
    float* kernel = new float[kernel_size];
    for (int i = 0; i < kernel_size; i++)
    {
        kernel[i] = exp(-(i - kernel_size / 2) * (i - kernel_size / 2) / (2 * sigma * sigma));
    }

    // allocate device memory
    uchar* d_input_image, *d_output_image;
    float* d_kernel;
    cudaMalloc(&d_input_image, image_size);
    cudaMalloc(&d_output_image, image_size);
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));

    // copy input data to device memory
    cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // set up kernel launch parameters
    dim3 threads_per_block(32, 1, 1);
    dim3 num_blocks(image_width / threads_per_block.x, image_height, 1);

    // launch kernel
    convolve_separable_kernel<<<num_blocks, threads_per_block>>>(d_input_image, d_output_image,
        image_width, image_height, d_kernel, kernel_size);

    // copy result from device memory
    cudaMemcpy(output_image, d_output_image, image_size, cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_input_image);
    cudaFree(d_output_image);
    cudaFree(d_kernel);

    // free host memory
    delete[] input_image;
    delete[] output_image;
    delete[] kernel;

    return 0;
}
