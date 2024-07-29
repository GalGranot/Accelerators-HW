#include "ex1.h"

/*=============================================================================
* constants
=============================================================================*/
#define COLOR_RANGE 256
#define TILES TILE_COUNT * TILE_COUNT
#define PIXELS_PER_TILE TILE_WIDTH * TILE_WIDTH
#define TILES_PER_LINE TILE_COUNT
#define THREAD_NUM 256

/*=============================================================================
* helper functions
=============================================================================*/

__device__ void compute_histograms(uchar *img, int histograms[COLOR_RANGE], int tile) {
    int tid = threadIdx.x;
    for(int i=0; i<COLOR_RANGE; i++){    
        if(tid < COLOR_RANGE) {
            histograms[tid] = 0; //#threads > COLOR_RANGE
        }
    }
    __syncthreads();
    int left = (tile%TILE_COUNT)*TILE_WIDTH;
    //int right = left + TILE_WIDTH - 1;
    int top = (tile/TILE_COUNT)*TILE_WIDTH;
    //int bottom = top + IMG_WIDTH * TILE_WIDTH -1;

    // Calculate the histogram for the tile
    int pixels_per_thread = PIXELS_PER_TILE / THREAD_NUM;
    for(int i = tid*pixels_per_thread; i< (tid+1)*pixels_per_thread; i++){
        int x = left + i%TILE_WIDTH;
        int y = top + i/TILE_WIDTH;
        int index = y*IMG_WIDTH + x;
        atomicAdd(&histograms[img[index]], 1);
        }
    
/*** 
    //iterate over the pixels in the tile
    //currenty, each thread work on 16 pixels
    int pixels_per_thread = PIXELS_PER_TILE / THREAD_NUM;

    for (int i=tid*pixels_per_thread ; i<((tid+1)*pixels_per_thread); i++){
        atomicAdd(&histograms[img[i]], 1);
    }
***/

    __syncthreads(); 
}


__device__ void compute_map(int cdf[COLOR_RANGE],uchar *maps){
    int tid = threadIdx.x;
    for(int i=0; i<COLOR_RANGE; i++){    
        if (tid<COLOR_RANGE)
        {
            double map_value = ((float(cdf[tid])) * (COLOR_RANGE-1) ) / (PIXELS_PER_TILE);
            maps[tid] = (uchar)map_value;
        }
    }
    __syncthreads();
}

/*=============================================================================
* required functions
=============================================================================*/

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increment;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid >= stride && tid < COLOR_RANGE) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < COLOR_RANGE) {
            arr[tid] += increment;
        }
        __syncthreads();
    }

    __syncthreads();

    return;
}

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__ 
void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) {

    /* steps:
        1. divide image into tiles
        2. compute histogram for each tile
        3. compute cdf for each histogram
        4. compute map for each cdf
        5. interpolate_device() and return
    */

    //save to register
    int bx = blockIdx.x;

    //index 
    int index = bx * IMG_WIDTH * IMG_HEIGHT;
    
    for(int tile=0; tile<TILES; tile++){
        //shared memory 
        __shared__ int histogram [COLOR_RANGE];
        //compute the histogram for each tile
        compute_histograms(&all_in[index], histogram, tile);
        __syncthreads();

        //compute CDF for each histogram
        prefix_sum(histogram, COLOR_RANGE);
        __syncthreads();

        //compute map from cdf
        compute_map(histogram, &maps[(bx*TILES + tile)* COLOR_RANGE]);
        __syncthreads();

   }
   __syncthreads();

   //interpolate device
    interpolate_device(&maps[bx*TILES*COLOR_RANGE], &all_in[index], &all_out[index]);
    __syncthreads();
    return; 
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    uchar * d_all_in;
    uchar * d_all_out;
    uchar * d_maps;
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;

    //TODO: allocate GPU memory for a single input image, a single output image, and maps
    int size = IMG_HEIGHT*IMG_WIDTH * sizeof(uchar);
    
    CUDA_CHECK(cudaMalloc((void**) &context->d_all_in,size));
    CUDA_CHECK(cudaMalloc((void**) &context->d_all_out,size));
    CUDA_CHECK(cudaMalloc((void**) &context->d_maps,TILES*COLOR_RANGE*sizeof(uchar)));
    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: in a for loop:
    //   1. copy the relevant image from images_in to the GPU memory you allocated
    //   2. invoke GPU kernel on this image
    //   3. copy output from GPU memory to relevant location in images_out_gpu_serial

    int size = IMG_HEIGHT*IMG_WIDTH;
    for(int i=0; i<N_IMAGES; i++){
        //1
        CUDA_CHECK(cudaMemcpy(context->d_all_in, &images_in[i*size], size*sizeof(uchar),cudaMemcpyHostToDevice));
        //2 
        process_image_kernel<<<1, THREAD_NUM>>>(context->d_all_in, context->d_all_out, context->d_maps) ;
        CUDA_CHECK(cudaDeviceSynchronize());
        //3
        CUDA_CHECK(cudaMemcpy(&images_out[i*size], context->d_all_out, size*sizeof(uchar),cudaMemcpyDeviceToHost));
    }
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //TODO: free resources allocated in task_serial_init
    CUDA_CHECK(cudaFree(context->d_all_in));
    CUDA_CHECK(cudaFree(context->d_all_out));
    CUDA_CHECK(cudaFree(context->d_maps));
    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // TODO define bulk-GPU memory buffers
    uchar * d_all_in;
    uchar * d_all_out;
    uchar * d_maps;
};

/* Allocate GPU memory for all the input images, output images, and maps.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    //TODO: allocate GPU memory for all the input images, output images, and maps
    int size = N_IMAGES * IMG_HEIGHT * IMG_WIDTH * sizeof(uchar);
    
    CUDA_CHECK(cudaMalloc((void**) &context->d_all_in,size));
    CUDA_CHECK(cudaMalloc((void**) &context->d_all_out,size));
    CUDA_CHECK(cudaMalloc((void**) &context->d_maps,N_IMAGES*TILES*COLOR_RANGE*sizeof(uchar)));
    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    int size = N_IMAGES * IMG_HEIGHT *IMG_WIDTH * sizeof(uchar);
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out
    CUDA_CHECK(cudaMemcpy(context->d_all_in, images_in, size,cudaMemcpyHostToDevice));
    process_image_kernel<<<N_IMAGES, THREAD_NUM>>>(context->d_all_in, context->d_all_out, context->d_maps) ;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(images_out, context->d_all_out, size,cudaMemcpyDeviceToHost));
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //TODO: free resources allocated in gpu_bulk_init
    CUDA_CHECK(cudaFree(context->d_all_in));
    CUDA_CHECK(cudaFree(context->d_all_out));
    CUDA_CHECK(cudaFree(context->d_maps));
    
    free(context);
}
