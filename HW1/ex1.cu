//ex1.cu

/*=============================================================================
* includes
=============================================================================*/
#include "ex1.h"

/*=============================================================================
* constants
=============================================================================*/
#define COLOR_RANGE 256
#define TILES TILE_COUNT * TILE_COUNT
#define PIXELS_PER_TILE TILE_WIDTH * TILE_WIDTH
#define TILES_PER_LINE TILE_COUNT

/*=============================================================================
* helper functions
=============================================================================*/
__device__ inline int calc_offset(int i) { return }
__device__
void compute_histograms(uchar *img, int histograms[TILES][COLOR_RANGE]) {
    int tid = threadIdx.x;
    for (int i = 0; i < TILES; i++) {
        if(tid < COLOR_RANGE) {
            histograms[i][tid] = 0; //FIXME: assuming #threads > COLOR_RANGE
        }
    }
    __syncthreads();
    /*
    each thread needs to: 
        1. read a pixel value
        2. learn its block index
        3. atomic write to appropriate histogram
    */
   
   //FIXME: isn't correct right now
    for (int i = 0; i < TILES; i++) { 
        int line_offset = ((int)std::round(i / TILES_PER_LINE)) * IMG_WIDTH;
        int tile_offset = (i % TILES_PER_LINE) * TILE_WIDTH;
        //FIXME: need to add thread offset inside tile and disperse threads
        // throughout tile while dropping lines when appropriate
        int offset = line_offset + tile_offset /*+thread offset*/; 
        int value = img[offset];

        int tile; //FIXME: calc correct tile
        atomicAdd_block(&histograms[tile][value], 1);
    }
    __syncthreads();
}

/*=============================================================================
* required functions
=============================================================================*/

__global__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increment;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid >= stride) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
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
void interpolate_device(uchar *maps ,uchar *in_img, uchar *out_img);

__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) {
    /* steps:
        1. divide image into tiles
        2. compute histogram for each tile
        3. compute cdf for each histogram
        4. compute map from each pixel value to another
        5. interpolate_device() and return
    */
    __shared__ int histograms[TILE_COUNT][COLOR_RANGE];
    interpolate_device(maps, all_in, all_out);
    return;
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    // TODO define task serial memory buffers
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;

    //TODO: allocate GPU memory for a single input image, a single output image, and maps

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
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //TODO: free resources allocated in task_serial_init

    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // TODO define bulk-GPU memory buffers
};

/* Allocate GPU memory for all the input images, output images, and maps.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    //TODO: allocate GPU memory for all the input images, output images, and maps

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //TODO: free resources allocated in gpu_bulk_init

    free(context);
}
