#include "ex2.h"
#include <cuda/atomic>

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

__device__ void prefix_sum(int arr[], int arr_size) 
{
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
__device__ void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

__device__ void process_image(uchar *in, uchar *out, uchar* maps) {
    
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

__global__ void process_image_kernel(uchar *in, uchar *out, uchar* maps) {
    process_image(in, out, maps);
}

struct Active_Stream
{
    cudaStream_t* stream;
    int* img_id;
    Active_Stream(cudaStream_t* steam, int img_id_val) : stream(stream)
    {
        img_id = new int(img_id_val);
    }
    void activate()
    { 
        process_image_kernel<<<1, STREAM_THREAD_NUMTHREAD_NUM>>>(); //!FIXME how to call this?
    }
    ~Active_Stream() { delete img_id; }
};

class streams_server : public image_processing_server
{
private:
    // TODO define stream server context (memory buffers, streams, etc...)
    cudaStream_t streams[STREAMS_NUM];
    std::unordered_set<Active_Stream> active_streams;

public:
    streams_server()
    {
        // TODO initialize context (memory buffers, streams, etc...)
        for(cudaStream_t& stream : streams) {
            CUDA_CHECK(cudaStreamCreate(&stream));
        }
    }

    ~streams_server() override
    {
        for(cudaStream_t& stream : streams) {
            CUDA_CHECK(cudaStreamDestroy(&stream));
        }
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO place memory transfers and kernel invocation in streams if possible.
        for(cudaStream_t& stream : streams) {
            cudaError_t query = cudaStreamQuery(stream);
            if(query == cudaErrorNotReady) {
                continue;
            }
            else if(query == cudaSuccess) {
                Active_Stream active_stream(&stream, img_id);
                active_stream.insert(active_stream);
                active_stream.activate();
                return true;
            }
            else {
                CUDA_CHECK(query);
            }
        }
        return false;
    }

    bool dequeue(int *img_id) override
    {
        if(active_streams.empty()) {
            return false;
        }
        //!FIXME maybe they meant querying a single stream here?
        for(cudaStream_t& active_stream : active_streams) {
            cudaError_t query = cudaStreamQuery(*(active_stream.stream));
            if(cudaErrorNotReady == query) {
                continue;
            }
            else if(query == cudaSuccess) {
                *img_id = *(active_stream.img_id);
                active_streams.erase(active_stream);
                return true;
            }
            else {
                CUDA_CHECK(query);
            }
        }
        return false;
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

// TODO implement a lock
// TODO implement a MPMC queue
// TODO implement the persistent kernel
// TODO implement a function for calculating the threadblocks count

class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)
public:
    queue_server(int threads)
    {
        // TODO initialize host state
        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
    }

    ~queue_server() override
    {
        // TODO free resources allocated in constructor
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO push new task into queue if possible
        return false;
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        return false;

        // TODO return the img_id of the request that was completed.
        //*img_id = ... 
        return true;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
