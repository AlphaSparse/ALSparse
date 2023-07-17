#include <cassert>
#include <iostream>
#include <vector>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, char const* const func, char const* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const* const file, int const line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <int BLOCK_SIZE = 1024, int RADIUS = 5>
__global__ void stencil_1d_kernel(int const* d_in, int* d_out,
                                  int valid_array_size)
{
    extern __shared__ int temp[];

    // This has to be int because we will use negative indices.
    int const gindex{static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x)};
    int const lindex{static_cast<int>(threadIdx.x) + RADIUS};

    int const valid_block_size{
        min(BLOCK_SIZE,
            valid_array_size - static_cast<int>(blockIdx.x * blockDim.x))};

    // Read input elements into shared memory
    if (gindex < valid_array_size)
    {
        temp[lindex] = d_in[gindex];
        if (RADIUS <= valid_block_size)
        {
            if (threadIdx.x < RADIUS)
            {
                temp[lindex - RADIUS] = d_in[gindex - RADIUS];
                temp[lindex + valid_block_size] =
                    d_in[gindex + valid_block_size];
            }
        }
        else
        {
            for (int i{0}; i < RADIUS; i += valid_block_size)
            {
                // Some threads might have to do one more job than other
                // threads.
                if (lindex - RADIUS + i < RADIUS)
                {
                    temp[lindex - RADIUS + i] = d_in[gindex - RADIUS + i];
                    temp[lindex + valid_block_size + i] =
                        d_in[gindex + valid_block_size + i];
                }
            }
        }
    }
    // Synchronize (ensure all the data is available)
    __syncthreads();

    if (gindex >= valid_array_size)
    {
        return;
    }

    // Apply the stencil
    int result{0};
    for (int offset{-RADIUS}; offset <= RADIUS; offset++)
    {
        result += temp[lindex + offset];
    }

    // Store the result
    d_out[gindex] = result;
}

void stencil_1d_cpu(int const* h_in, int* h_out, int radius,
                    int valid_array_size)
{
    for (int i{0}; i < valid_array_size; ++i)
    {
        int result{0};
        for (int offset{-radius}; offset <= radius; offset++)
        {
            result += h_in[i + offset];
        }
        h_out[i] = result;
    }
}

int main(int argc, char** argv)
{
    constexpr int const valid_array_size{1024 * 100 + 1};
    constexpr int const block_size{1024};
    constexpr int const grid_size{(valid_array_size + block_size - 1) /
                                  block_size};
    constexpr int const radius{6000};

    int const array_size{valid_array_size + 2 * radius};
    std::vector<int> const h_in(array_size, 1);
    std::vector<int> h_out{h_in};
    std::vector<int> h_out_reference{h_in};

    stencil_1d_cpu(h_in.data() + radius, h_out_reference.data() + radius,
                   radius, valid_array_size);

    int* d_in;
    int* d_out;

    CHECK_CUDA_ERROR(cudaMalloc(&d_in, array_size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, array_size * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in.data(), array_size * sizeof(int),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_out, h_out.data(), array_size * sizeof(int),
                                cudaMemcpyHostToDevice));

    int const sharedMemoryBytes{(block_size + radius * 2) * sizeof(int)};
    CHECK_CUDA_ERROR(cudaFuncSetAttribute(
        stencil_1d_kernel<block_size, radius>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemoryBytes));
    stencil_1d_kernel<block_size, radius>
        <<<grid_size, block_size, sharedMemoryBytes>>>(
            d_in + radius, d_out + radius, valid_array_size);
    CHECK_LAST_CUDA_ERROR();

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_out.data(), d_out, array_size * sizeof(int),
                                cudaMemcpyDeviceToHost));

    for (int i{0}; i < h_out_reference.size(); ++i)
    {
        assert(h_out[i] == h_out_reference[i]);
    }

    CHECK_CUDA_ERROR(cudaFree(d_in));
    CHECK_CUDA_ERROR(cudaFree(d_out));
}