#include <algorithm>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "utils.h"

#define MAX_THREADS_PER_BLOCK 512

// NOTE "scan" and "prefix-sum" (psum) are used interchangably in this context

__global__
void block_psum(const unsigned int * const g_in,
                      unsigned int * const g_out,
                      unsigned int * const g_sums,
                const size_t n)
{
  extern __shared__ unsigned int smem[];
  const size_t bx = blockIdx.x * blockDim.x;
  const size_t tx = threadIdx.x;
  const size_t px = bx + tx;
  int offset = 1;

  // init
  smem[2*tx]   = g_in[2*px];
  smem[2*tx+1] = g_in[2*px+1];

  ////
  // up sweep
  ////
  for (int d = n >> 1; d > 0; d >>= 1)
  {
    __syncthreads();

    if (tx < d)
    {
      int ai = offset * (2*tx+1) - 1;
      int bi = offset * (2*tx+2) - 1;

      smem[bi] += smem[ai];
    }
    offset <<= 1;
  }

  // save block sum and clear last element
  if (tx == 0) {
    if (g_sums != NULL)
      g_sums[blockIdx.x] = smem[n-1];
    smem[n-1] = 0;
  }

  ////
  // down sweep
  ////
  for (int d = 1; d < n; d <<= 1)
  {
    offset >>= 1;
    __syncthreads();

    if (tx < d)
    {
      int ai = offset * (2*tx+1) - 1;
      int bi = offset * (2*tx+2) - 1;

      // swap
      unsigned int t = smem[ai];
      smem[ai]  = smem[bi];
      smem[bi] += t;
    }
  }
  __syncthreads();

  // save scan result
  g_out[2*px]   = smem[2*tx];
  g_out[2*px+1] = smem[2*tx+1];
}

__global__
void scatter_incr(      unsigned int * const d_array,
                  const unsigned int * const d_incr)
{
  const size_t bx = 2 * blockDim.x * blockIdx.x;
  const size_t tx = threadIdx.x;
  const unsigned int u = d_incr[blockIdx.x];
  d_array[bx + 2*tx]   += u;
  d_array[bx + 2*tx+1] += u;
}

// TODO 1) current version only works for len <= MAX_THREADS_PER_BLOCK * MAX_THREADS_PER_BLOCK
// TODO 2) current version doesnt handle bank conflicts
void psum(const unsigned int * const d_in,
                unsigned int * const d_out,
          const size_t len)
{
  const unsigned int nthreads = MAX_THREADS_PER_BLOCK;
  const unsigned int block_size = 2 * nthreads;
  const unsigned int smem = block_size * sizeof(unsigned int);
  // n = smallest multiple of block_size such that larger than or equal to len
  const size_t n = len % block_size == 0 ? len : (1+len/block_size)*block_size;
  // number of blocks
  int nblocks = n/block_size;

  // allocate memories on gpu
  unsigned int *d_scan, *d_sums, *d_incr;
  checkCudaErrors(cudaMalloc(&d_scan, sizeof(unsigned int)*n));
  checkCudaErrors(cudaMalloc(&d_sums, sizeof(unsigned int)*nblocks));
  checkCudaErrors(cudaMalloc(&d_incr, sizeof(unsigned int)*nblocks));

  // scan array by blocks (block_size = 2 * num threads)
  block_psum<<<nblocks, nthreads, smem>>>(d_in, d_scan, d_sums, block_size);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // scan block sums
  // TODO case when nblocks is bigger than block_size (see TODO 1)
  block_psum<<<1, nthreads, smem>>>(d_sums, d_incr, NULL, block_size);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // scatter block sums back to scanned blocks
  scatter_incr<<<nblocks, nthreads>>>(d_scan, d_incr);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // copy scan result back to d_out (cutoff at length len)
  checkCudaErrors(cudaMemcpy(d_out, d_scan, sizeof(unsigned int)*len, cudaMemcpyDeviceToDevice));

  // free allocated memories
  checkCudaErrors(cudaFree(d_incr));
  checkCudaErrors(cudaFree(d_sums));
  checkCudaErrors(cudaFree(d_scan));
}

__global__
void histo_01(const unsigned int * const d_data,
                    unsigned int * const d_histo,
              const unsigned int bit,
              const size_t len)
{
  int px = blockDim.x * blockIdx.x + threadIdx.x;
  if (px >= len)
    return;
  const unsigned int bin = (d_data[px] >> bit) & 1U;
  atomicAdd(&(d_histo[bin]), 1);
}

__global__
void map_ones(const unsigned int * const d_data,
                    unsigned int * const d_ones,
              const unsigned int bit,
              const size_t len)
{
  const unsigned int px = blockDim.x * blockIdx.x + threadIdx.x;
  if (px >= len)
    return;
  d_ones[px] = (d_data[px] >> bit) & 1U;    // =1 if d_data[px] at bit position is 1, =0 otherwise
}

__global__
void flip_01(      unsigned int * const d_bits,
             const size_t len)
{
  const unsigned int px = blockDim.x * blockIdx.x + threadIdx.x;
  if (px >= len)
    return;
  d_bits[px] ^= 1U;     // toggle 0(1) to 1(0)
}

__global__
void permute(const unsigned int * const d_in,
                   unsigned int * const d_out,
             const unsigned int * const d_zeros,
             const unsigned int * const d_scan0,
             const unsigned int * const d_scan1,
             const unsigned int * const d_h01,
             const size_t len)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= len)
    return;
  unsigned int pos = (d_zeros[idx]) ? d_scan0[idx] : d_scan1[idx] + d_h01[0];
  d_out[pos] = d_in[idx];
}

void bitonic_sort(      unsigned int * const d_inputVals,
                        unsigned int * const d_inputPos,
                        unsigned int * const d_outputVals,
                        unsigned int * const d_outputPos,
                  const size_t numElems)
{
  unsigned int nthreads = MAX_THREADS_PER_BLOCK;
  unsigned int nblocks = 1 + numElems/MAX_THREADS_PER_BLOCK;

  // allocate memories on gpu
  unsigned int * d_h01;         // histo of 0's and 1's
  checkCudaErrors(cudaMalloc(&d_h01, sizeof(unsigned int)*2));
  unsigned int * d_p01;         // predicate of 0's or 1's
  checkCudaErrors(cudaMalloc(&d_p01, sizeof(unsigned int)*numElems));
  unsigned int * d_scan0;       // scan of d_p01 when d_p01 is flipped to represent 0's
  checkCudaErrors(cudaMalloc(&d_scan0, sizeof(unsigned int)*numElems));
  unsigned int * d_scan1;       // scan of d_p01 when d_p01 is flipeed to represent 1's
  checkCudaErrors(cudaMalloc(&d_scan1, sizeof(unsigned int)*numElems));

  // ping pong (dummy) pointers
  unsigned int *d_ping1, *d_pong1, *d_ping2, *d_pong2;

  // loop from lowest bit to highest bit
  for (unsigned int bit = 0U; bit < sizeof(unsigned int) * CHAR_BIT ; bit++)
  {
    // ping pong input/output pointers (depending on bit is odd/even)
    d_ping1 = (bit & 1) ? d_outputVals : d_inputVals;
    d_pong1 = (bit & 1) ? d_inputVals  : d_outputVals;
    d_ping2 = (bit & 1) ? d_outputPos  : d_inputPos;
    d_pong2 = (bit & 1) ? d_inputPos   : d_outputPos;

    // reset histo to zeros at each bin
    checkCudaErrors(cudaMemset(d_h01, 0U, sizeof(unsigned int)*2));
    // perform histo at bit position
    histo_01<<<nblocks, nthreads>>>(d_ping1, d_h01, bit, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    // map position of ones (at current bit position) to d_p01
    map_ones<<<nblocks, nthreads>>>(d_ping1, d_p01, bit, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    // scan predicate of ones
    psum(d_p01, d_scan1, numElems);
    // flip d_p01 to represent predicate of zeros
    flip_01<<<nblocks, nthreads>>>(d_p01, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    // scan predicate of zeros
    psum(d_p01, d_scan0, numElems);

    // combine above results to get sorted position (wrt current bit position)
    permute<<<nblocks, nthreads>>>(d_ping1, d_pong1, d_p01, d_scan0, d_scan1, d_h01, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    permute<<<nblocks, nthreads>>>(d_ping2, d_pong2, d_p01, d_scan0, d_scan1, d_h01, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  }

  // copy results to out{vals,pos} if numElems is even
  if (numElems & 1 == 0)
  {
    checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice));
  }

  // gpu clean up
  checkCudaErrors(cudaFree(d_scan1));
  checkCudaErrors(cudaFree(d_scan0));
  checkCudaErrors(cudaFree(d_p01));
  checkCudaErrors(cudaFree(d_h01));
}


int main(int argc, char* argv[])
{

  size_t len = 10;    // default len 10

  if (argc >= 2)
  {
    std::istringstream ss(argv[1]);
    int tmp;
    if (ss >> tmp)
      len = tmp;
  }

  thrust::host_vector<unsigned int> h_val(len);
  thrust::host_vector<unsigned int> h_pos(len);

  // generate random uints to fill h_val
  thrust::generate(h_val.begin(), h_val.end(), rand);
  // set elements of h_pos to 0, 1, 2, ...
  thrust::sequence(h_pos.begin(), h_pos.end());

  // make device vectors
  thrust::device_vector<unsigned int> d_inval(h_val);
  thrust::device_vector<unsigned int> d_outval(h_val);
  thrust::device_vector<unsigned int> d_inpos(h_pos);
  thrust::device_vector<unsigned int> d_outpos(h_pos);

  // corresponding device pointers
  thrust::device_ptr<unsigned int> dp_inval = d_inval.data();
  thrust::device_ptr<unsigned int> dp_outval = d_outval.data();
  thrust::device_ptr<unsigned int> dp_inpos = d_inpos.data();
  thrust::device_ptr<unsigned int> dp_outpos = d_outpos.data();

  bitonic_sort(thrust::raw_pointer_cast(dp_inval),
               thrust::raw_pointer_cast(dp_inpos),
               thrust::raw_pointer_cast(dp_outval),
               thrust::raw_pointer_cast(dp_outpos),
               len);

  // simple output
  int width = (int)log10(len) + 1;
  for (size_t i = 0; i < len; i++)
    std::cout << std::setw(width) << d_outpos[i] << ": " << d_outval[i] << "\n";

  return 0;
}
