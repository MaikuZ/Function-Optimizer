#ifndef FO_RANDOM_GENERATOR_HEADER
#define FO_RANDOM_GENERATOR_HEADER
/*!
  Author: MaikuZ
*/
#include "utilities.h"


__global__ void setUp(curandState *state, int seed, int n) {
  int bid = blockIdx.x + blockIdx.y * blockDim.x + blockIdx.z * (blockDim.x * blockDim.y);
  int tid = threadIdx.x;
  int gid = tid + bid * kBlock;
  if (gid >= n)
    return;
  
  curand_init(seed, gid, clock64(), &state[gid]);
}

class RandomGenerator {
public:
  RandomGenerator(int n = kBlock) : n_(n) {
    srand(time(0));
    seed_ = rand();
    gpuErrchk(cudaMalloc(&d_state_, sizeof(curandState) * n_));
    int no_blocks = n_ / kBlock + (n_ % kBlock != 0);
    setUp<<<no_blocks, kBlock>>>(d_state_, seed_, n_);
  }

  curandState *getState() {
    return d_state_;
  }

  ~RandomGenerator() {
    gpuErrchk(cudaFree(d_state_));
  }
private:
  unsigned int seed_;
  int n_;
  curandState *d_state_;
};

#endif