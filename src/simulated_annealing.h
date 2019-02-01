#ifndef FO_SIMULATED_ANNEALING_HEADER
#define FO_SIMULATED_ANNEALING_HEADER
/*!
  Author: MaikuZ
*/
#include "pfunc.h"

/*! The function for the simulated annealing is being passed using 
templates. This makes it cumbersome to have the definitions
inside of a .cpp file. That's why the implementation of the 
class is in the header file only.
*/

/// The function prototype for the temperature function.
typedef float(*pointerFunctionTemperature_t)(int k, int k_max);

// The temperature function to be used.
__device__ float temperature(int k, int k_max) {
  return 50 * pow(0.8, k);
}

template<pointerFunction_t Function, pointerFunctionTemperature_t Temp, int Dim>
__global__ static void kernel(State state, float * devValues, float (* devArgs)[Dim], int n, int k_max) {
  int bid = blockIdx.x + blockIdx.y * blockDim.x + blockIdx.z * (blockDim.x * blockDim.y);
  int tid = threadIdx.x;
  int gid = tid + bid * kBlock;
  if (gid >= n)
    return;

  curandState *rand_state = state.rand_state + gid;

  // Initialise current_position
  float current_position[kMaximumDimension];
  float current_value;
  for (int i = 0;i < Dim;i++) {
    current_position[i] = (state.boundaries[i].max - state.boundaries[i].min) *
                       curand_uniform(rand_state) +
                       state.boundaries[i].min;
  }

  current_value = Function(current_position);
  for (int i = 0;i < k_max;i++) {
    float new_position[kMaximumDimension];
    for (int j = 0;j < Dim;j++) {
      // new random position - inside the boundaries.
      float l = (state.boundaries[j].max - state.boundaries[j].min) / sqrtf(n);
      if (current_position[j] - l < state.boundaries[j].min) {
        new_position[j] = state.boundaries[j].min  + l * curand_uniform(rand_state);
      } else if (current_position[j] + l > state.boundaries[j].max) {
        new_position[j] = state.boundaries[j].max - l * curand_uniform(rand_state);
      } else {
        new_position[j] = current_position[j] + (curand_uniform(rand_state) - 0.5f) * l;
      }
    }
    float new_value = Function(new_position);

    float temp = Temp(i, k_max);
    float rand_result = curand_uniform(rand_state);
    
    if (exp((current_value - new_value) / temp) > rand_result) {
      for (int j = 0;j < Dim;j++) {
        current_position[j] = new_position[j];
      }
      current_value = new_value;
    }
  }

  // Pmax the whole block.
  __shared__ float blockValues[kBlock];
  blockValues[tid] = current_value;
  pfunc1024<float, dev_min>(kBlock, blockValues);

  // Find the winner for this block. 
  // Realistically speaking, there should be no conflicts whatsoever.
  __shared__ int winner;
  winner = false;
  __syncthreads();
  devValues[bid] = blockValues[kBlock - 1];

  if (blockValues[kBlock - 1] == current_value) {
    if (atomicMax(&winner, true) == false) {
      for (int j = 0;j < Dim;j++) {
        devArgs[bid][j] = current_position[j];
      }
    }
  }
}


template<pointerFunction_t Function,  pointerFunctionTemperature_t Temp, int Dim> 
class SimulatedAnnealing : public GlobalOptimumFinder<Function, Dim> {
public:
  SimulatedAnnealing() : GlobalOptimumFinder<Function, Dim> () {
  }
  
  void findMinimum(std::vector <float> &out, int n) {
    RandomGenerator gen(n);
    int no_blocks = n / kBlock + (n % kBlock != 0);

    State state = {this->devDomainSpace_, gen.getState()};
    DevArray<float> devValues(no_blocks);
    DevArray<float[Dim]> devArgs(no_blocks);

    kernel<Function, Temp, Dim><<<no_blocks, kBlock>>>
    (state, devValues.data, devArgs.data, n, 10000);
    pfunc<float, dev_min>(no_blocks, devValues.data);

    float bestVal;
    gpuErrchk(cudaMemcpy(&bestVal, &devValues.data[no_blocks - 1], sizeof(float) * 1, cudaMemcpyDeviceToHost));
    
    DevArray<int> devWinner(no_blocks);
    int no_blocks_for_winners = no_blocks/kBlock + (no_blocks % kBlock != 0);
    calculateWinners<<<no_blocks_for_winners, kBlock>>>(devValues.data, devWinner.data, bestVal, no_blocks);
    pfunc<int, dev_min>(no_blocks, devWinner.data);

    int winner;
    gpuErrchk(cudaMemcpy(&winner, &devWinner.data[no_blocks - 1], sizeof(int) * 1, cudaMemcpyDeviceToHost));
    assert(winner < no_blocks);
    gpuErrchk(cudaMemcpy(out.data(), &devArgs.data[winner], sizeof(float) * Dim, cudaMemcpyDeviceToHost));
    return;
  }
private:
};

#endif