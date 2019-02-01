#ifndef FO_PARTICLE_SWARM_HEADER
#define FO_PARTICLE_SWARM_HEADER
/*!
  Author: MaikuZ
*/
#include "global_optimum_finder.h" 
#include "random_gen.h"
#include "pfunc.h"

/*! The function for the particle swarm is being passed using 
templates. This makes it cumbersome to have the definitions
inside of a .cpp file. That's why the implementation of the 
class is in the header file only.
*/

template<pointerFunction_t Function, int Dim>
__global__ static void kernel(State state, int n, float * devValues, float (* devArgs)[Dim], int k_max, 
float omega, float phi_p, float phi_g) {
  int bid = blockIdx.x + blockIdx.y * blockDim.x + blockIdx.z * (blockDim.x * blockDim.y);
  int tid = threadIdx.x;
  int gid = tid + bid * kBlock;
  if (gid >= n)
    return;
  
  curandState *rand_state = state.rand_state + gid;

  __shared__ float swarm_best_position[Dim];
  __shared__ float swarm_best_value;
  swarm_best_value = INF;
  __syncthreads();

  float current_position[Dim];
  float current_value;
  float current_best_position[Dim];
  float current_best_value;
  float current_velocity[Dim];

  for (int i = 0;i < Dim;i++) {
    current_position[i] = (state.boundaries[i].max - state.boundaries[i].min) *
                       curand_uniform(rand_state) / powf(n/1024, 1.0f/Dim) +
                       state.boundaries[i].min + 
                       (state.boundaries[i].max - state.boundaries[i].min) / powf(n/1024, 1.0f/Dim) * bid;
    current_best_position[i] = current_position[i];
    current_velocity[i] = (curand_uniform(rand_state) - 0.5) / powf(n, 1.0f/Dim) * 2.0f *
                        (state.boundaries[i].max - state.boundaries[i].min);
  }
  current_best_value = Function(current_best_position);
  
  for (int i = 0;i < k_max;i++) {
    atomicMinFloat(&swarm_best_value, current_best_value);
      //printf("Updating?%f\n", current_best_value);
    __shared__ int winner;
    winner = false;
    __syncthreads();
    if (swarm_best_value == current_best_value &&
        atomicMax(&winner, true) == false) {
      for (int j = 0;j < Dim;j++) {
        swarm_best_position[j] = current_best_position[j];
      }
    }
    __syncthreads();
    if (tid > 128 * 7) {
      for (int j = 0;j < Dim;j++) {
        current_position[j] = curand_uniform(rand_state) * (state.boundaries[i].max - state.boundaries[i].min);
      }
    } else {
      for (int j = 0;j < Dim;j++) {
        float r_p = curand_uniform(rand_state);
        float r_g = curand_uniform(rand_state);
        current_velocity[j] = current_velocity[j] * omega
        + phi_p  * r_p * (current_position[j] - current_best_position[j])
        + phi_g * r_g * (current_position[j] - swarm_best_position[j]);
      }
      for (int j = 0;j < Dim;j++) {
        current_position[j] += current_velocity[j];
        current_position[j] = dev_max(current_position[j], state.boundaries[j].min);
        current_position[j] = dev_min(current_position[j], state.boundaries[j].max);
      }
    }
    current_value = Function(current_position);
    if (current_value < current_best_value) {
      for(int j = 0;j < Dim;j++) {
        current_best_position[j] = current_position[j];
      }
      current_best_value = current_value;
    }
  }
  __syncthreads();
  devValues[bid] = swarm_best_value;
  for (int i = 0;i < Dim;i++) {
    devArgs[bid][i] = swarm_best_position[i];
  }
}

template<pointerFunction_t Function, int Dim> 
class ParticleSwarm : public GlobalOptimumFinder<Function, Dim> {
public:
  ParticleSwarm() : GlobalOptimumFinder<Function, Dim> () {
  }
  
  void findMinimum(std::vector <float> &out, int n) {
    RandomGenerator gen(n);
    int no_blocks = n / kBlock + (n % kBlock != 0);

    State state = {this->devDomainSpace_, gen.getState()};
    DevArray<float> devValues(no_blocks);
    DevArray<float[Dim]> devArgs(no_blocks);

    kernel<Function, Dim><<<no_blocks, kBlock>>>(state, n, devValues.data, devArgs.data, 5000, 0.729, 2.05, 5.05);

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