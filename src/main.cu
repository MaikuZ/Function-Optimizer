/*!
  Author: MaikuZ
*/
#include <iostream>
#include <map>
#include <vector>
#include <string>

#include "particle_swarm.h"
#include "simulated_annealing.h"
#include "functions.h"
#include "function_list.h"
#include "random_gen.h"


void write_argument_list(std::vector<float> &arg) {
  std::cout << "(";
  for (int i = 0;i < arg.size() - 1;i++)
    std::cout << arg[i] << ", ";
  std::cout << arg.back() << ")";
}

#define TEST_ALL_FUNCTIONS_PS(NAME, DIM, OPT_VAL, BOUNDS) \
{\
  std::cout << "Testing function: '" << #NAME << "', dimensions: " << DIM << ", minimum value: " << OPT_VAL << std::endl; \
  std::vector<float> best_arg(DIM);\
  ParticleSwarm<NAME, DIM> finder;\
  finder.findMinimum(best_arg, 1024 * 1024);    \
  std::cout << "ParticleSwarm found the minimum at: ";\
  write_argument_list(best_arg);\
  std::cout << std::endl << "The value is: " << NAME(best_arg.data()) << std::endl;\
}\

#define TEST_ALL_FUNCTIONS_SA(NAME, DIM, OPT_VAL, BOUNDS) \
{\
  std::cout << "Testing function: '" << #NAME << "', dimensions: " << DIM << ", minimum value: " << OPT_VAL << std::endl; \
  std::vector<float> best_arg(DIM);\
  SimulatedAnnealing<NAME, temperature, DIM> finder;\
  finder.setDomainSpace(BOUNDS());\
  finder.findMinimum(best_arg,  1024 * 1024);    \
  std::cout << "Simulated Annealing found the minimum at: ";\
  write_argument_list(best_arg);\
  std::cout << std::endl << "The value is: " << NAME(best_arg.data()) << std::endl;\
}\

__global__ void cuda_test(bool *working, int device) {
  printf("Running on device: %d\n", device);
  *working = true;
} 

bool is_cuda_working(int device_no) {
  bool *d_check;
  gpuErrchk(cudaMalloc(&d_check, sizeof(bool) * 1));
  cuda_test<<<1,1>>>(d_check, device_no);
  bool is_working;
  gpuErrchk(cudaMemcpy(&is_working, d_check, sizeof(bool) * 1, cudaMemcpyDeviceToHost));
  return is_working;
}

int main() {
  srand(time(0));
  int device_no = 3;
  gpuErrchk(cudaSetDevice(device_no));
  if(!is_cuda_working(device_no)) {
    std::cout << "cuda is not working on device: " << device_no << std::endl;
    return 0;
  }

  FOR_EACH_TEST_FUNCTION(TEST_ALL_FUNCTIONS_PS)
  FOR_EACH_TEST_FUNCTION(TEST_ALL_FUNCTIONS_SA)

  std::cout << "All functions were tested!" << std::endl;
  return 0;
}