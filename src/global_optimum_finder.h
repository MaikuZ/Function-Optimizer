#ifndef FO_GLOBAL_OPTIMUM_FINDER_HEADER
#define FO_GLOBAL_OPTIMUM_FINDER_HEADER
/*!
  Author: MaikuZ
*/
#include <vector>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include "utilities.h"

template<pointerFunction_t Function, int Dim> 
class GlobalOptimumFinder {
public:
  GlobalOptimumFinder() {
    assert((Dim > 0) && (Dim < kMaximumDimension));
    gpuErrchk(cudaMalloc(&devDomainSpace_, Dim * sizeof(Bound)));
    // Fill default values
    thrust::device_ptr<Bound> dev_ptr(devDomainSpace_);
    thrust::fill(dev_ptr, dev_ptr + Dim, Bound {-2, 2});
  }
  ~GlobalOptimumFinder() {
    gpuErrchk(cudaFree(devDomainSpace_));
  }
  virtual void findMinimum(std::vector<float> &out, int n) = 0;
  
  void setDomainSpace(std::vector<Bound> &boundaries) {
    if (Dim != boundaries.size()) 
      return;

    thrust::device_ptr<Bound > dev_ptr(this->devDomainSpace_);
    thrust::copy(boundaries.begin(), boundaries.end(), dev_ptr);
    return;
  }

  void setDomainSpace(std::vector<Bound> &&boundaries) {
    if (Dim != boundaries.size()) 
      return;

    thrust::device_ptr<Bound > dev_ptr(this->devDomainSpace_);
    thrust::copy(boundaries.begin(), boundaries.end(), dev_ptr);
    return;
  }
protected:
  Bound * devDomainSpace_;
};

__global__ void calculateWinners(float *T, int *Winner, float value, int n) {
  int bid = blockIdx.x + blockIdx.y * blockDim.x + blockIdx.z * (blockDim.x * blockDim.y);
  int tid = threadIdx.x;
  int gid = tid + bid * kBlock;
  if (gid < n) {
    if(T[gid] == value) {
      Winner[gid] = gid;
    } else {
      Winner[gid] = INF;
    }
  }
}
#endif