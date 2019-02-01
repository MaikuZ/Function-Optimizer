#ifndef FO_UTILITIES_HEADER
#define FO_UTILITIES_HEADER
/*!
  Author: MaikuZ
*/
#include <cuda.h>
#include <curand_kernel.h>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

constexpr float kEpsilon = 0.0001;
constexpr float INF = 1 << 30;
constexpr int kMaximumDimension = 16;
constexpr int kBlock = 1024;

__device__ float atomicMinFloat (float * p, float v) {
  int *addr_as_i = (int*) p;
  int old = *p, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr_as_i, assumed, __float_as_int(fminf(v, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

typedef float(*pointerFunction_t)(float * Arg);

struct Bound {
  float min;
  float max;
};

struct State {
  Bound * boundaries;
  curandState * rand_state;
};

HD inline bool equals(float a, float b) {
  if (abs(a - b) < kEpsilon)
    return true;
  return false;
}

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

template <class T> 
struct DevArray {
  DevArray(int size) : size(size) {
    gpuErrchk(cudaMalloc(&data, sizeof(T) * size));
  }

  ~DevArray() {
    gpuErrchk(cudaFree(data));
  }

  void FromHost(const T * hostData, int n) {
    assert(n < size);
    gpuErrchk(cudaMemcpy(&data, sizeof(T) * n, cudaMemcpyHostToDevice));
  }

  void ToHost(T * hostData) {
    gpuErrchk(cudaMemcpy(&data, sizeof(T) * size, cudaMemcpyDeviceToHost));
  }

  int size;
  T * data;
};

#endif