#ifndef FO_PFUNC_HEADER
#define FO_PFUNC_HEADER
/*!
  Author: MaikuZ
*/

/// Templatised prefix func. <Type, Function>pfunc1024(Type *Array)
/// Function to be used as the second template argument has to be:
/// __device__
/// it has to be associative.
/// For instance: pfunc1024<int, dev_sum> is psum1024

template<class T>
__device__ T dev_min(T a, T b) {
  return a < b ? a : b;
}

template<class T>
__device__ T dev_max(T a, T b) {
  return a > b ? a : b;
}

template<class T>
__device__ T dev_sum(T a, T b) {
  return a + b;
}

template<class Type, Type(*Function)(Type, Type) >
__device__ void pfunc32(Type * shArr) {
  int laneIdx = threadIdx.x % 32;
  for (int i = 0, k = 1;i < 5;i++, k *= 2) {
    if (laneIdx - k >= 0)  
      shArr[laneIdx] = Function(shArr[laneIdx - k], shArr[laneIdx]);
    __syncwarp();
  }
}

// pfuncs all 1024 sized blocks, concurrently
template<class Type, Type(*Function)(Type, Type) >
__device__ void pfunc1024(int n, Type * arr) {
  int tid = threadIdx.x;
  
  __shared__ Type shArr[kBlock];
  if (tid < n)
    shArr[tid] = arr[tid];
    
  __syncthreads();

  int w_begin = (tid / 32) * 32; // round down to the multiple of 32
  pfunc32<Type, Function>(shArr + w_begin);
  __syncthreads();

  // The array of last elements of each 32'sized block
  __shared__ Type lastArr[32];
  if (tid % 32 == 32 - 1)
    lastArr[tid / 32] = shArr[tid];
  __syncthreads(); 

  if (tid < 32)
    pfunc32<Type, Function>(lastArr);
  __syncthreads();

  if (tid >= 32 && tid < n)
    arr[tid] = Function(shArr[tid], lastArr[tid / 32 - 1]);
  __syncthreads();
}

/// pfuncs1024 all blocks concurrently
template<class Type, Type(*Function)(Type, Type) >
__device__ void pfunc1024Blocks(int n, Type * arr) {
  int bid = blockIdx.x + blockIdx.y * blockDim.x + blockIdx.z * (blockDim.x * blockDim.y);

  arr = arr + bid * kBlock;
  if (bid * kBlock > n)
    return;

  /// Each block is working on its own segment of the array - concurrently
  if ((bid + 1) * kBlock > n) {
    n = n % (kBlock + 1);
  }
  else 
    n = kBlock;  
  pfunc1024<Type, Function>(n, arr);
}

/// pfunc's all 1024 sized blocks. This resulting in:
/// [a_0, pfunc(a_0, a_1), ..., pfunc(a_0, ..., a_1023),
///  a_1024, pfunc(a_1024, a_1025), ..., pfunc(a_1024, ..., a_2047),
///  ...
///  a_1024*k, pfunc(a_1024*k, a_(1024*k + 1), ..., pfunc(a_1024*k, a_(1024*k + 1023))]
template<class Type, Type(*Function)(Type, Type) >
__global__ void stage_1(int n, Type *arr) {
  pfunc1024Blocks<Type, Function>(n, arr);
}

/// takes all the last elements from all blocks and pfuncs them and afterwards
/// they are inserted in their positions. This resulting in:
/// [a_0, pfunc(a_1, a_2), ..., pfunc(a_0, ... ,a_1023),
///  a_1024, pfunc(a_1024, a_1025), ..., pfunc(a_0, ..., a_2047),
///  a_2048, pfunc(a_2048, a_2049), ..., pfunc(a_0, ..., a_3071),
///  ...
///  a_1024*k, pfunc(a_1024*k, a_(1024*k + 1), ..., pfunc(a_0, a_(1024*k + 1023))]
template<class Type, Type(*Function)(Type, Type) >
__global__ void stage_2(int n, Type *arr) {
  int bid = blockIdx.x + blockIdx.y * blockDim.x + blockIdx.z * (blockDim.x * blockDim.y);
  int tid = threadIdx.x;
  if (bid != 0)
    return;
  
  __shared__ Type lastArr[kBlock];
  int lastElementOfBlock = kBlock * (tid + 1) - 1;
  if (lastElementOfBlock < n)
    lastArr[tid] = arr[lastElementOfBlock];
  __syncthreads();

  pfunc1024<Type, Function>(kBlock, lastArr);
  __syncthreads();

  if (lastElementOfBlock < n)
    arr[lastElementOfBlock] = lastArr[tid];
}

/// using the previous block last element holding the pfunc(a_0, ..., a_1024 * k  - 1), each 
/// thread calculates its own pfunc(a_0, ..., a_i) =  
/// Function(pfunc(a_0, ..., a_1024 * k - 1), pfunc(a_1024 * k, ..., a_i))
template<class Type, Type(*Function)(Type, Type) >
__global__ void stage_3(int n, Type *arr) {
  int bid = blockIdx.x + blockIdx.y * blockDim.x + blockIdx.z * (blockDim.x * blockDim.y);
  int tid = threadIdx.x;
  int gid = bid * kBlock + tid;
  if (bid == 0)
    return;

  int lastElementOfPreviousBlock = kBlock * bid - 1;
  if (tid != kBlock && gid < n)
    arr[gid] = Function(arr[lastElementOfPreviousBlock], arr[gid]);
}

/// pfuncs up to 1024 * 1024 sized array
template<class Type, Type(*Function)(Type, Type) >
void pfunc(int n, Type *arr) {
  assert(n <= 1024 * 1024);
  if (n <= kBlock) {
    stage_1<Type, Function><<<1, kBlock>>>(n, arr);
  }
  int no_blocks = n / kBlock + 1;
  gpuErrchk(cudaDeviceSynchronize());
  stage_1<Type, Function><<<no_blocks, kBlock>>>(n, arr);
  gpuErrchk(cudaDeviceSynchronize());
  stage_2<Type, Function><<<1, kBlock>>>(n, arr);
  gpuErrchk(cudaDeviceSynchronize());
  stage_3<Type, Function><<<no_blocks, kBlock>>>(n, arr);
}
#endif