# Memory reuse in CUDA Graph
These test cases are written to clarify the details and attention that should be paid in programming memory reuse in CUDA Graph.

## (1) test cases for Section III-B
### 1.cu
A sample shows that  the effect of improved performance of memory reuse in one stream:  
  - test1() use cudaMallocAsync with cudaFreeAsync, do not synchronize the stream in each loop, the result shows that the memory is reused.  
  - test2() use cudaMallocAsync with cudaFreeAsync, synchronize the stream in each loop, the result shows that the memory is not reused.  
  - test3() only use cudaMallocAsync, allocate memory without freeing memory, the result shows that the memory is not reused.  
  - test4() use cudaMallocAsync with cudaFreeAsync, set the threshold to UNIT_MAX and synchronize the stream in each loop, the result shows that the memory is reused.  
  - test5() use cudaMallocAsync with cudaFreeAsync, set the threshold to 32MB and synchronize the stream in each loop, the result shows that part of the memory is reused.  
  - test6() use cudaMalloc with cudaFree.  
  - test7() only use cudaMalloc, allocate memory without freeing memory.  
### 2.cu
A sample confirms that  the memory pool has a granularity of 32MB.
### 3.cu
A sample shows that the memory pool will not be defragmented during the stream operations.  And, the memory pool will be managed with a simple strategy and the fragments will be handled at a subsequent time.
## (2) test cases for Section III-C
### 4.cu
A sample  shows that two conclusions: if the threshold is set, the physical memory will be released to the threshold during every synchronization operation; if the threshold is not set, the physical memory will be fully released during every synchronization operation.
### 5.cu
A sample compares that performance differences of cudaMallocAsync between setting the threshold and do not set the threshold. 
### 6.cu
A sample shows that the total physical memory does not change after setting the threshold for the memory pool.
### 7.cu
A sample shows that system overhead will be too large when first calling cudaMallocAsync and explores why.
### 8.cu
A sample tests whether or not the implicit synchronization API frees the memory pool.
### 9.cu
A sample shows that physical memory is actually allocated after calling cudaMallocAsync.
## (3) test cases for Section V-B
### 10.cu
A sample tests that the situation of memory reuse: the memory size requested by new alloc node is larger than, less than or equal to the memory size requested by alloc node.
### 11.cu
A sample compares that with the differences between two kinds of execution orders: allocA->freeA->allocB and allocA->allocB->freeA.
### 12.cu
A sample tests that whether or not can the memory of the previous node be reused when memory size of the previous node is 1GB, and memory size of the two subsequent nodes is 0.5GB.
## (4) test cases for Section V-C
### 13.cu
A sample shows three finds: (1) An instantiated graph which is launched in multiple streams, even if there is no alloc and free nodes in this graph, it can only be executed serially; (2) A graph which has no alloc nodes can be instantiated to multiple executable graphs; (3) A graph which has alloc nodes cannot be instantiated to multiple executable graphs.
## (5) test cases for Section V-D
### 14.cu
A sample shows that all unused memory in the graph memory pool is not released back to the OS during every synchronization operation.
### 15.cu
A sample confirms that the allocation comes from the graph memory pool instead of the default memory pool when launching an allocation graph.
### 16.cu
A sample that shows all graphs share the same memory pool.
## (6) test cases for Section V-E
### 17.cu
A sample shows that graph memory should avoid frequent trim operations.
### 18.cu
A sample shows that cudaGraphUpload can complete the mapping first and move it out of critical path, which can significantly reduce the launch overhead.

## (7) Supported Platform
### GPU
Our study have been conducted on a NVIDIA GPU V100 server.  However,  other GPUs such as A100 and H100 should also be supported. 

### Supported Operating System

All kinds of Linux such as Ubuntu 20.04.

### CPU Architecture

x86_64, ppc64le, armv7l

## (8) Prerequisites

GPU driver should be installed, then download and install the [CUDA Toolkit 11.7](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.

## (9) Build and Run

The Linux samples are built using Makefile. To use the Makefile, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make
$ ./test_case
```
If you want to remove all .o files
```
$ cd <sample_dir>
$ make clean
```

