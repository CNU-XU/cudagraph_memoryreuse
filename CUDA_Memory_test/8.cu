/*This code tests:Does the implicit synchronization API free the memory pool*/
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

struct usageStatistics {
    cuuint64_t reserved;
    cuuint64_t reservedHigh;
    cuuint64_t used;
    cuuint64_t usedHigh;
};
cudaError_t poolAttrGet(cudaMemPool_t memPool, struct usageStatistics* statistics)
{
    std::cout << "-------MemPool Attribute-------" << std::endl;
    cudaError_t cudaStatus;
    cudaStatus = cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent, &(statistics->reserved));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolGetAttribute failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, &(statistics->reservedHigh));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolGetAttribute failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent, &(statistics->used));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolGetAttribute failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, &(statistics->usedHigh));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolGetAttribute failed!");
        return cudaErrorInvalidValue;
    }
    std::cout << "reserved is     : " << statistics->reserved << std::endl;
    std::cout << "reservedHigh is : " << statistics->reservedHigh << std::endl;
    std::cout << "used is         : " << statistics->used << std::endl;
    std::cout << "usedHigh is     : " << statistics->usedHigh << std::endl << std::endl;
    return cudaSuccess;
}

cudaError_t test1() {
    std::cout << std::endl << "This code tests:Does the implicit synchronization API free the memory pool" << std::endl << std::endl;
    int device = 0;
    cudaError_t cudaStatus;
    int* bn = (int*)malloc(1 << 30);
    int* cn = 0ULL;
    int* dn = 0ULL;
    struct usageStatistics statistics = { 0,0,0,0 };
    cudaStatus = cudaMalloc((void**)&cn, 1<<30);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaErrorInvalidValue;
    }
    cudaMemPoolProps poolProps = { };//set pool properties
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.location.id = 0;
    poolProps.location.type = cudaMemLocationTypeDevice;
    poolProps.handleTypes = cudaMemHandleTypeNone;
    cudaMemPool_t memPool;
    cudaStream_t stream;
    cudaStatus = cudaMemPoolCreate(&memPool, &poolProps);//create explicit pool
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolCreate failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamCreate(&stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolCreate failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaDeviceSetMemPool(device, memPool);//set explicit pool as current pool
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSetMemPool failed!");
        return cudaErrorInvalidValue;
    }


    cudaStatus = cudaMallocAsync((void**)&dn, 1 << 30, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocAsync failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamSynchronize(stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamSynchronize failed!");
        return cudaErrorInvalidValue;
    }
    std::cout << "after mallocasync " << std::endl;
    cudaStatus = poolAttrGet(memPool, &statistics);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "poolAttrGet failed!");
        return cudaErrorInvalidValue;
    }


 
    cudaStatus = cudaFreeAsync(dn, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaFreeAsync failed!");
        return cudaErrorInvalidValue;
    }
    std::cout << "after freeasync" << std::endl;
    cudaStatus = poolAttrGet(memPool, &statistics);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "poolAttrGet failed!");
        return cudaErrorInvalidValue;
    }


    cudaStatus = cudaMemcpy(bn, cn, 1 << 30, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaErrorInvalidValue;
    }
    std::cout << "after cudaMemcpy" << std::endl;
    cudaStatus = poolAttrGet(memPool, &statistics);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "poolAttrGet failed!");
        return cudaErrorInvalidValue;
    }


    cudaStatus = cudaFree(cn);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaFree failed!");
        return cudaErrorInvalidValue;
    }
    std::cout << "after cudaFree" << std::endl;
    cudaStatus = poolAttrGet(memPool, &statistics);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "poolAttrGet failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaStreamSynchronize(stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamSynchronize failed!");
        return cudaErrorInvalidValue;
    }
    std::cout << "after streamsync" << std::endl;
    cudaStatus = poolAttrGet(memPool, &statistics);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "poolAttrGet failed!");
        return cudaErrorInvalidValue;
    }
     
    cudaStatus = cudaMemPoolDestroy(memPool);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolDestroy failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamDestroy(stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamDestroy failed!");
        return cudaErrorInvalidValue;
    }
    return cudaStatus;
}



int main()
{
    cudaError_t cudaStatus;
    int device = 0;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    int driverVersion = 0;
    int deviceSupportsMemoryPools = 0;

    cudaStatus = cudaDriverGetVersion(&driverVersion);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDriverGetVersion failed!");
        return 1;
    }
    printf("Driver version is: %d.%d\n", driverVersion / 1000,
        (driverVersion % 100) / 10);

    if (driverVersion < 11040) {
        printf("Waiving execution as driver does not support Graph Memory Nodes\n");
        return 1;
    }

    cudaStatus = cudaDeviceGetAttribute(&deviceSupportsMemoryPools,
        cudaDevAttrMemoryPoolsSupported, device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute failed!");
        return 1;
    }
    if (!deviceSupportsMemoryPools) {
        printf("Waiving execution as device does not support Memory Pools\n");
        return 1;
    }
    else {
        printf("Running sample.\n");
    }

    cudaStatus = test1();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test1 failed!");
        return 1;
    }


    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


