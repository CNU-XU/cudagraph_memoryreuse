/*This code tests:
does system overhead will be too large when first calling cudaMallocAsync and explores why
*/
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>


#define SIZE 64*1024*1024 //size of allocation

cudaError_t test1() {

    int device = 0;// Choose which GPU to run on, change this on a multi-GPU system.
    std::cout << std::endl << "This code tests:  does system overhead will be too large when first calling cudaMallocAsync and explores why." << std::endl << std::endl;
    std::cout << "Only cudaMallocAsync, not cudaFreeAsync---Allocate the same size each time" << std::endl << std::endl;
    cudaError_t cudaStatus;
    int* dPtr[10];
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
        fprintf(stderr, "cudaStreamCreate failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaDeviceSetMemPool(device, memPool);//set explicit pool as current pool
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSetMemPool failed!");
        return cudaErrorInvalidValue;
    }

    unsigned long long int setVal = UINT64_MAX;//set threshold
    cudaStatus = cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolSetAttribute failed!");
        return cudaErrorInvalidValue;
    }
    float time = 0.0f;
    cudaEvent_t start, stop;
    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        return cudaErrorInvalidValue;
    }


    for (int i = 0; i < 10; i++) {
        cudaStatus = cudaEventRecord(start, stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaEventRecord failed!");
            return cudaErrorInvalidValue;
        }
        cudaStatus = cudaMallocAsync((void**)&(dPtr[i]), SIZE, stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMallocAsync failed!");
            return cudaErrorInvalidValue;
        }
        cudaStatus = cudaEventRecord(stop, stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaEventRecord failed!");
            return cudaErrorInvalidValue;
        }
        cudaStatus = cudaEventSynchronize(stop);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaEventSynchronize failed!");
            return cudaErrorInvalidValue;
        }
        cudaEventElapsedTime(&time, start, stop);
        std::cout << i + 1 << " cudaMallocAsync cost time is " << time << std::endl;
    }
    for (int i = 0; i < 10; i++) {
        cudaStatus = cudaFreeAsync(dPtr[i], stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaFreeAsync failed!");
            return cudaErrorInvalidValue;
        }
    }

    cudaStatus = cudaStreamSynchronize(stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamSynchronize failed!");
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
    return cudaSuccess;
}

cudaError_t test2() {

    int device = 0;// Choose which GPU to run on, change this on a multi-GPU system.
    std::cout << std::endl << "This code tests:  does system overhead will be too large when first calling cudaMallocAsync and explores why." << std::endl << std::endl;
    std::cout << "cudaMallocAsync+cudaFreeAsync---Allocate the same size each time" << std::endl << std::endl;
    cudaError_t cudaStatus;
    int* dPtr[10];
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
        fprintf(stderr, "cudaStreamCreate failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaDeviceSetMemPool(device, memPool);//set explicit pool as current pool
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSetMemPool failed!");
        return cudaErrorInvalidValue;
    }

    unsigned long long int setVal = UINT64_MAX;//set threshold
    cudaStatus = cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolSetAttribute failed!");
        return cudaErrorInvalidValue;
    }
    float time = 0.0f;
    cudaEvent_t start, stop;
    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        return cudaErrorInvalidValue;
    }


    for (int i = 0; i < 10; i++) {
        cudaStatus = cudaEventRecord(start, stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaEventRecord failed!");
            return cudaErrorInvalidValue;
        }
        cudaStatus = cudaMallocAsync((void**)&(dPtr[i]), SIZE, stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMallocAsync failed!");
            return cudaErrorInvalidValue;
        }
        cudaStatus = cudaEventRecord(stop, stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaEventRecord failed!");
            return cudaErrorInvalidValue;
        }
        cudaStatus = cudaEventSynchronize(stop);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaEventSynchronize failed!");
            return cudaErrorInvalidValue;
        }
        cudaStatus = cudaFreeAsync(dPtr[i], stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaFreeAsync failed!");
            return cudaErrorInvalidValue;
        }
        cudaEventElapsedTime(&time, start, stop);
        std::cout << i + 1 << " cudaMallocAsync cost time is " << time << std::endl;
    }

    cudaStatus = cudaStreamSynchronize(stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamSynchronize failed!");
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
    return cudaSuccess;
}


cudaError_t test3() {

    int device = 0;// Choose which GPU to run on, change this on a multi-GPU system.
    std::cout << std::endl << "This code tests:  does system overhead will be too large when first calling cudaMallocAsync and explores why." << std::endl << std::endl;
    std::cout << "cudaMallocAsync+cudaFreeAsync---The allocation for each request is larger than before" << std::endl << std::endl;
    cudaError_t cudaStatus;
    int* dPtr[10];
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
        fprintf(stderr, "cudaStreamCreate failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaDeviceSetMemPool(device, memPool);//set explicit pool as current pool
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSetMemPool failed!");
        return cudaErrorInvalidValue;
    }

    unsigned long long  int setVal = UINT64_MAX;//set threshold
    cudaStatus = cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemPoolSetAttribute failed!");
        return cudaErrorInvalidValue;
    }
    float time = 0.0f;
    cudaEvent_t start, stop;
    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        return cudaErrorInvalidValue;
    }


    for (int i = 0; i < 10; i++) {
        cudaStatus = cudaEventRecord(start, stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaEventRecord failed!");
            return cudaErrorInvalidValue;
        }
        cudaStatus = cudaMallocAsync((void**)&(dPtr[i]), (i+1)*SIZE, stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMallocAsync failed!");
            return cudaErrorInvalidValue;
        }
        cudaStatus = cudaEventRecord(stop, stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaEventRecord failed!");
            return cudaErrorInvalidValue;
        }
        cudaStatus = cudaEventSynchronize(stop);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaEventSynchronize failed!");
            return cudaErrorInvalidValue;
        }
        cudaStatus = cudaFreeAsync(dPtr[i], stream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaFreeAsync failed!");
            return cudaErrorInvalidValue;
        }
        cudaEventElapsedTime(&time, start, stop);
        std::cout << i + 1 << " cudaMallocAsync cost time is " << time << std::endl;
    }

    cudaStatus = cudaStreamSynchronize(stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamSynchronize failed!");
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
    return cudaSuccess;
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

    cudaStatus = test1();//Only cudaMallocAsync, not cudaFreeAsync---Allocate the same size each time
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test1 failed!");
        return 1;
    }


 //   cudaStatus = test2();//cudaMallocAsync+cudaFreeAsync---Allocate the same size each time
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test2 failed!");
        return 1;
    }

//    cudaStatus = test3();//cudaMallocAsync+cudaFreeAsync---The allocation for each request is larger than before
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test3 failed!");
        return 1;
    }
    // cudaDeviceReset must be called before exiting in order for profiling and
// tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;

}
