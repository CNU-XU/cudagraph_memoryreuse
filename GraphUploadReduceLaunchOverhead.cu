﻿/*This code tests: upload separates memory allocation and mapping from launch*/
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>


#define LOOPTIMES 1000
#define SIZE (long long)1024*1024*1024


struct usageStatistics {
    cuuint64_t reserved;
    cuuint64_t reservedHigh;
    cuuint64_t used;
    cuuint64_t usedHigh;
};
cudaError_t GraphPoolAttrGet(int  device, struct usageStatistics* statistics)
{
    std::cout << "-------Graph MemPool Attribute-------" << std::endl;
    cudaError_t cudaStatus;
    cudaStatus = cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrReservedMemCurrent, &(statistics->reserved));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetGraphMemAttribute failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrReservedMemHigh, &(statistics->reservedHigh));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetGraphMemAttribute failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrUsedMemCurrent, &(statistics->used));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetGraphMemAttribute failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrUsedMemHigh, &(statistics->usedHigh));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetGraphMemAttribute failed!");
        return cudaErrorInvalidValue;
    }
    std::cout << "reserved is     : " << statistics->reserved << std::endl;
    std::cout << "reservedHigh is : " << statistics->reservedHigh << std::endl;
    std::cout << "used is         : " << statistics->used << std::endl;
    std::cout << "usedHigh is     : " << statistics->usedHigh << std::endl << std::endl;
    return cudaSuccess;
}


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


__global__ void clockBlock(clock_t clock_count) {
    unsigned int start_clock = (unsigned int)clock();

    clock_t clock_offset = 0;

    while (clock_offset < clock_count) {
        unsigned int end_clock = (unsigned int)clock();
        clock_offset = (clock_t)(end_clock - start_clock);
    }
}

cudaError_t test1() {
    std::cout << "Do not upload the graph before launch in this test" << std::endl << std::endl;
    int device = 0;
    cudaError_t cudaStatus;
    int* d_a = NULL;
    cudaStream_t stream;
    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreateWithFlags failed!");
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
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    float kernelTime = 50;  // time for each thread to run in microseconds
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!");
        return cudaErrorInvalidValue;
    }
    clock_t time_clocks = (clock_t)((kernelTime / 1000.0) * deviceProp.clockRate);


    cudaStatus = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamBeginCapture failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaMallocAsync((void**)&d_a, SIZE, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocAsync failed!");
        return cudaErrorInvalidValue;
    }
    clockBlock << <1, 1, 0, stream >> > (time_clocks);
    cudaStatus = cudaFreeAsync(d_a, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaFreeAsync failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamEndCapture(stream, &graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamEndCapture failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);//instantiate graph
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaEventRecord(start, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }
    for (int i = 0; i < LOOPTIMES; i++) {
        cudaStatus = cudaGraphLaunch(graphExec, stream);//launch graph
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphLaunch failed!");
            return cudaErrorInvalidValue;
        }
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
    cudaStatus = cudaEventElapsedTime(&time, start, stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventElapsedTime failed!");
        return cudaErrorInvalidValue;
    }
    printf("time is %f\n\n", time);
    cudaStatus = cudaGraphDestroy(graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphDestroy failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphExecDestroy(graphExec);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphExecDestroy failed!");
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
    std::cout << "upload the graph before launch in this test" << std::endl << std::endl;
    int device = 0;
    cudaError_t cudaStatus;
    int* d_a = NULL;
    int* d_b = NULL;
    cudaStream_t stream;
    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreateWithFlags failed!");
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
    cudaGraph_t graph,graph1;
    cudaGraphExec_t graphExec,graphExec1;

    float kernelTime = 50;  // time for each thread to run in microseconds
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!");
        return cudaErrorInvalidValue;
    }
    clock_t time_clocks = (clock_t)((kernelTime / 1000.0) * deviceProp.clockRate);


    cudaStatus = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamBeginCapture failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaMallocAsync((void**)&d_a, SIZE, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocAsync failed!");
        return cudaErrorInvalidValue;
    }
    clockBlock << <1, 1, 0, stream >> > (time_clocks);
    cudaStatus = cudaFreeAsync(d_a, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaFreeAsync failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamEndCapture(stream, &graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamEndCapture failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);//instantiate graph
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamBeginCapture failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaMallocAsync((void**)&d_b, SIZE, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocAsync failed!");
        return cudaErrorInvalidValue;
    }
    clockBlock << <1, 1, 0, stream >> > (time_clocks);
    cudaStatus = cudaFreeAsync(d_b, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaFreeAsync failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamEndCapture(stream, &graph1);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamEndCapture failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaGraphInstantiate(&graphExec1, graph1, NULL, NULL, 0);//instantiate graph
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphUpload(graphExec, stream);//upload graph 
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphUpload failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphUpload(graphExec1, stream);//upload graph 
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphUpload failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaEventRecord(start, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }
    std::cout<<"launch different graph in same stream"<<std::endl;
    for (int i = 0; i < LOOPTIMES; i++) {
        cudaStatus = cudaGraphLaunch(graphExec, stream);//launch graph
        cudaStatus = cudaGraphLaunch(graphExec1, stream);//launch graph
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphLaunch failed!");
            return cudaErrorInvalidValue;
        }
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
    cudaStatus = cudaEventElapsedTime(&time, start, stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventElapsedTime failed!");
        return cudaErrorInvalidValue;
    }
    printf("time is %f\n\n", time);
    cudaStatus = cudaGraphDestroy(graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphDestroy failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphExecDestroy(graphExec);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphExecDestroy failed!");
        return cudaErrorInvalidValue;
    }
     cudaStatus = cudaGraphDestroy(graph1);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphDestroy failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphExecDestroy(graphExec1);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphExecDestroy failed!");
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
    std::cout << "upload the graph before launch in this test" << std::endl << std::endl;
    int device = 0;
    cudaError_t cudaStatus;
    int* d_a = NULL;
    cudaStream_t stream;
    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreateWithFlags failed!");
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
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    float kernelTime = 50;  // time for each thread to run in microseconds
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!");
        return cudaErrorInvalidValue;
    }
    clock_t time_clocks = (clock_t)((kernelTime / 1000.0) * deviceProp.clockRate);


    cudaStatus = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamBeginCapture failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaMallocAsync((void**)&d_a, SIZE, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocAsync failed!");
        return cudaErrorInvalidValue;
    }
    clockBlock << <1, 1, 0, stream >> > (time_clocks);
    cudaStatus = cudaFreeAsync(d_a, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaFreeAsync failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamEndCapture(stream, &graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamEndCapture failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);//instantiate graph
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed!");
        return cudaErrorInvalidValue;
    }
    
    cudaStatus = cudaGraphUpload(graphExec, stream);//upload graph 
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphUpload failed!");
        return cudaErrorInvalidValue;
    }
    std::cout<<"launch same graph in same stream"<<std::endl;
    cudaStatus = cudaEventRecord(start, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }
    for (int i = 0; i < 2*LOOPTIMES; i++) {
        cudaStatus = cudaGraphLaunch(graphExec, stream);//launch graph
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphLaunch failed!");
            return cudaErrorInvalidValue;
        }
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
    cudaStatus = cudaEventElapsedTime(&time, start, stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventElapsedTime failed!");
        return cudaErrorInvalidValue;
    }
    printf("time is %f\n\n", time);
    cudaStatus = cudaGraphDestroy(graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphDestroy failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphExecDestroy(graphExec);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphExecDestroy failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamDestroy(stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamDestroy failed!");
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}


cudaError_t test4() {
    std::cout << "Upload the graph before launch in this test" << std::endl << std::endl;
    int device = 0;
    cudaError_t cudaStatus;
    int* d_a = NULL;
    cudaStream_t stream;
    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreateWithFlags failed!");
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
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    float kernelTime = 50;  // time for each thread to run in microseconds
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!");
        return cudaErrorInvalidValue;
    }
    clock_t time_clocks = (clock_t)((kernelTime / 1000.0) * deviceProp.clockRate);


    cudaStatus = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamBeginCapture failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaMallocAsync((void**)&d_a, SIZE, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocAsync failed!");
        return cudaErrorInvalidValue;
    }
    clockBlock << <1, 1, 0, stream >> > (time_clocks);
    cudaStatus = cudaFreeAsync(d_a, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaFreeAsync failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamEndCapture(stream, &graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamEndCapture failed!");
        return cudaErrorInvalidValue;
    }

    cudaStatus = cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);//instantiate graph
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed!");
        return cudaErrorInvalidValue;
    }



    cudaStatus = cudaEventRecord(start, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphUpload(graphExec, stream);//upload graph 
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphUpload failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaEventRecord(stop, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaEventSynchronize(stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaEventElapsedTime(&time, start, stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }
    printf("upload time is %f\n", time);

    cudaStatus = cudaEventRecord(start, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }
    for (int i = 0; i < LOOPTIMES; i++) {
        cudaGraphLaunch(graphExec, stream);//launch graph
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaEventRecord failed!");
            return cudaErrorInvalidValue;
        }
    }

    cudaStatus = cudaEventRecord(stop, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaEventSynchronize(stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaEventElapsedTime(&time, start, stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed!");
        return cudaErrorInvalidValue;
    }
    printf("launch time is %f\n", time);
    cudaStatus = cudaGraphDestroy(graph);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphDestroy failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaGraphExecDestroy(graphExec);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphExecDestroy failed!");
        return cudaErrorInvalidValue;
    }
    cudaStatus = cudaStreamDestroy(stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamDestroy failed!");
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}
int main(int argc, char** argv)
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

    std::cout << std::endl << "This code tests: upload separates memory allocation and mapping from launch" << std::endl << std::endl;

    cudaStatus = test1(); //no upload
    cudaStatus = test2(); //upload launch different graph in same stream
    cudaStatus = test3(); //upload launch same graph in same stream
    cudaStatus = test4(); //upload
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "test failed!");
        return 1;
    }


    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

