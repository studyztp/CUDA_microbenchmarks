// author: Zhantong Qiu
// email: ztqiu@ucdavis.edu
// last edited: 12/2/2022
#include <iostream>
#include <fstream>
#include <cstdlib>

__global__ 
void share_mem_stride(int* final_next, int* time, int num_itr, int shared_ele, int arr_stride) {
    uint32_t start = 0, stop = 0;
    int tid = threadIdx.x;
    extern __shared__ int loc_arr [];
    if(tid == 0) {
        for(int i = 0; i < shared_ele; i ++)
        {
            loc_arr[i]=(i+arr_stride)%shared_ele;
        }
    }
    __syncthreads();
    int next = tid*arr_stride;
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
    for (int i = 0; i < num_itr; i++) {
        next = loc_arr[next];
    }
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    __syncthreads();
    final_next[tid+blockDim.x*blockIdx.x] = next;
    time[tid+blockDim.x*blockIdx.x] = (((int)(stop-start))/num_itr);
}

__global__
void mem_stride(int* final_next, int* time, int* test_arr, int num_itr, int arr_stride) {
    uint32_t start = 0, stop = 0;
    int tid = threadIdx.x;
    int next = tid*arr_stride;
    for (int i = 0; i < num_itr; i++) {
        next = test_arr[next];
    }
    next = tid*arr_stride;
    __syncthreads();
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
    for (int i = 0; i < num_itr; i++) {
        next = test_arr[next];
    }
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    __syncthreads();
    final_next[tid] = next;
    time[tid] = (((int)(stop-start))/num_itr);
}

int main(int argc, char const *argv[]) {
    int* time;
    int* final_next;
    int blocks = 1;
    int threads = 1;
    int num_itr = 100000;
    int shared_mem = 65536;
    int shared_ele = -1;
    bool plot_all_threads = false;
    bool plot_mem = false;
    std::string filename = "share_mem_data.csv";
    cudaError err;

    for(int ari = 1; ari < argc; ari++) {
        if(!strcmp(argv[ari],"-b")) {
            if (ari < argc) {
                blocks = atoi(argv[++ari]);
            }
            else {
                std::cout << "please pass in block number \n";
                exit(-1);
            }
        } else if (!strcmp(argv[ari],"-t")) {
            if (ari < argc) {
                threads = atoi(argv[++ari]);
            }
            else {
                std::cout << "please pass in thread number \n";
                exit(-1);
            }
        } else if (!strcmp(argv[ari], "-i")){
            if (ari < argc) {
                num_itr = atoi(argv[++ari]);
            }
            else {
                std::cout << "please pass in iteration number \n";
                exit(-1);
            }
        } else if (!strcmp(argv[ari], "-m")){
            if (ari < argc) {
                shared_mem = atoi(argv[++ari]);
            }
            else {
                std::cout << "please pass in shared memory size \n";
                exit(-1);
            }
        } else if (!strcmp(argv[ari], "-s")){
            if (ari < argc) {
                shared_ele = atoi(argv[++ari]);
            }
            else {
                std::cout << "please pass in shared memory test array element "
                    <<"number \n";
                exit(-1);
            }
        } else if (!strcmp(argv[ari], "-n")){
            if (ari < argc) {
                filename = argv[++ari];
                filename += ".csv";
            }
            else {
                std::cout << "please pass in an output filename\n";
                exit(-1);
            }
        } else if (!strcmp(argv[ari], "-p")){
            plot_all_threads = true;
        } else if (!strcmp(argv[ari], "-q")){
            plot_mem = true;
        }else if (!strcmp(argv[ari], "-h")) {
            std::cout << "[-b #] can setup number of blocks (default 1)\n" <<
                        "[-t #] can setup number of threads (default 1)\n" <<
                        "[-i #] can setup number of iteration (default 100000)\n" <<
                        "[-m #] can setup share memory size (default 65536)\n" <<
                        "[-s #] can setup number of test array elements "<<
                        "(default share memory size/4)\n"<<
                        "[-n #] can setup output filename (default as " <<
                        "share_mem_data)\n" <<
                        "[-p] output csv with average latency cycle from " <<
                        "thread 1 to thread 1024 (default false)\n" <<
                        "[-q] output csv with average latency cycle from " <<
                        "thread 1 to thread 1024 for non-shared memory"<<
                        "(default false)\n" <<
                        "The binary default allocate 64KB dynamic shared " <<
                        "memory per block\n";
            exit(-1);
        } else {
            std::cout << argv[ari] << " is not vaild \n";
            std::cout << "[-b #] can setup number of blocks (default 1)\n" <<
                        "[-t #] can setup number of threads (default 1)\n" <<
                        "[-i #] can setup number of iteration (default 100000)\n" <<
                        "[-m #] can setup share memory size (default 65536)\n" <<
                        "[-s #] can setup number of test array elements "<<
                        "(default share memory size/4)\n"<<
                        "[-n #] can setup output filename (default as " <<
                        "share_mem_data)\n" <<
                        "[-p] output csv with average latency cycle from " <<
                        "thread 1 to thread 1024 (default false)\n" <<
                        "[-q] output csv with average latency cycle from " <<
                        "thread 1 to thread 1024 for non-shared memory"<<
                        "(default false)\n" <<
                        "The binary default allocate 64KB dynamic shared " <<
                        "memory per block\n";
            exit(-1);
        }
    }
    if(shared_ele < 0)
        shared_ele = shared_mem/4;
    std::ofstream file;
    int sum = 0;
    
    
    cudaFuncSetAttribute(share_mem_stride, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
    file.open(filename);
    int final_sum = 0;
    file << "stride" << ", " <<"threads"<<", "<< "time" << ", "  << "final next" << std::endl;
    if (plot_mem) {
        cudaMallocManaged(&time, 1024*sizeof(int));
        cudaMallocManaged(&final_next, 1024*sizeof(int));
        int* test_arr;
        cudaMallocManaged(&test_arr,shared_ele*sizeof(int));

        for (int num_thread = 1; num_thread <= 1024; num_thread *=2) {
            // printf("starting with %i threads in use\n",num_thread);
            for (int stride = 1; stride <= 32; stride *=2) {
                // printf("starting with stride:%i\n",stride);
                if(stride*(num_thread-1)<shared_ele){
                    for(int i = 0; i < shared_ele; i ++)
                    {
                        test_arr[i]=(i+stride)%shared_ele;
                    }
                    mem_stride<<<1,num_thread>>>(final_next, time, test_arr, num_itr, stride);
                    cudaDeviceSynchronize();
                    mem_stride<<<1,num_thread>>>(final_next, time, test_arr, num_itr, stride);
                    cudaDeviceSynchronize();
                    err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        printf("error: %s\n", cudaGetErrorString(err));
                        file.close();
                        exit(-1);
                    }
                    sum = 0;
                    
                    for(int i = 0; i < num_thread; i++) {
                        sum += time[i];
                        final_sum += final_next[i];
                    }
                    file << stride << ", " << num_thread<<", " << sum/num_thread/blocks << ", " << final_sum << std::endl;
                } 

            }
        }

    } else if(!plot_all_threads){
        cudaMallocManaged(&time, threads*blocks*sizeof(int));
        cudaMallocManaged(&final_next, threads*blocks*sizeof(int));
        for (int stride = 1; stride <= shared_ele*4; stride *=2) {
            // printf("starting with stride:%i\n",stride);
            share_mem_stride<<<blocks,threads,shared_mem>>>(final_next, time, num_itr, shared_ele, stride);
            cudaDeviceSynchronize();
            share_mem_stride<<<blocks,threads,shared_mem>>>(final_next, time, num_itr, shared_ele, stride);
            cudaDeviceSynchronize();
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("error: %s\n", cudaGetErrorString(err));
                file.close();
                exit(-1);
            }
            sum = 0;
            
            for(int i = 0; i < threads*blocks; i++) {
                sum += time[i];
                final_sum += final_next[i];
            }
            file << stride << ", " << threads<<", " << sum/threads/blocks << ", " << final_sum << std::endl;
            // printf("latency is %i cycles\n",sum/threads/blocks);
        }
    }  else {
        cudaMallocManaged(&time, 1024*blocks*sizeof(int));
        cudaMallocManaged(&final_next, 1024*blocks*sizeof(int));
        for (int num_thread = 1; num_thread <= 1024; num_thread *=2) {
            // printf("starting with %i threads in use\n",num_thread);
            for (int stride = 1; stride <= 32; stride *=2) {
                // printf("starting with stride:%i\n",stride);
                if(stride*(num_thread-1)<shared_ele){
                    share_mem_stride<<<blocks,num_thread,shared_mem>>>(final_next, time, num_itr, shared_ele, stride);
                    cudaDeviceSynchronize();
                    share_mem_stride<<<blocks,num_thread,shared_mem>>>(final_next, time, num_itr, shared_ele, stride);
                    cudaDeviceSynchronize();
                    err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        printf("error: %s\n", cudaGetErrorString(err));
                        file.close();
                        exit(-1);
                    }
                    sum = 0;
                    
                    for(int i = 0; i < num_thread*blocks; i++) {
                        sum += time[i];
                        final_sum += final_next[i];
                    }
                    file << stride << ", " << num_thread<<", " << sum/num_thread/blocks << ", " << final_sum << std::endl;
                } 
                // printf("latency is %i cycles\n",sum/num_thread/blocks);
            }
        }
    }
    
    printf("number of blocks:%i\nnumber of iterations:%i\n", blocks, num_itr);
    printf("max share memory size:%i\n", shared_mem);
    printf("length of array(4 bytes each element):%i\n",shared_ele);
    printf("output filename:%s\n",filename.c_str());
    file.close();
    return 0;
}