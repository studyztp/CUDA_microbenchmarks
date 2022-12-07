// author: Zhantong Qiu
// email: ztqiu@ucdavis.edu
// last edited: 12/2/2022
#include <iostream>
#include <fstream>
#include <cstdlib>

__global__ 
void gpu_bw(int* arr, int* res, int* clk, bool* participated, int arr_size) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    uint32_t start = 0, stop = 0;

    for(int i = index; i < arr_size; i += stride) {
        arr[i] = res[i];
    }
    __syncthreads();
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
    for(int i = index; i < arr_size; i += stride) {
        res[i] = arr[i];
    }
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    __syncthreads();
    clk[index] = (int)(stop-start);
    participated[index] = index < arr_size ? true : false;
}

int main(int argc, char const *argv[]) {
    int test_size = 1<<21; //2M elements
    int blocks = 1;
    int maxBlock = 48;
    int threads = 256;
    int *clk, *res, *arr;
    bool *participated;
    std::ofstream file;
    std::string filename = "scale_data.csv";
    for(int ari = 1; ari < argc; ari++) {
        if(!strcmp(argv[ari],"-b")) {
            if (ari < argc) {
                maxBlock = atoi(argv[++ari]);
            } else {
                std::cout << "please pass in maximun block number \n";
                exit(-1);
            }
        } else if (!strcmp(argv[ari],"-t")) {
            if (ari < argc) {
                threads = atoi(argv[++ari]);
            } else {
                std::cout << "please pass in thread number \n";
                exit(-1);
            }
        } else if (!strcmp(argv[ari],"-m")) {
            if (ari < argc) {
                test_size = atoi(argv[++ari]);
            } else {
                std::cout << "please pass in test size in bytes\n";
                exit(-1);
            }
        } else if (!strcmp(argv[ari],"-n")) {
            if (ari < argc) {
                filename = argv[++ari];
                filename += ".csv";
            } else {
                std::cout << "please pass in number of iterations\n";
                exit(-1);
            }
        } else if (!strcmp(argv[ari], "-h")) {
            std::cout << "[-b #] can setup maximun number of blocks (default 48)\n" <<
                        "[-t #] can setup number of threads (default 256)\n" <<
                        "[-m #] can setup test memory size (default 2MB)\n" <<
                        "[-n #] can setup output filename (default as " <<
                        "scale_data)\n" << std::endl;
            exit(-1);
        } else {
            std::cout << argv[ari] << " is not vaild \n";
            std::cout << "[-b #] can setup maximun number of blocks (default 48)\n" <<
                        "[-t #] can setup number of threads (default 256)\n" <<
                        "[-m #] can setup test memory size (default 2MB)\n" <<
                        "[-n #] can setup output filename (default as " <<
                        "scale_data)\n" <<std::endl;
            exit(-1);
        }
    }
    int arr_size = test_size/sizeof(int)/2;
    int* check = new int [threads*maxBlock];
    file.open(filename);
    file << "blocks" <<", " << "BW" << ", " << "time" << std::endl;
    cudaError err;
    cudaMallocManaged(&arr, arr_size*sizeof(int));
    cudaMallocManaged(&res, arr_size*sizeof(int));
    cudaMallocManaged(&clk, threads*maxBlock*sizeof(int));
    cudaMallocManaged(&participated, threads*maxBlock*sizeof(bool));

    for(int i = 0; i < arr_size; i ++) {
        arr[i] = i;
        res[i] = i;
    }
    for(int i = 0; i < threads*maxBlock; i++) {
        check[i] = i+2;
    }
    float time = 0;
    float bw = 0;
    int counter = 0;
    for(blocks = 1; blocks <= maxBlock; blocks++) {
        gpu_bw<<<blocks,threads>>>(arr,res,clk,participated,arr_size);
        cudaDeviceSynchronize();
        gpu_bw<<<blocks,threads>>>(arr,res,clk,participated,arr_size);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("error: %s\n", cudaGetErrorString(err));
            file.close();
            exit(-1);
        }
        time = 0;
        counter = 0;
        for(int z = 0; z < blocks*threads; z ++) {
            if(participated[z]){
                counter ++;
                time += clk[z];
            }
            if(check[z]==res[z])
                printf("matched");
        }
        time = (1/(1365*pow(10,6)))*time/counter;
        bw = ((2*test_size)/time)/pow(10,9);
        file << blocks << ", " << bw << ", " << time << std::endl;
    }
    delete [] check;
    file.close();

    return 0;
}