#include "CL/sycl.hpp"
#include <iostream>
#include <iostream>
#include <fstream>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>

using namespace std;

using namespace cl::sycl;


//****************************************************
//                 Programs
//****************************************************

//*****************************
//        Vec-Add
//*****************************

void vecAddRunAKernelOnOneDevice(int size){

    int* a= (int*)malloc(sizeof(int) * size);
    int* b= (int*)malloc(sizeof(int) * size);
    for(int i=0; i < size; ++i) {
        a[i] = i;
        b[i] = i*2;
    }
    
    int* c= (int*)malloc(sizeof(int) * size);
    
    { // By sticking all the SYCL work in a {} block, we ensure
        // all SYCL tasks must complete before exiting the block
        
        //TODO: we can only work with host, why???
        host_selector device_selector_host;

        // Create a queue to work on
        queue myQueue(device_selector_host);
        
        // Create buffers from a & b vectors with 2 different syntax
        buffer<int> A (a, size);
        buffer<int> B (b,size);
        
        // A buffer of N float using the storage of c
        buffer<int> C(c, size);
        
        /* The command group describing all operations needed for the kernel
         execution */
        myQueue.submit([&](handler &cgh) {
            // In the kernel A and B are read, but C is written
            auto ka = A.get_access<access::mode::read>(cgh);
            auto kb = B.get_access<access::mode::read>(cgh);
            auto kc = C.get_access<access::mode::write>(cgh);
            
            // Enqueue a single, simple task
            cgh.single_task<class sequential_vector>([=] () {
                for (int i = 0; i != size; i++)
                    kc[i] = (ka[i] + kb[i])/2;
            });
        }); // End of our commands for this queue
    } // End scope, so we wait for the queue to complete
    
    std::cout << "Result:" << std::endl;
    for (int i = 0; i != size; i++)
        std::cout << c[i] << " ";
    std::cout << std::endl;

}


void vecAdd(int size){
    std::cout<<" Run on one default device: \n";
    vecAddRunAKernelOnOneDevice(size);
    std::cout<<" \n\n";
    
    /*
    std::cout<<" Run on one CPU device: \n";
    vecAddRunAKernelOnOneDevice(CL_DEVICE_TYPE_CPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on one GPU device: \n";
    vecAddRunAKernelOnOneDevice(CL_DEVICE_TYPE_GPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on two devices with 50%-50%: \n";
    vecAddRunAKernelOnTwoDevices(CL_DEVICE_TYPE_ALL,size);
    std::cout<<" \n\n";
     */
}

//*****************************
//        Bit-Compression
//*****************************
void bitCompressionRunAKernelOnOneDevice(int size){
    
    uint* num_bits = (uint*) malloc(sizeof(uint) * size);
    uint4* input = (uint4*) malloc(sizeof(uint4) * size);
    for (cl_uint i = 0; i < size; ++i) {
        input[i] = (uint4){15, 15, 15, 15};
        num_bits[i] = (int)pow(2, ((i % 3) + 1));
    }
    
    uint* output = (uint*) malloc(sizeof(uint) * size);
    
    { // By sticking all the SYCL work in a {} block, we ensure
        // all SYCL tasks must complete before exiting the block
        
        //TODO: we can only work with host, why???
        host_selector device_selector_host;
        
        // Create a queue to work on
        queue myQueue(device_selector_host);
        
        // Create buffers from a & b vectors with 2 different syntax
        buffer<uint4> A (input,size);
        buffer<uint> B (num_bits,size);
        
        // A buffer of N float using the storage of c
        buffer<uint> C(output, size);
        
        /* The command group describing all operations needed for the kernel
         execution */
        myQueue.submit([&](handler &cgh) {
            // In the kernel A and B are read, but C is written
            auto input_buffer = A.get_access<access::mode::read>(cgh);
            auto num_bits_buffer = B.get_access<access::mode::read>(cgh);
            auto output_buffer = C.get_access<access::mode::write>(cgh);
            
            int workItemNo=size;
            // Enqueue a single, simple task
            cgh.parallel_for<class bit_compression>(range<1>(workItemNo),[=](item<1> myItem) {
                
                 int id = myItem.get_linear_id();
                
                 if(id >= workItemNo) return;
                
                 uint4 in = (uint4)input_buffer[id];
                 int bits = (int)num_bits_buffer[id];
                 uint tmp = 0;
                
                 if (bits == 2) {
                   tmp |= (in.x() << (32-bits)) & 3221225472u;
                   tmp |= (in.y() << (28-bits)) &  805306368u;
                   tmp |= (in.z() << (24-bits)) &  201326592u;
                   tmp |= (in.w() << (20-bits)) &   50331648u;
                  } else if (bits == 4) {
                   tmp |= (in.x() << (32-bits)) & 4026531840u;
                   tmp |= (in.y() << (28-bits)) &  251658240u;
                   tmp |= (in.z() << (24-bits)) &   15728640u;
                   tmp |= (in.w() << (20-bits)) &     983040u;
                  } else if (bits == 8) {
                   tmp |= (in.x() << (32-bits)) & 4278190080u;
                   tmp |= (in.y() << (28-bits)) &   16711680u;
                   tmp |= (in.z() << (24-bits)) &      65280u;
                   tmp |= (in.w() << (20-bits)) &        255u;
                }
                output_buffer[id] = tmp;
                
            });
        }); // End of our commands for this queue
    } // End scope, so we wait for the queue to complete
    
    std::cout << "Result:" << std::endl;
    for (int i = 0; i != size; i++)
        std::cout << output[i] << " ";
    std::cout << std::endl;
    
}




void bitCompression(int size){
    std::cout<<" Run on one default device: \n";
    bitCompressionRunAKernelOnOneDevice(size);
    std::cout<<" \n\n";
    
    /*
    std::cout<<" Run on one CPU device: \n";
    bitCompressionRunAKernelOnOneDevice(CL_DEVICE_TYPE_CPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on one GPU device: \n";
    bitCompressionRunAKernelOnOneDevice(CL_DEVICE_TYPE_GPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on two devices with 50%-50%: \n";
    bitCompressionRunAKernelOnTwoDevices(CL_DEVICE_TYPE_ALL,size);
    std::cout<<" \n\n";
     */
}

//*****************************
//        Mat-Mul
//*****************************
void matMulRunAKernelOnOneDevice(int size){
    
    int sqrtofthesize=sqrt(size);
    int width = (int)floor(sqrtofthesize);
    
    int sizeOfTheMatrix = width * width;
    
    int* input1 = (int*)malloc(sizeof(int) * sizeOfTheMatrix);
    int* input2 = (int*) malloc(sizeof(int) * sizeOfTheMatrix);
    
    for(int i=0; i < sizeOfTheMatrix; ++i) {
        input1[i] = i;
        input2[i] = i;
    }
    
    int* output = (int *)malloc(sizeof(int) * sizeOfTheMatrix);
    

    
    { // By sticking all the SYCL work in a {} block, we ensure
        // all SYCL tasks must complete before exiting the block

        host_selector device_selector_host;
        
        // Create a queue to work on
        queue myQueue(device_selector_host);
        
        // Create buffers from a & b vectors with 2 different syntax
        buffer<int> A (input1,sizeOfTheMatrix);
        buffer<int> B (input2,sizeOfTheMatrix);
        
        // A buffer of N float using the storage of c
        buffer<int> C(output, sizeOfTheMatrix);
        
        // The command group describing all operations needed for the kernel

        myQueue.submit([&](handler &cgh) {
            // In the kernel A and B are read, but C is written
            auto input1_buffer = A.get_access<access::mode::read>(cgh);
            auto input2_buffer = B.get_access<access::mode::read>(cgh);
            auto output_buffer = C.get_access<access::mode::write>(cgh);
            
            int workItemNo=sizeOfTheMatrix;
            // Enqueue a single, simple task
            cgh.parallel_for<class mat_mul>(range<1>(workItemNo),[=](item<1> myItem) {
                
                int id = myItem.get_linear_id();
                
                if (id >= workItemNo) return;
                int tx = id % width;
                int ty = id / width;
                int sum = 0;
                for (int k = 0; k < width; ++k) {
                   sum += input1_buffer[ty * width + k] * input2_buffer[k * width + tx];
                }
                output_buffer[id] = sum;
          
                
            });
        }); // End of our commands for this queue
    } // End scope, so we wait for the queue to complete
    
    std::cout << "Result:" << std::endl;
    for (int i = 0; i != sizeOfTheMatrix; i++)
        std::cout << output[i] << " ";
    std::cout << std::endl;
}


void matMul(int size){
    std::cout<<" Run on one default device: \n";
    matMulRunAKernelOnOneDevice(size);
    std::cout<<" \n\n";
    
    /*
    std::cout<<" Run on one CPU device: \n";
    matMulRunAKernelOnOneDevice(CL_DEVICE_TYPE_CPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on one GPU device: \n";
    matMulRunAKernelOnOneDevice(CL_DEVICE_TYPE_GPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on two devices with 50%-50%: \n";
    matMulRunAKernelOnTwoDevices(CL_DEVICE_TYPE_ALL,size);
    std::cout<<" \n\n";
    */
}

//*****************************
//        Lin-Reg
//*****************************

int float_compare(const void* elem1, const void* elem2) {
    if(*(const float*)elem1 < *(const float*)elem2) return -1;
    return *(const float*)elem1 > *(const float*)elem2;
}

inline float random01_float() {
    return (float) rand()/(RAND_MAX);
}

void fill_random_float(float* arrayPtr, int width, int height, float rangeMin, float rangeMax){
    double range = (double)(rangeMax - rangeMin);
    for(int i = 0; i < height; i++)
        for(int j = 0; j < width; j++) {
            int index = i*width + j;
            arrayPtr[index] = rangeMin + (float)(range * random01_float());
        }
}

bool compare_float(const float *refData, const float *data, const int length, const float epsilon) {
    float error = 0.0f;
    float ref = 0.0f;
    
    for(int i = 1; i < length; ++i) {
        float diff = refData[i] - data[i];
        error += diff * diff;
        ref += refData[i] * refData[i];
    }
    
    float normRef = sqrtf((float) ref);
    if (std::fabs(ref) < 1e-7f) {
        return false;
    }
    float normError = sqrtf((float) error);
    error = normError / normRef;
    
    return error < epsilon;
}


void linRegPrintTheResult(float * input1, float *input2, float *alpha, float *beta,
                          float * output, int size){
    //print out the result (on the host side)
    printf("======================\n= Linear Regression Done\n");
    float* output2 = (float *)malloc(sizeof(float) * size);
    for(unsigned int j = 0; j < size; ++j) {
        const int gid = j;
        float a = alpha[gid];
        float b = beta[gid];
        float error = 0;
        for(int i=0; i<size; i++) {
            float e = (a * input1[i] + b) - input2[i];
            error += e * e;
        }
        output2[gid] = error;
    }
    
    bool check = compare_float(output, output2, size, 0.000001);
    printf("======================\n");
    printf("Result check: %s\n", check ? "OK" : "FAIL");
    free(output2);
}


void linRegRunAKernelOnOneDevice(int size){

    float* input1 = (float*) malloc(sizeof(float) * size);
    float* input2 = (float*) malloc(sizeof(float) * size);
    float* alpha  = (float*) malloc(sizeof(float) * size);
    float* beta   = (float*) malloc(sizeof(float) * size);
    
    fill_random_float(input2, size, 1, -1.0f, 1.0f);
    qsort(input2, size, sizeof(float), float_compare);
    float step = 2.0f / size;
    for(int i=0; i < size; i++)
        input1[i] = -1.0f + i * step;
    
    fill_random_float(alpha, size, 1, -1.0f, 1.0f);
    fill_random_float(beta, size, 1, -1.0f, 1.0f);
    
    float* output = (float*) malloc(sizeof(float) * size);
    
    { // By sticking all the SYCL work in a {} block, we ensure
        // all SYCL tasks must complete before exiting the block
        
        host_selector device_selector_host;
        
        // Create a queue to work on
        queue myQueue(device_selector_host);
        
        // Create buffers from a & b vectors with 2 different syntax
        buffer<float> A (input1,size);
        buffer<float> B (input2,size);
        buffer<float> C (alpha,size);
        buffer<float> D (beta,size);
        
        // A buffer of N float using the storage of c
        buffer<float> E(output,size);
        
        // The command group describing all operations needed for the kernel
        
        myQueue.submit([&](handler &cgh) {
            // In the kernel A and B are read, but C is written
            auto input1_buffer = A.get_access<access::mode::read>(cgh);
            auto input2_buffer = B.get_access<access::mode::read>(cgh);
            auto alpha_buffer  = C.get_access<access::mode::read>(cgh);
            auto beta_buffer   = D.get_access<access::mode::read>(cgh);
            auto output_buffer = E.get_access<access::mode::write>(cgh);
            
            int workItemNo=size;
            // Enqueue a single, simple task
            cgh.parallel_for<class lin_reg>(range<1>(workItemNo),[=](item<1> myItem) {
                
                int id = myItem.get_linear_id();
                
                if (id >= workItemNo) return;
                
                float a = alpha_buffer[id];
                float b = beta_buffer[id];
                float error = 0;
                
                 for(int i=0; i<workItemNo; i++)
                {
                  float e = (a * input1_buffer[i] + b) - input2_buffer[i];
                  error += e * e;
                }
                output_buffer[id] = error;
        
                
            });
        }); // End of our commands for this queue
    } // End scope, so we wait for the queue to complete
    
 
    linRegPrintTheResult(input1,input2,alpha,beta,output,size);
 
    
   /*
    std::string kernel_code="__kernel void lin_reg(__global float* input1, __global float* input2, __global float* alpha, __global float* beta, __global float* output, int num_elements) {"
    "int gid = get_global_id(0);"
    "if (gid >= num_elements) return;"
    
    "float a = alpha[gid];"
    "float b = beta[gid];"
    "float error = 0;"
    
    " for(int i=0; i<num_elements; i++)"
    "{"
    "float e = (a * input1[i] + b) - input2[i];"
    "error += e * e;"
    "}"
    "output[gid] = error;"
    "}";
  */
}



void linReg(int size){
    std::cout<<" Run on one default device: \n";
    linRegRunAKernelOnOneDevice(size);
    std::cout<<" \n\n";
    /*
    std::cout<<" Run on one CPU device: \n";
    linRegRunAKernelOnOneDevice(CL_DEVICE_TYPE_CPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on one GPU device: \n";
    linRegRunAKernelOnOneDevice(CL_DEVICE_TYPE_GPU,size);
    std::cout<<" \n\n";
     */

}

//*****************************
//        Syr2k
//*****************************

void syr2k_check(float *A, float *B, float *C, float ALPHA, float BETA, int m, int n, int size)
{
    int i, j, k;
    int nn = size / n;
    
    for (i = 0; i < size; i++)
    {
        C[i] = (i / n + 2) * BETA;
    }
    
    
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            for (k = 0; k < m; k++)
            {
                C[i*n + j] += ALPHA * A[i*m + k] * B[j*m + k];
                C[i*n + j] += ALPHA * B[i*m + k] * A[j*m + k];
            }
        }
    }
}

void syr2kPrintTheResult(float * input_a, float * input_b,  float ALPHA, float BETA,
                         int M, int N, float * output, int newSize){
    //print out the result (on the host side)
    printf("======================\n= syr2k program working\n");
    
    float* val =  (float*)malloc(sizeof(float) * newSize);
    
    syr2k_check(input_a, input_b, val , ALPHA, BETA, M, N, newSize);
    
    unsigned int check = 1;
    for(unsigned int i = 0; i < newSize; ++i) {
        if(output[i] != val[i]) {
            check = 0;
            printf("= fail at %d, expected %f / actual %f\n", i, val[i], output[i]);
            break;
        }
    }
    printf("======================\n");
    printf("Result check: %s\n", check ? "OK" : "FAIL");
    free(val);
}

//TODO: why does it work only till 12??
void syr2kRunAKernelOnOneDevice(int size){

    int sqrtofthesize=sqrt(size);
    int width = (int)floor(sqrtofthesize);
    int newSize = width * width;
    
    float* input_a = (float*)malloc(sizeof(float) * newSize);
    float* input_b = (float*)malloc(sizeof(float) * newSize);
    float* output =  (float*)malloc(sizeof(float) * newSize);
    
    float ALPHA = 1;
    float BETA = 1;
    int M = width;
    int N = width;
    
    for(int i=0; i < size; ++i) {
        input_a[i] = i % 19;
        input_b[i] = (size - i) % 17;
        output[i] = i / M + 2;
    }
    
    
    { // By sticking all the SYCL work in a {} block, we ensure
        // all SYCL tasks must complete before exiting the block
        
        host_selector device_selector_host;
        
        // Create a queue to work on
        queue myQueue(device_selector_host);
        
        // Create buffers from a & b vectors with 2 different syntax
        buffer<float> A (input_a,newSize);
        buffer<float> B (input_b,newSize);
        
        // A buffer of N float using the storage of c
        buffer<float> C(output, newSize);
        
        // The command group describing all operations needed for the kernel
        
        myQueue.submit([&](handler &cgh) {
            // In the kernel A and B are read, but C is written
            auto input_a_buffer = A.get_access<access::mode::read>(cgh);
            auto input_b_buffer = B.get_access<access::mode::read>(cgh);
            auto output_buffer = C.get_access<access::mode::write>(cgh);
            
            int workItemNo=newSize;
            // Enqueue a single, simple task
            cgh.parallel_for<class syr2k>(range<1>(workItemNo),[=](item<1> myItem) {
                
                int id = myItem.get_linear_id();
                
                if (id >= workItemNo) return;
                int j = id % N;
                int i = id / N;
                
                if ((i < N))
                {
                  output_buffer[id] *= BETA;
                
                  int k;
                  float tmp = 0;
                  for(k = 0; k < M; k++)
                  {
                    tmp += ALPHA * input_a_buffer[i * M + k] * input_b_buffer[j * M + k] +
                           ALPHA * input_b_buffer[i * M + k] * input_a_buffer[j * M + k];
                  }
                  output_buffer[id] += tmp;
                }
                
            });
        }); // End of our commands for this queue
    } // End scope, so we wait for the queue to complete


    syr2kPrintTheResult(input_a,input_b,ALPHA,BETA,M,N,output,newSize);
 /*
    
    std::string kernel_code="__kernel void syr2k(__global float *a, __global float *b, __global float *c, float alpha, float beta, int m, int n){"
    "int j = get_global_id(0) % n;"
    "int i = get_global_id(0) / n;"
    
    "if ((i < n))"
    "{"
    "c[get_global_id(0)] *= beta;"
    
    "int k;"
    "float tmp = 0;"
    "for(k = 0; k < m; k++)"
    "{"
    " tmp += alpha * a[i * m + k] * b[j * m + k] + alpha * b[i * m + k] * a[j * m + k];"
    "}"
    "c[get_global_id(0)] += tmp;"
    "}"
    "}";
     */
}


void syr2k(int size){
    std::cout<<" Run on one default device: \n";
    syr2kRunAKernelOnOneDevice(size);
    std::cout<<" \n\n";
    /*
    std::cout<<" Run on one CPU device: \n";
    syr2kRunAKernelOnOneDevice(CL_DEVICE_TYPE_CPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on one GPU device: \n";
    syr2kRunAKernelOnOneDevice(CL_DEVICE_TYPE_GPU,size);
    std::cout<<" \n\n";
     */
}


//*****************************
//        Convolution
//*****************************

void convolutionPrintTheResult(int * input, int * mask, int * output, int width, int mask_width, int newSize){
    //print out the result (on the host side)
    printf("======================\n= Convolution Done\n");
    unsigned int check = 1;
    for(unsigned int i = 0; i < newSize; ++i) {
        int x = i % width;
        int y = i / width;
        int sum = 0;
        int offset = mask_width/2;
        if (!(x < offset || y < offset || x >= width-offset || y >= width-offset)) {
            int tmpx = x - offset;
            int tmpy = y - offset;
            for (int r = 0; r < mask_width; ++r) {
                for (int c = 0; c < mask_width; ++c) {
                    sum += mask[r * mask_width + c] * input[(tmpy + r ) * width + tmpx + c];
                }
            }
        }
        
        if(output[i] != sum) {
            check = 0;
            printf("= fail at %d, expected %d / actual %d\n", i, sum, output[i]);
            break;
        }
    }
    printf("======================\n");
    printf("Result check: %s\n", check ? "OK" : "FAIL");
}


void convolutionRunAKernelOnOneDevice(int size){

    int sqrtofthesize=sqrt(size);
    int width = (int)floor(sqrtofthesize);
    int newSize = width * width;
    
    int mask_width = 22;
    int mask_size = mask_width * mask_width;
    
    int* input  = (int*)malloc(sizeof(int) * newSize);
    int* mask   = (int*) malloc(sizeof(int) * mask_size);
    
    for(int i=0; i < mask_size; ++i){
        mask[i] = 1;
      }
    mask[mask_size/2] = 0;
    
    for(int i=0; i < newSize; ++i) {
        input[i] = 1;//rand() % 10;
      }
    
    int* output = (int*)malloc(sizeof(int) * newSize);
    
    
    { // By sticking all the SYCL work in a {} block, we ensure
        // all SYCL tasks must complete before exiting the block
        
        host_selector device_selector_host;
        
        // Create a queue to work on
        queue myQueue(device_selector_host);
        
        // Create buffers from a & b vectors with 2 different syntax
        buffer<int> A (input,newSize);
        buffer<int> B (mask,mask_size);
        
        // A buffer of N float using the storage of c
        buffer<int> C(output, newSize);
        
        // The command group describing all operations needed for the kernel
        
        myQueue.submit([&](handler &cgh) {
            // In the kernel A and B are read, but C is written
            auto input_buffer = A.get_access<access::mode::read>(cgh);
            auto mask_buffer = B.get_access<access::mode::read>(cgh);
            auto output_buffer = C.get_access<access::mode::write>(cgh);
            
            int workItemNo=newSize;
            // Enqueue a single, simple task
            cgh.parallel_for<class convolution>(range<1>(workItemNo),[=](item<1> myItem) {
                
                int id = myItem.get_linear_id();
                
                if (id >= workItemNo) return;
                
                int tx = id % width;
                int ty = id / width;
                int offset = mask_width/2;
                
                if (tx < offset || ty < offset || tx >= (width-offset) || ty >= (width-offset)) {
                   output_buffer[id] = 0;
                   return;
                }
                int sum = 0;
                
                int tmpx = tx - offset;
                int tmpy = ty - offset;
                for (int r = 0; r < mask_width; ++r) {
                  for (int c = 0; c < mask_width; ++c) {
                     sum += mask_buffer[r * mask_width + c] * input_buffer[(tmpy + r ) * width + tmpx + c];
                   }
                }
                output_buffer[id] = sum;
                
            });
        }); // End of our commands for this queue
    } // End scope, so we wait for the queue to complete
  
    convolutionPrintTheResult(input,mask,output,width,mask_width,newSize);
}



void convolution(int size){
    
    std::cout<<" Run on one default device: \n";
    convolutionRunAKernelOnOneDevice(size);
    std::cout<<" \n\n";
    
    /*
    std::cout<<" Run on one CPU device: \n";
    convolutionRunAKernelOnOneDevice(CL_DEVICE_TYPE_CPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on one GPU device: \n";
    convolutionRunAKernelOnOneDevice(CL_DEVICE_TYPE_GPU,size);
    std::cout<<" \n\n";
     */
}


//*****************************
// Main Functions
//*****************************

void sixKernels(int size, int programmNumber){

    if (programmNumber==1)
        vecAdd(size);
    if (programmNumber==2)
        bitCompression(size);
    if (programmNumber==3)
        matMul(size);
    if (programmNumber==4)
        linReg(size);
    if (programmNumber==5)
        syr2k(size);
    if (programmNumber==6)
        convolution(size);
}

int main() {
    
	std::cout << "Starting now..." << std::endl;
    int size, programmNumber;
    
    cout << "Enter the program number that you would like to test now:  \n";
    cout << "1. Vector Addition \n";
    cout << "2. Bit Compression \n";
    cout << "3. Matrix Multiplication \n";
    cout << "4. Linear Regession \n";
    cout << "5. Syr2k \n";
    cout << "6. Convolution \n";
    
    cin >> programmNumber; // input the program number
    std::cout<<" \n";
    
    cout << "Enter the size: ";
    cin >> size; // input the size of the input for each kernel
    std::cout<<" \n";
    
    //std::cout<<size<<" ";
    
    sixKernels(size,programmNumber);
    
    std::cout<<" \n";
    
    return 0;
}
