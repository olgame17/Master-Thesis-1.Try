#include <iostream>
#include <OpenCL/cl.hpp>
#include <fstream>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
//#include "CL/sycl.hpp"
using namespace std;

//*****************************
// OpenCL Defintion Functions
//*****************************

cl::Platform getADefaultPlatform(std::vector<cl::Platform> all_platforms){
    
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
    
    return default_platform;
}

cl::Device getOneDevice(cl::Platform default_platform,std::vector<cl::Device> all_devices, cl_device_type type){
    
    default_platform.getDevices(type, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device device=all_devices[0];
    std::cout<< "Using device: "<<device.getInfo<CL_DEVICE_NAME>()<<"\n";
    return device;
}


std::vector<cl::Device> getTwoDevices(cl::Platform default_platform,std::vector<cl::Device> all_devices){
    
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    
    cl::Device device1=all_devices[0];
    cl::Device device2=all_devices[1];
    
    std::cout<< "Using device: "<<device1.getInfo<CL_DEVICE_NAME>()<<" and \n";
    std::cout<< "using device: "<<device2.getInfo<CL_DEVICE_NAME>()<<"\n";
    return all_devices;
}

void programBuild(cl::Device default_device, cl::Program program){
    if(program.build({default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
        exit(1);
    }
}

//****************************************************
// Kernel Defintion Functions (specific to a Kernel)
//****************************************************

//*****************************
//        Vec-Add
//*****************************

void vecAddCopyInputHostArrayToDeviceArray(int * A,int * B,cl::Buffer buffer_A,cl::Buffer buffer_B,cl::CommandQueue queue, int size){
    //write arrays A and B to the device
    queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(int)*size,A);
    queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(int)*size,B);
}


void vecAddRunTheKernel(cl::Kernel kernel_add,cl::Program program,
                        cl::Buffer buffer_A,cl::Buffer buffer_B,
                        cl::Buffer buffer_C,cl::CommandQueue queue,
                        int size){
    
    kernel_add.setArg(0,buffer_A);
    kernel_add.setArg(1,buffer_B);
    kernel_add.setArg(2,buffer_C);
    kernel_add.setArg(3,size);
    queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(size),cl::NullRange);
    queue.finish();
}

void vecAddCopyOutputDeviceArrayToHostArray(int * C,cl::Buffer buffer_C,
                                            cl::CommandQueue queue,int size){
    //read result C from the device to array C
    queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(int)*size,C);
    
}

void vecAddPrintTheResult(int * C, int size){
    //print out the result (on the host side)
    printf("======================\n= Vector Addition\n");
    std::cout<<" result: \n";
    for(int i=0;i<size;i++){
        std::cout<<C[i]<<" ";
    }
}

void vecAddPrintTheResultForTwoDevices(int * C1,int * C2,int halfOhTheSize){
    //print out the result (on the host side)
    printf("======================\n= Vector Addition\n");
    std::cout<<" result: \n";
    for(int i=0;i<halfOhTheSize;i++){
        std::cout<<C1[i]<<" ";
    }
    for(int i=0;i<halfOhTheSize;i++){
        std::cout<<C2[i]<<" ";
    }
}

//*****************************
//        Bit-Compression
//*****************************

void bitCompressionCopyInputHostArrayToDeviceArray(cl_uint4 * input,cl_uint * num_bits,cl::Buffer buffer_A,cl::Buffer buffer_B,cl::CommandQueue queue, int size){
    //write arrays input and num_bits to the device
    queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(cl_uint4)*size,input);
    queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(cl_uint)*size,num_bits);
}


void bitCompressionRunTheKernel(cl::Kernel bit_compression,cl::Program program,
                        cl::Buffer buffer_A,cl::Buffer buffer_B,
                        cl::Buffer buffer_C,cl::CommandQueue queue,
                        int size){
    
    bit_compression.setArg(0,buffer_A);
    bit_compression.setArg(1,buffer_B);
    bit_compression.setArg(2,buffer_C);
    bit_compression.setArg(3,size);
    queue.enqueueNDRangeKernel(bit_compression,cl::NullRange,cl::NDRange(size),cl::NullRange);
    queue.finish();
}

void bitCompressionCopyOutputDeviceArrayToHostArray(cl_uint * C,cl::Buffer buffer_C,
                                            cl::CommandQueue queue,int size){
    //read result C from the device to array C
    queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(cl_uint)*size,C);
    
}

void bitCompressionPrintTheResult(cl_uint * C, int size){
    //print out the result (on the host side)
    printf("======================\n= Bit Compression Done\n");
    std::cout<<" result: \n";
    for(int i=0;i<size;i++){
        std::cout<<C[i]<<" ";
    }
}

void bitCompressionPrintTheResultForTwoDevices(cl_uint *output1, cl_uint *output2,int halfOfTheSize){
    //print out the result (on the host side)
    printf("======================\n= Bit Compression Done\n");
    std::cout<<" result: \n";
    for(int i=0;i<halfOfTheSize;i++){
        std::cout<<output1[i]<<" ";
    }
    for(int i=0;i<halfOfTheSize;i++){
        std::cout<<output2[i]<<" ";
    }
}

//*****************************
//        Mat-Mul
//*****************************

void matMulCopyInputHostArrayToDeviceArray(int * input1,int * input2,cl::Buffer buffer_A,cl::Buffer buffer_B,cl::CommandQueue queue, int size){
    //write arrays input and num_bits to the device
    queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(int)*size,input1);
    queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(int)*size,input2);
}


void matMulRunTheKernel(cl::Kernel mat_mul,cl::Program program,
                                cl::Buffer buffer_A,cl::Buffer buffer_B,
                                cl::Buffer buffer_C,cl::CommandQueue queue,
                                int size, int width){
    
    mat_mul.setArg(0,buffer_A);
    mat_mul.setArg(1,buffer_B);
    mat_mul.setArg(2,buffer_C);
    mat_mul.setArg(3,size);
    mat_mul.setArg(4,width);
    queue.enqueueNDRangeKernel(mat_mul,cl::NullRange,cl::NDRange(size),cl::NullRange);
    queue.finish();
}

void matMulCopyOutputDeviceArrayToHostArray(int * C,cl::Buffer buffer_C,
                                                    cl::CommandQueue queue,int size){
    //read result C from the device to array C
    queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(int)*size,C);
    
}

void matMulPrintTheResult(int * input1, int *input2, int * output, int size, int width){
    //print out the result (on the host side)
    printf("======================\n= Matrix Multiplication Done\n");
    unsigned int check = 1;
    for(unsigned int i = 0; i < size; ++i) {
        int sum = 0;
        int x = i % width;
        int y = i / width;
        for(unsigned int k = 0; k < width; ++k)
            sum += input1[y * width + k] * input2[k * width +  x];
        
        if(output[i] != sum) {
            check = 0;
            printf("= fail at %d, expected %d / actual %d\n", i, sum, output[i]);
            break;
        }
    }
    printf("======================\n");
    printf("Result check: %s\n", check ? "OK" : "FAIL");
}

void  matMulPrintTheResultOnTwoDevices(int * input1device1,int * input1device2,
                                       int * input2device1,int * input2device2,
                                       int * output1,int * output2,
                                       int halfSizeOfTheMatrix,int newWidth){
    //print out the result (on the host side)
    printf("======================\n= Matrix Multiplication Done\n");
    unsigned int check = 1;
    for(unsigned int i = 0; i < halfSizeOfTheMatrix; ++i) {
        int sum = 0;
        int x = i % newWidth;
        int y = i / newWidth;
        for(unsigned int k = 0; k < newWidth; ++k)
            sum += input1device1[y * newWidth + k] * input2device1[k * newWidth +  x];
        
        if(output1[i] != sum) {
            check = 0;
            printf("= fail at %d, expected %d / actual %d\n", i, sum, output1[i]);
            break;
        }
    }
    for(unsigned int i = 0; i < halfSizeOfTheMatrix; ++i) {
        int sum = 0;
        int x = i % newWidth;
        int y = i / newWidth;
        for(unsigned int k = 0; k < newWidth; ++k)
            sum += input1device2[y * newWidth + k] * input2device2[k * newWidth +  x];
        
        if(output1[i] != sum) {
            check = 0;
            printf("= fail at %d, expected %d / actual %d\n", i, sum, output2[i]);
            break;
        }
    }
    printf("======================\n");
    printf("Result check: %s\n", check ? "OK" : "FAIL");
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
    if (fabs(ref) < 1e-7f) {
        return false;
    }
    float normError = sqrtf((float) error);
    error = normError / normRef;
    
    return error < epsilon;
}

void linRegCopyInputHostArrayToDeviceArray(float * input1, float * input2,
                                           float * alpha, float * beta,
                                           cl::Buffer buffer_A,cl::Buffer buffer_B,
                                           cl::Buffer buffer_C,cl::Buffer buffer_D,
                                           cl::CommandQueue queue, int size){
    //write arrays input1,input2,alpha,beta to the device
    queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(float)*size,input1);
    queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(float)*size,input2);
    queue.enqueueWriteBuffer(buffer_C,CL_TRUE,0,sizeof(float)*size,alpha);
    queue.enqueueWriteBuffer(buffer_D,CL_TRUE,0,sizeof(float)*size,beta);
}


void linRegRunTheKernel(cl::Kernel lin_reg,cl::Program program,
                        cl::Buffer buffer_A,cl::Buffer buffer_B,
                        cl::Buffer buffer_C,cl::Buffer buffer_D,
                        cl::Buffer buffer_E,cl::CommandQueue queue,int size){
    
    lin_reg.setArg(0,buffer_A);
    lin_reg.setArg(1,buffer_B);
    lin_reg.setArg(2,buffer_C);
    lin_reg.setArg(3,buffer_D);
    lin_reg.setArg(4,buffer_E);
    lin_reg.setArg(5,size);
    queue.enqueueNDRangeKernel(lin_reg,cl::NullRange,cl::NDRange(size),cl::NullRange);
    queue.finish();
}

void linRegCopyOutputDeviceArrayToHostArray(float * output,cl::Buffer buffer_E,
                                            cl::CommandQueue queue,int size){
    //read result C from the device to array C
    queue.enqueueReadBuffer(buffer_E,CL_TRUE,0,sizeof(float)*size,output);
    
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

void  linRegPrintTheResultTwoDevices(float * input1device1,float * input1device2,
                                     float * input2device1,float * input2device2,
                                     float * alpha1,float * alpha2,
                                     float * beta1, float * beta2,
                                     float * output1,float * output2,
                                     int halfOfTheSize){
    //print out the result (on the host side)
    printf("======================\n= Linear Regression Done\n");
    float* outputNew1 = (float *)malloc(sizeof(float) * halfOfTheSize);
    float* outputNew2 = (float *)malloc(sizeof(float) * halfOfTheSize);
    for(unsigned int j = 0; j < halfOfTheSize; ++j) {
        const int gid = j;
        float a = alpha1[gid];
        float b = beta1[gid];
        float error = 0;
        for(int i=0; i<halfOfTheSize; i++) {
            float e = (a * input1device1[i] + b) - input2device1[i];
            error += e * e;
        }
        outputNew1[gid] = error;
    }
    
    bool check = compare_float(output1, outputNew1, halfOfTheSize, 0.000001);
    
    bool check2=false;
    if (check){
        for(unsigned int j = 0; j < halfOfTheSize; ++j) {
            const int gid = j;
            float a = alpha2[gid];
            float b = beta2[gid];
            float error = 0;
            for(int i=0; i<halfOfTheSize; i++) {
                float e = (a * input1device2[i] + b) - input2device2[i];
                error += e * e;
            }
            outputNew2[gid] = error;
        }
        check2 = compare_float(output2, outputNew2, halfOfTheSize, 0.000001);
        
    }
    
    printf("======================\n");
    printf("Result check: %s\n", check2 ? "OK" : "FAIL");
    free(outputNew1);
    free(outputNew2);
}

//*****************************
//        Sobel-Filter
//*****************************

void sobelFilterCopyInputHostArrayToDeviceArray(float * input1, float * input2,
                                           float * alpha, float * beta,
                                           cl::Buffer buffer_A,cl::Buffer buffer_B,
                                           cl::Buffer buffer_C,cl::Buffer buffer_D,
                                           cl::CommandQueue queue, int size){
    //write arrays input1,input2,alpha,beta to the device
    queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(float)*size,input1);
    queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(float)*size,input2);
    queue.enqueueWriteBuffer(buffer_C,CL_TRUE,0,sizeof(float)*size,alpha);
    queue.enqueueWriteBuffer(buffer_D,CL_TRUE,0,sizeof(float)*size,beta);
}


void sobelFilterRunTheKernel(cl::Kernel lin_reg,cl::Program program,
                        cl::Buffer buffer_A,cl::Buffer buffer_B,
                        cl::Buffer buffer_C,cl::Buffer buffer_D,
                        cl::Buffer buffer_E,cl::CommandQueue queue,int size){
    
    lin_reg.setArg(0,buffer_A);
    lin_reg.setArg(1,buffer_B);
    lin_reg.setArg(2,buffer_C);
    lin_reg.setArg(3,buffer_D);
    lin_reg.setArg(4,buffer_E);
    lin_reg.setArg(5,size);
    queue.enqueueNDRangeKernel(lin_reg,cl::NullRange,cl::NDRange(size),cl::NullRange);
    queue.finish();
}

void sobelFilterCopyOutputDeviceArrayToHostArray(float * output,cl::Buffer buffer_E,
                                            cl::CommandQueue queue,int size){
    //read result C from the device to array C
    queue.enqueueReadBuffer(buffer_E,CL_TRUE,0,sizeof(float)*size,output);
    
}

void sobelFilterPrintTheResult(float * input1, float *input2, float *alpha, float *beta,
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

//*****************************
//        Syr2k
//*****************************

void syr2kCopyInputHostArrayToDeviceArray(float * input_a, float * input_b,
                                                float * output, cl::Buffer buffer_A,
                                                cl::Buffer buffer_B, cl::Buffer buffer_C,
                                                cl::CommandQueue queue, int newSize){
    //write arrays input_a,input_b,output to the device
    queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(float)*newSize,input_a);
    queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(float)*newSize,input_b);
    queue.enqueueWriteBuffer(buffer_C,CL_TRUE,0,sizeof(float)*newSize,output);
}

void syr2kRunTheKernel(cl::Kernel syr2k ,cl::Program program,
                             cl::Buffer buffer_A,cl::Buffer buffer_B,
                             cl::Buffer buffer_C, float ALPHA, float BETA,
                            int M, int N, cl::CommandQueue queue,int newSize){
    
    syr2k.setArg(0,buffer_A);
    syr2k.setArg(1,buffer_B);
    syr2k.setArg(2,buffer_C);
    syr2k.setArg(3,ALPHA);
    syr2k.setArg(4,BETA);
    syr2k.setArg(5,M);
    syr2k.setArg(6,N);
    queue.enqueueNDRangeKernel(syr2k,cl::NullRange,cl::NDRange(newSize),cl::NullRange);
    queue.finish();
}

void syr2kCopyOutputDeviceArrayToHostArray(float * output,cl::Buffer buffer_C,
                                                 cl::CommandQueue queue,int newSize){
    //read result C from the device to array C
    queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(float)*newSize,output);
}

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

//*****************************
//        Convolution
//*****************************

void convolutionCopyInputHostArrayToDeviceArray(int * input, int * mask,
                                          int * output, cl::Buffer buffer_A,
                                          cl::Buffer buffer_B, cl::Buffer buffer_C,
                                          cl::CommandQueue queue, int newSize, int mask_size){
    //write arrays input_a,input_b,output to the device
    queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(int)*newSize,input);
    queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(int)*mask_size,mask);
    queue.enqueueWriteBuffer(buffer_C,CL_TRUE,0,sizeof(int)*newSize,output);
}


void convolutionRunTheKernel(cl::Kernel convolution ,cl::Program program,
                       cl::Buffer buffer_A,cl::Buffer buffer_B,
                       cl::Buffer buffer_C, int width, int mask_width,
                        cl::CommandQueue queue,int newSize){
    
    convolution.setArg(0,buffer_A);
    convolution.setArg(1,buffer_B);
    convolution.setArg(2,buffer_C);
    convolution.setArg(3,newSize);
    convolution.setArg(4,width);
    convolution.setArg(5,mask_width);
    queue.enqueueNDRangeKernel(convolution,cl::NullRange,cl::NDRange(newSize),cl::NullRange);
    queue.finish();
}

void convolutionCopyOutputDeviceArrayToHostArray(int * output,cl::Buffer buffer_C,
                                           cl::CommandQueue queue,int newSize){
    //read result C from the device to array C
    queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(int)*newSize,output);
}


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


//****************************************************
//                 Programs
//****************************************************

//*****************************
//        Vec-Add
//*****************************
void vecAddRunAKernelOnOneDevice(cl_device_type type, int size){
    std::vector<cl::Platform> all_platforms;
    cl::Platform default_platform=getADefaultPlatform(all_platforms);
    
    std::vector<cl::Device> all_devices;
    cl::Device device=getOneDevice(default_platform,all_devices,type);
    
    cl::Context context(type);
    
    cl::Program::Sources sources;
    
    //TODO: why do we need to devide it to two?????
    std::string kernel_code=    "__kernel void vec_add(__global int* input1, __global int* input2, __global int* output, int num_elements) {"
    "    int gid = get_global_id(0);"
    "   if (gid >= num_elements) return;"
    "    output[gid] = (input1[gid] + input2[gid])/2;"
    "}";
    
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    
    cl::Program program(context,sources);
    programBuild(device, program);
    
    //define Buffer names
    //1.on Host
    int* A = (int*)malloc(sizeof(int) * size);
    int* B= (int*)malloc(sizeof(int) * size);
    for(int i=0; i < size; ++i) {
        A[i] = i;
        B[i] = i*2;
    }
    //2. on Device
    cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(int)*size);
    cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(int)*size);
    cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(int)*size);
    
    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,device);
    
    vecAddCopyInputHostArrayToDeviceArray(A,B,buffer_A,buffer_B,queue,size);
    
    cl::Kernel kernel_add=cl::Kernel(program,"vec_add");
    vecAddRunTheKernel(kernel_add,program,buffer_A,buffer_B,buffer_C,queue,size);
    
    int* C= (int*)malloc(sizeof(int) * size);
    vecAddCopyOutputDeviceArrayToHostArray(C,buffer_C,queue,size);
    
    vecAddPrintTheResult(C,size);
    
    free(A);
    free(B);
    free(C);
}

void vecAddRunAKernelOnTwoDevices(cl_device_type type, int size){
    std::vector<cl::Platform> all_platforms;
    cl::Platform default_platform=getADefaultPlatform(all_platforms);
    
    std::vector<cl::Device> all_devices;
    std::vector<cl::Device> devices=getTwoDevices(default_platform,all_devices);
    
    //CPU and GPU share the same context
    cl::Context context(type);
    
    cl::Program::Sources sources;
    
    //TODO: why do we need to devide it to two?????
    std::string kernel_code=    "__kernel void vec_add(__global int* input1, __global int* input2, __global int* output, int num_elements) {"
    "    int gid = get_global_id(0);"
    "   if (gid >= num_elements) return;"
    "    output[gid] = (input1[gid] + input2[gid])/2;"
    "}";
    
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    
    //two differnet programs to each of the devices
    cl::Program program1(context,sources);
    cl::Program program2(context,sources);
    
    programBuild(devices[0], program1);
    programBuild(devices[1], program2);
    
    //define Buffer names
    //1.on Host
    int halfOhTheSize=size/2;
    int* A1 = (int*)malloc(sizeof(int) * halfOhTheSize);
    int* A2 = (int*)malloc(sizeof(int) * halfOhTheSize);
    int* B1 = (int*)malloc(sizeof(int) * halfOhTheSize);
    int* B2 = (int*)malloc(sizeof(int) * halfOhTheSize);
    for(int i=0; i < halfOhTheSize; ++i) {
        A1[i] = i;
        A2[i] = i+halfOhTheSize;
        B1[i] = i*2;
        B2[i] = (i+halfOhTheSize)*2;
    }
    if((size%2)!= 0){
        // TODO!!
    }

    //2. on Device
    cl::Buffer buffer_A1(context,CL_MEM_READ_WRITE,sizeof(int)*halfOhTheSize);
    cl::Buffer buffer_A2(context,CL_MEM_READ_WRITE,sizeof(int)*halfOhTheSize);
    cl::Buffer buffer_B1(context,CL_MEM_READ_WRITE,sizeof(int)*halfOhTheSize);
    cl::Buffer buffer_B2(context,CL_MEM_READ_WRITE,sizeof(int)*halfOhTheSize);
    cl::Buffer buffer_C1(context,CL_MEM_READ_WRITE,sizeof(int)*halfOhTheSize);
    cl::Buffer buffer_C2(context,CL_MEM_READ_WRITE,sizeof(int)*halfOhTheSize);
    
    //create queue to which we will push commands for the device.
    cl::CommandQueue queue1(context,devices[0]);
    cl::CommandQueue queue2(context,devices[1]);
    
    vecAddCopyInputHostArrayToDeviceArray(A1,B1,buffer_A1,buffer_B1,queue1,halfOhTheSize);
    vecAddCopyInputHostArrayToDeviceArray(A2,B2,buffer_A2,buffer_B2,queue2,halfOhTheSize);
    
    cl::Kernel kernel_add_device1=cl::Kernel(program1,"vec_add");
    cl::Kernel kernel_add_device2=cl::Kernel(program2,"vec_add");
    
    vecAddRunTheKernel(kernel_add_device1,program1,buffer_A1,buffer_B1,buffer_C1,queue1,halfOhTheSize);
    vecAddRunTheKernel(kernel_add_device2,program2,buffer_A2,buffer_B2,buffer_C2,queue2,halfOhTheSize);
    
    int* C1= (int*)malloc(sizeof(int) * halfOhTheSize);
    int* C2= (int*)malloc(sizeof(int) * halfOhTheSize);
    
    vecAddCopyOutputDeviceArrayToHostArray(C1,buffer_C1,queue1,halfOhTheSize);
    vecAddCopyOutputDeviceArrayToHostArray(C2,buffer_C2,queue2,halfOhTheSize);
    
    vecAddPrintTheResultForTwoDevices(C1,C2,halfOhTheSize);
    
    free(A1);
    free(A2);
    free(B1);
    free(B2);
    free(C1);
    free(C2);
}

void vecAdd(int size){
    std::cout<<" Run on one default device: \n";
    vecAddRunAKernelOnOneDevice(CL_DEVICE_TYPE_ALL,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on one CPU device: \n";
    vecAddRunAKernelOnOneDevice(CL_DEVICE_TYPE_CPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on one GPU device: \n";
    vecAddRunAKernelOnOneDevice(CL_DEVICE_TYPE_GPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on two devices with 50%-50%: \n";
    vecAddRunAKernelOnTwoDevices(CL_DEVICE_TYPE_ALL,size);
    std::cout<<" \n\n";
}

//*****************************
//        Bit-Compression
//*****************************
void bitCompressionRunAKernelOnOneDevice(cl_device_type type, int size){
    std::vector<cl::Platform> all_platforms;
    cl::Platform default_platform=getADefaultPlatform(all_platforms);
    
    std::vector<cl::Device> all_devices;
    cl::Device device=getOneDevice(default_platform,all_devices,type);
    
    cl::Context context(type);
    
    cl::Program::Sources sources;
    
    //TODO: why do we need to devide it to two?????
    std::string kernel_code=    "__kernel void bit_compression(__global uint4* input, __global uint* num_bits, __global uint* output, int length) {"
    "  int gid = get_global_id(0);"
    "if(gid >= length) return;"
    
    "  uint4 in = input[gid];"
    " int bits = num_bits[gid];"
    "  uint tmp = 0;"
    "  if (bits == 2) {"
    "    tmp |= (in.x << (32-bits)) & 3221225472u;"
    "    tmp |= (in.y << (28-bits)) &  805306368u;"
    "    tmp |= (in.z << (24-bits)) &  201326592u;"
    "    tmp |= (in.w << (20-bits)) &   50331648u;"
    "  } else if (bits == 4) {"
    "    tmp |= (in.x << (32-bits)) & 4026531840u;"
    "    tmp |= (in.y << (28-bits)) &  251658240u;"
    "    tmp |= (in.z << (24-bits)) &   15728640u;"
    "    tmp |= (in.w << (20-bits)) &     983040u;"
    "  } else if (bits == 8) {"
    "    tmp |= (in.x << (32-bits)) & 4278190080u;"
    "    tmp |= (in.y << (28-bits)) &   16711680u;"
    "    tmp |= (in.z << (24-bits)) &      65280u;"
    "    tmp |= (in.w << (20-bits)) &        255u;"
    " }"
    " output[gid] = tmp;"
    "}";
    
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    
    cl::Program program(context,sources);
    programBuild(device, program);
    
    //define Buffer names
    //1.on Host
    cl_uint* num_bits = (cl_uint*) malloc(sizeof(cl_uint) * size);
    cl_uint4* input = (cl_uint4*) malloc(sizeof(cl_uint4) * size);
    for (cl_uint i = 0; i < size; ++i) {
        input[i] = (cl_uint4){15, 15, 15, 15};
        num_bits[i] = (int)pow(2, ((i % 3) + 1));
        
    }
    
    //2. on Device
    cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(cl_uint4)*size);
    cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(cl_uint)*size);
    cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(cl_uint)*size);
    
    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,device);
    
    bitCompressionCopyInputHostArrayToDeviceArray(input,num_bits,buffer_A,buffer_B,queue,size);
    
    cl::Kernel bit_compression=cl::Kernel(program,"bit_compression");
    bitCompressionRunTheKernel(bit_compression,program,buffer_A,buffer_B,buffer_C,queue,size);
    
    cl_uint* output = (cl_uint*) malloc(sizeof(cl_uint) * size);
    bitCompressionCopyOutputDeviceArrayToHostArray(output,buffer_C,queue,size);
    
    bitCompressionPrintTheResult(output,size);
    
    free(num_bits);
    free(input);
    free(output);
}

void bitCompressionRunAKernelOnTwoDevices(cl_device_type type, int size){
    std::vector<cl::Platform> all_platforms;
    cl::Platform default_platform=getADefaultPlatform(all_platforms);
    
    std::vector<cl::Device> all_devices;
    std::vector<cl::Device> devices=getTwoDevices(default_platform,all_devices);
    
    //The devices share a context
    cl::Context context(type);
    
    cl::Program::Sources sources;
    
    std::string kernel_code=    "__kernel void bit_compression(__global uint4* input, __global uint* num_bits, __global uint* output, int length) {"
    "  int gid = get_global_id(0);"
    "if(gid >= length) return;"
    
    "  uint4 in = input[gid];"
    " int bits = num_bits[gid];"
    "  uint tmp = 0;"
    "  if (bits == 2) {"
    "    tmp |= (in.x << (32-bits)) & 3221225472u;"
    "    tmp |= (in.y << (28-bits)) &  805306368u;"
    "    tmp |= (in.z << (24-bits)) &  201326592u;"
    "    tmp |= (in.w << (20-bits)) &   50331648u;"
    "  } else if (bits == 4) {"
    "    tmp |= (in.x << (32-bits)) & 4026531840u;"
    "    tmp |= (in.y << (28-bits)) &  251658240u;"
    "    tmp |= (in.z << (24-bits)) &   15728640u;"
    "    tmp |= (in.w << (20-bits)) &     983040u;"
    "  } else if (bits == 8) {"
    "    tmp |= (in.x << (32-bits)) & 4278190080u;"
    "    tmp |= (in.y << (28-bits)) &   16711680u;"
    "    tmp |= (in.z << (24-bits)) &      65280u;"
    "    tmp |= (in.w << (20-bits)) &        255u;"
    " }"
    " output[gid] = tmp;"
    "}";
    
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    
    cl::Program program1(context,sources);
    cl::Program program2(context,sources);
    
    programBuild(devices[0], program1);
    programBuild(devices[1], program2);
    
    //define Buffer names
    //1.on Host
    int halfOfTheSize=size/2;
    cl_uint* num_bits1 = (cl_uint*) malloc(sizeof(cl_uint) * halfOfTheSize);
    cl_uint* num_bits2 = (cl_uint*) malloc(sizeof(cl_uint) * halfOfTheSize);
    cl_uint4* input1 = (cl_uint4*) malloc(sizeof(cl_uint4) * halfOfTheSize);
    cl_uint4* input2 = (cl_uint4*) malloc(sizeof(cl_uint4) * halfOfTheSize);
    for (cl_uint i = 0; i < halfOfTheSize; ++i) {
        input1[i] = (cl_uint4){15, 15, 15, 15};
        input2[i] = (cl_uint4){15, 15, 15, 15};
        num_bits1[i] = (int)pow(2, ((i % 3) + 1));
        num_bits2[i] = (int)pow(2, (((i+halfOfTheSize) % 3) + 1));
    }
    
    //2. on Device
    cl::Buffer buffer_A1(context,CL_MEM_READ_WRITE,sizeof(cl_uint4)*halfOfTheSize);
    cl::Buffer buffer_A2(context,CL_MEM_READ_WRITE,sizeof(cl_uint4)*halfOfTheSize);
    cl::Buffer buffer_B1(context,CL_MEM_READ_WRITE,sizeof(cl_uint)*halfOfTheSize);
    cl::Buffer buffer_B2(context,CL_MEM_READ_WRITE,sizeof(cl_uint)*halfOfTheSize);
    cl::Buffer buffer_C1(context,CL_MEM_READ_WRITE,sizeof(cl_uint)*halfOfTheSize);
    cl::Buffer buffer_C2(context,CL_MEM_READ_WRITE,sizeof(cl_uint)*halfOfTheSize);
    
    //create queue to which we will push commands for the device.
    cl::CommandQueue queue1(context,devices[0]);
    cl::CommandQueue queue2(context,devices[1]);
    
    bitCompressionCopyInputHostArrayToDeviceArray(input1,num_bits1,buffer_A1,buffer_B1,queue1,halfOfTheSize);
     bitCompressionCopyInputHostArrayToDeviceArray(input2,num_bits2,buffer_A2,buffer_B2,queue2,halfOfTheSize);
    
    cl::Kernel bit_compression_1=cl::Kernel(program1,"bit_compression");
    cl::Kernel bit_compression_2=cl::Kernel(program2,"bit_compression");
    
    bitCompressionRunTheKernel(bit_compression_1,program1,buffer_A1,buffer_B1,buffer_C1,queue1,halfOfTheSize);
    bitCompressionRunTheKernel(bit_compression_2,program2,buffer_A2,buffer_B2,buffer_C2,queue2,halfOfTheSize);
    
    cl_uint* output1 = (cl_uint*) malloc(sizeof(cl_uint) * halfOfTheSize);
    cl_uint* output2 = (cl_uint*) malloc(sizeof(cl_uint) * halfOfTheSize);
    
    bitCompressionCopyOutputDeviceArrayToHostArray(output1,buffer_C1,queue1,halfOfTheSize);
    bitCompressionCopyOutputDeviceArrayToHostArray(output2,buffer_C2,queue2,halfOfTheSize);
    
    bitCompressionPrintTheResultForTwoDevices(output1, output2,halfOfTheSize);
    
    free(num_bits1);
    free(num_bits2);
    free(input1);
    free(input2);
    free(output1);
    free(output2);
}


void bitCompression(int size){
    std::cout<<" Run on one default device: \n";
    bitCompressionRunAKernelOnOneDevice(CL_DEVICE_TYPE_ALL,size);
    std::cout<<" \n\n";

    std::cout<<" Run on one CPU device: \n";
    bitCompressionRunAKernelOnOneDevice(CL_DEVICE_TYPE_CPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on one GPU device: \n";
    bitCompressionRunAKernelOnOneDevice(CL_DEVICE_TYPE_GPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on two devices with 50%-50%: \n";
    bitCompressionRunAKernelOnTwoDevices(CL_DEVICE_TYPE_ALL,size);
    std::cout<<" \n\n";
}

//*****************************
//        Mat-Mul
//*****************************
void matMulRunAKernelOnOneDevice(cl_device_type type, int size){
    std::vector<cl::Platform> all_platforms;
    cl::Platform default_platform=getADefaultPlatform(all_platforms);
    
    std::vector<cl::Device> all_devices;
    cl::Device device=getOneDevice(default_platform,all_devices,type);
    
    cl::Context context(type);
    
    cl::Program::Sources sources;
    
    std::string kernel_code="__kernel void mat_mul(__global int* input1, __global int* input2, __global int* output, int num_elements, int width) {"
    "int gid = get_global_id(0);"
    "if (gid >= num_elements) return;"
    "int tx = gid % width;"
    "int ty = gid / width;"
    "int sum = 0; "
    "for (int k = 0; k < width; ++k) {"
    "sum += input1[ty * width + k] * input2[k * width + tx];"
    "}"
    "output[gid] = sum;"
    "}";
    
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    
    cl::Program program(context,sources);
    programBuild(device, program);
    
    //define Buffer names
    //1.on Host
    
    int width = (int)floor(sqrt(size));
    int sizeOfTheMatrix = width * width;
    
    int* input1 = (int*)malloc(sizeof(int) * sizeOfTheMatrix);
    int* input2 = (int*) malloc(sizeof(int) * sizeOfTheMatrix);
    
    for(int i=0; i < sizeOfTheMatrix; ++i) {
        input1[i] = i;
        input2[i] = i;
    }
    
    //2. on Device
    cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(int)*sizeOfTheMatrix);
    cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(int)*sizeOfTheMatrix);
    cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(int)*sizeOfTheMatrix);
    
    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,device);
    
    matMulCopyInputHostArrayToDeviceArray(input1,input2,buffer_A,buffer_B,queue,sizeOfTheMatrix);
    
    cl::Kernel mat_mul=cl::Kernel(program,"mat_mul");
    matMulRunTheKernel(mat_mul,program,buffer_A,buffer_B,buffer_C,queue,sizeOfTheMatrix,width);
    
    int* output = (int *)malloc(sizeof(int) * size);
    matMulCopyOutputDeviceArrayToHostArray(output,buffer_C,queue,sizeOfTheMatrix);
    
    matMulPrintTheResult(input1,input2,output,sizeOfTheMatrix,width);
    
    for(int i=0;i<size;i++){
        std::cout<<output[i]<<" ";
    }
    
    free(input1);
    free(input2);
    free(output);
}

void matMulRunAKernelOnTwoDevices(cl_device_type type, int size){
    std::vector<cl::Platform> all_platforms;
    cl::Platform default_platform=getADefaultPlatform(all_platforms);
    
    std::vector<cl::Device> all_devices;
    std::vector<cl::Device> devices=getTwoDevices(default_platform,all_devices);
    
    //The devices share a context
    cl::Context context(type);
    
    cl::Program::Sources sources;
    
    std::string kernel_code="__kernel void mat_mul(__global int* input1, __global int* input2, __global int* output, int num_elements, int width) {"
    "int gid = get_global_id(0);"
    "if (gid >= num_elements) return;"
    "int tx = gid % width;"
    "int ty = gid / width;"
    "int sum = 0; "
    "for (int k = 0; k < width; ++k) {"
    "sum += input1[ty * width + k] * input2[k * width + tx];"
    "}"
    "output[gid] = sum;"
    "}";
    
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    
    cl::Program program1(context,sources);
    cl::Program program2(context,sources);
    
    programBuild(devices[0], program1);
    programBuild(devices[1], program2);
    
    //define Buffer names
    //1.on Host
    
    int width = (int)floor(sqrt(size));
    int sizeOfTheMatrix = width * width;
    
    int newWidth = width;
    int halfSizeOfTheMatrix = size/2;
    
    int* input1device1 = (int*)malloc(sizeof(int) * halfSizeOfTheMatrix);
    int* input1device2 = (int*)malloc(sizeof(int) * halfSizeOfTheMatrix);
    int* input2device1 = (int*) malloc(sizeof(int) * halfSizeOfTheMatrix);
    int* input2device2 = (int*) malloc(sizeof(int) * halfSizeOfTheMatrix);
    
    for(int i=0; i < halfSizeOfTheMatrix; ++i) {
        input1device1[i] = i;
        input1device2[i] = i+halfSizeOfTheMatrix;
        input2device1[i] = i;
        input2device2[i] = i+halfSizeOfTheMatrix;
    }
    
    //2. on Device
    cl::Buffer buffer_A1(context,CL_MEM_READ_WRITE,sizeof(int)*halfSizeOfTheMatrix);
    cl::Buffer buffer_A2(context,CL_MEM_READ_WRITE,sizeof(int)*halfSizeOfTheMatrix);
    cl::Buffer buffer_B1(context,CL_MEM_READ_WRITE,sizeof(int)*halfSizeOfTheMatrix);
    cl::Buffer buffer_B2(context,CL_MEM_READ_WRITE,sizeof(int)*halfSizeOfTheMatrix);
    cl::Buffer buffer_C1(context,CL_MEM_READ_WRITE,sizeof(int)*halfSizeOfTheMatrix);
    cl::Buffer buffer_C2(context,CL_MEM_READ_WRITE,sizeof(int)*halfSizeOfTheMatrix);
    
    //create queue to which we will push commands for the device.
    cl::CommandQueue queue1(context,devices[0]);
    cl::CommandQueue queue2(context,devices[1]);
    
    matMulCopyInputHostArrayToDeviceArray(input1device1,input2device1,buffer_A1,buffer_B1,queue1,halfSizeOfTheMatrix);
    matMulCopyInputHostArrayToDeviceArray(input1device2,input2device2,buffer_A2,buffer_B2,queue2,halfSizeOfTheMatrix);
    
    cl::Kernel mat_mul_1=cl::Kernel(program1,"mat_mul");
    cl::Kernel mat_mul_2=cl::Kernel(program2,"mat_mul");
    
    matMulRunTheKernel(mat_mul_1,program1,buffer_A1,buffer_B1,buffer_C1,queue1,halfSizeOfTheMatrix,newWidth);
    matMulRunTheKernel(mat_mul_2,program2,buffer_A2,buffer_B2,buffer_C2,queue2,halfSizeOfTheMatrix,newWidth);
    
    int* output1 = (int *)malloc(sizeof(int) * halfSizeOfTheMatrix);
    int* output2 = (int *)malloc(sizeof(int) * halfSizeOfTheMatrix);
    
    matMulCopyOutputDeviceArrayToHostArray(output1,buffer_C1,queue1,halfSizeOfTheMatrix);
    matMulCopyOutputDeviceArrayToHostArray(output2,buffer_C2,queue2,halfSizeOfTheMatrix);
    
    matMulPrintTheResultOnTwoDevices(input1device1,input1device2, input2device1, input2device2,output1,output2,halfSizeOfTheMatrix,newWidth);
    
    free(input1device1);
    free(input1device2);
    free(input2device1);
    free(input2device2);
    free(output1);
    free(output2);
}

void matMul(int size){
    std::cout<<" Run on one default device: \n";
    matMulRunAKernelOnOneDevice(CL_DEVICE_TYPE_ALL,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on one CPU device: \n";
    matMulRunAKernelOnOneDevice(CL_DEVICE_TYPE_CPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on one GPU device: \n";
    matMulRunAKernelOnOneDevice(CL_DEVICE_TYPE_GPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on two devices with 50%-50%: \n";
    matMulRunAKernelOnTwoDevices(CL_DEVICE_TYPE_ALL,size);
    std::cout<<" \n\n";

}

//*****************************
//        Lin-Reg
//*****************************
void linRegRunAKernelOnOneDevice(cl_device_type type, int size){
    std::vector<cl::Platform> all_platforms;
    cl::Platform default_platform=getADefaultPlatform(all_platforms);
    
    std::vector<cl::Device> all_devices;
    cl::Device device=getOneDevice(default_platform,all_devices,type);
    
    cl::Context context(type);
    
    cl::Program::Sources sources;
    
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
    
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    
    cl::Program program(context,sources);
    programBuild(device, program);
    
    //define Buffer names
    //1.on Host

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
    
    //2. on Device
    cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(float)*size);
    cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(float)*size);
    cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(float)*size);
    cl::Buffer buffer_D(context,CL_MEM_READ_WRITE,sizeof(float)*size);
    cl::Buffer buffer_E(context,CL_MEM_READ_WRITE,sizeof(float)*size);
    
    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,device);
    
    linRegCopyInputHostArrayToDeviceArray(input1,input2,alpha,beta,buffer_A,buffer_B,buffer_C,buffer_D,queue,size);
    
    cl::Kernel lin_reg=cl::Kernel(program,"lin_reg");
    linRegRunTheKernel(lin_reg,program,buffer_A,buffer_B,buffer_C,buffer_D,buffer_E,queue,size);
    
    float* output = (float*) malloc(sizeof(float) * size);
    linRegCopyOutputDeviceArrayToHostArray(output,buffer_E,queue,size);
    
    linRegPrintTheResult(input1,input2,alpha,beta,output,size);
    
    free(input1);
    free(input2);
    free(alpha);
    free(beta);
    free(output);
}

void linRegRunAKernelOnTwoDevices(cl_device_type type, int size){
    std::vector<cl::Platform> all_platforms;
    cl::Platform default_platform=getADefaultPlatform(all_platforms);
    
    std::vector<cl::Device> all_devices;
    std::vector<cl::Device> devices=getTwoDevices(default_platform,all_devices);
    
    //The devices share a context
    
    cl::Context context(type);
    
    cl::Program::Sources sources;
    
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
    
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    
    cl::Program program1(context,sources);
    cl::Program program2(context,sources);
    
    programBuild(devices[0], program1);
    programBuild(devices[1], program2);
    
    //define Buffer names
    //1.on Host
    
    int halfOfTheSize=size/2;
    
    float* input1device1 = (float*) malloc(sizeof(float) * halfOfTheSize);
    float* input1device2 = (float*) malloc(sizeof(float) * halfOfTheSize);
    float* input2device1 = (float*) malloc(sizeof(float) * halfOfTheSize);
    float* input2device2 = (float*) malloc(sizeof(float) * halfOfTheSize);
    float* alpha1  = (float*) malloc(sizeof(float) * halfOfTheSize);
    float* alpha2  = (float*) malloc(sizeof(float) * halfOfTheSize);
    float* beta1   = (float*) malloc(sizeof(float) * halfOfTheSize);
    float* beta2   = (float*) malloc(sizeof(float) * halfOfTheSize);
    
    fill_random_float(input2device1, halfOfTheSize, 1, -1.0f, 1.0f);
    fill_random_float(input2device2, halfOfTheSize, 1, -1.0f, 1.0f);
    qsort(input2device1, halfOfTheSize, sizeof(float), float_compare);
    qsort(input2device2, halfOfTheSize, sizeof(float), float_compare);
    float step = 2.0f / halfOfTheSize;
    for(int i=0; i < halfOfTheSize; i++){
        input1device1[i] = -1.0f + i * step;
        input1device2[i] = -1.0f + (i+halfOfTheSize) * step;
    }
    
    fill_random_float(alpha1, halfOfTheSize, 1, -1.0f, 1.0f);
    fill_random_float(alpha2, halfOfTheSize, 1, -1.0f, 1.0f);
    fill_random_float(beta1, halfOfTheSize, 1, -1.0f, 1.0f);
    fill_random_float(beta2, halfOfTheSize, 1, -1.0f, 1.0f);
    
    //2. on Device
    cl::Buffer buffer_A1(context,CL_MEM_READ_WRITE,sizeof(float)*halfOfTheSize);
    cl::Buffer buffer_A2(context,CL_MEM_READ_WRITE,sizeof(float)*halfOfTheSize);
    cl::Buffer buffer_B1(context,CL_MEM_READ_WRITE,sizeof(float)*halfOfTheSize);
    cl::Buffer buffer_B2(context,CL_MEM_READ_WRITE,sizeof(float)*halfOfTheSize);
    cl::Buffer buffer_C1(context,CL_MEM_READ_WRITE,sizeof(float)*halfOfTheSize);
    cl::Buffer buffer_C2(context,CL_MEM_READ_WRITE,sizeof(float)*halfOfTheSize);
    cl::Buffer buffer_D1(context,CL_MEM_READ_WRITE,sizeof(float)*halfOfTheSize);
    cl::Buffer buffer_D2(context,CL_MEM_READ_WRITE,sizeof(float)*halfOfTheSize);
    cl::Buffer buffer_E1(context,CL_MEM_READ_WRITE,sizeof(float)*halfOfTheSize);
    cl::Buffer buffer_E2(context,CL_MEM_READ_WRITE,sizeof(float)*halfOfTheSize);
    
    //create queue to which we will push commands for the device.
    cl::CommandQueue queue1(context,devices[0]);
    cl::CommandQueue queue2(context,devices[1]);
    
    linRegCopyInputHostArrayToDeviceArray(input1device1,input2device1,alpha1,beta1,buffer_A1,buffer_B1,buffer_C1,buffer_D1,queue1,halfOfTheSize);
    linRegCopyInputHostArrayToDeviceArray(input1device2,input2device2,alpha2,beta2,buffer_A2,buffer_B2,buffer_C2,buffer_D2,queue2,halfOfTheSize);
    
    cl::Kernel lin_reg_1=cl::Kernel(program1,"lin_reg");
    cl::Kernel lin_reg_2=cl::Kernel(program2,"lin_reg");
    linRegRunTheKernel(lin_reg_1,program1,buffer_A1,buffer_B1,buffer_C1,buffer_D1,buffer_E1,queue1,halfOfTheSize);
    linRegRunTheKernel(lin_reg_2,program2,buffer_A2,buffer_B2,buffer_C2,buffer_D2,buffer_E2,queue2,halfOfTheSize);
    
    float* output1 = (float*) malloc(sizeof(float) * halfOfTheSize);
    float* output2 = (float*) malloc(sizeof(float) * halfOfTheSize);
    linRegCopyOutputDeviceArrayToHostArray(output1,buffer_E1,queue1,halfOfTheSize);
    linRegCopyOutputDeviceArrayToHostArray(output2,buffer_E2,queue2,halfOfTheSize);
    
    linRegPrintTheResultTwoDevices(input1device1,input1device2,input2device1,input2device2,alpha1,alpha2,beta1,beta2,output1,output2,halfOfTheSize);
    
    free(input1device1);
    free(input1device2);
    free(input2device1);
    free(input2device2);
    free(alpha1);
    free(alpha2);
    free(beta1);
    free(beta2);
    free(output1);
    free(output2);
}


void linReg(int size){
    std::cout<<" Run on one default device: \n";
    linRegRunAKernelOnOneDevice(CL_DEVICE_TYPE_ALL,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on one CPU device: \n";
    linRegRunAKernelOnOneDevice(CL_DEVICE_TYPE_CPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on one GPU device: \n";
    linRegRunAKernelOnOneDevice(CL_DEVICE_TYPE_GPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on two devices with 50%-50%: \n";
    linRegRunAKernelOnTwoDevices(CL_DEVICE_TYPE_ALL,size);
    std::cout<<" \n\n";
}

//*****************************
//        Sobel-Filter
//*****************************

/*
void sobelFilterRunAKernelOnOneDevice(int size){
    std::vector<cl::Platform> all_platforms;
    cl::Platform default_platform=getADefaultPlatform(all_platforms);
    
    std::vector<cl::Device> all_devices;
    cl_device_type type=CL_DEVICE_TYPE_ALL;
    cl::Device default_device=getOneDevice(default_platform,all_devices,type);
    
    cl::Context context({default_device});
    
    cl::Program::Sources sources;
    
    std::string kernel_code="__kernel void sobel_filter(__global uchar4* inputImage, __global uchar4* outputImage, int num_elements, uint width) {"
        "int gid = get_global_id(0);"
        "if (gid >= num_elements) return;"
        
        "int tx = gid % width;"
        "int ty = gid / width;"
        
        "if( tx >= 1 && tx < (width-1) && ty >= 1 && ty < num_elements/width - 1) {"
            "float4 i00 = convert_float4(inputImage[gid - 1 - width]);"
            "float4 i10 = convert_float4(inputImage[gid - width]);"
            "float4 i20 = convert_float4(inputImage[gid + 1 - width]);"
            "float4 i01 = convert_float4(inputImage[gid - 1]);"
            "float4 i11 = convert_float4(inputImage[gid]);"
            "float4 i21 = convert_float4(inputImage[gid + 1]);"
            "float4 i02 = convert_float4(inputImage[gid - 1 + width]);"
            "float4 i12 = convert_float4(inputImage[gid + width]);"
            "float4 i22 = convert_float4(inputImage[gid + 1 + width]);"
            "float4 two = (float4)2;"
            
            "float4 Gx =   i00 + two * i10 + i20 - i02  - two * i12 - i22;"
            "float4 Gy =   i00 - i20 + two * i01 - two * i21 + i02  -  i22;"
            
            "outputImage[gid] = convert_uchar4(hypot(Gx, Gy)/two);"
        "}"
    "}";
    
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    
    
    cl::Program program(context,sources);
    programBuild(default_device,program);
    
    //define Buffer names
    //1.on Host
    
    cl_uchar4* bmpPixel;
    //BITMAPINFO *bmpInfo;
    
    cl_uint width=size;
    cl_uint height=size;
    
    // read the input image
    //bmpPixel = icl_loadbmp_pixel_uchar4(INSIEME_TEST_BMP, &bmpInfo, args->size, &width, &height);
    int imageSize = width * height;
    
    // allocate memory for input & output image data
    cl_uchar4* inputImageData  = (cl_uchar4*)malloc(sizeof(cl_uchar4) * size);
    memcpy(inputImageData, bmpPixel, sizeof(cl_uchar4) * size);
    
    printf("width %d, height %d\n", width, height);
    
    //2. on Device
    cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(cl_uchar4)*size);
    cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(cl_uchar4)*size);
    
    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,default_device);
    
    linRegCopyInputHostArrayToDeviceArray(input1,input2,alpha,beta,buffer_A,buffer_B,buffer_C,buffer_D,queue,size);
    
    cl::Kernel lin_reg=cl::Kernel(program,"lin_reg");
    linRegRunTheKernel(lin_reg,program,buffer_A,buffer_B,buffer_C,buffer_D,buffer_E,queue,size);
    
    cl_uchar4* outputImageData = (cl_uchar4*)malloc(sizeof(cl_uchar4) * size);
    memset(outputImageData, 0, sizeof(cl_uchar4) * size);
    
    linRegCopyOutputDeviceArrayToHostArray(output,buffer_E,queue,size);
    
    linRegPrintTheResult(input1,input2,alpha,beta,output,size);
    
    free(input1);
    free(input2);
    free(alpha);
    free(beta);
    free(output);
}
 */

void sobelFilterRunAKernelOnOneCPUORGPUDevice(cl_device_type type, int size){
    std::vector<cl::Platform> all_platforms;
    cl::Platform default_platform=getADefaultPlatform(all_platforms);
    
    std::vector<cl::Device> all_devices;
    cl::Device device=getOneDevice(default_platform,all_devices,type);
    
    cl::Context context(type);
    
    cl::Program::Sources sources;
    
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
    
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    
    cl::Program program(context,sources);
    programBuild(device, program);
    
    //define Buffer names
    //1.on Host
    
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
    
    //2. on Device
    cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(float)*size);
    cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(float)*size);
    cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(float)*size);
    cl::Buffer buffer_D(context,CL_MEM_READ_WRITE,sizeof(float)*size);
    cl::Buffer buffer_E(context,CL_MEM_READ_WRITE,sizeof(float)*size);
    
    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,device);
    
    linRegCopyInputHostArrayToDeviceArray(input1,input2,alpha,beta,buffer_A,buffer_B,buffer_C,buffer_D,queue,size);
    
    cl::Kernel lin_reg=cl::Kernel(program,"lin_reg");
    linRegRunTheKernel(lin_reg,program,buffer_A,buffer_B,buffer_C,buffer_D,buffer_E,queue,size);
    
    float* output = (float*) malloc(sizeof(float) * size);
    linRegCopyOutputDeviceArrayToHostArray(output,buffer_E,queue,size);
    
    linRegPrintTheResult(input1,input2,alpha,beta,output,size);
    
    free(input1);
    free(input2);
    free(alpha);
    free(beta);
    free(output);
}

/*
void sobelFilter(int size){
    std::cout<<" Run on one default device: \n";
    sobelFilterRunAKernelOnOneDeviceCL_DEVICE_TYPE_ALL,size);
    std::cout<<" \n\n";
 
    std::cout<<" Run on one CPU device: \n";
    sobelFilterRunAKernelOnOneDevice(CL_DEVICE_TYPE_CPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on one GPU device: \n";
    sobelFilterRunAKernelOnOneDevice(CL_DEVICE_TYPE_GPU,size);
    std::cout<<" \n\n";
 
  }

*/

//*****************************
//        Syr2k
//*****************************

//TODO: why does it work only till 12??
void syr2kRunAKernelOnOneDevice(cl_device_type type, int size){
    std::vector<cl::Platform> all_platforms;
    cl::Platform default_platform=getADefaultPlatform(all_platforms);
    
    std::vector<cl::Device> all_devices;
    cl::Device device=getOneDevice(default_platform,all_devices,type);
    
    cl::Context context(type);
    
    cl::Program::Sources sources;
    
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
    
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    
    cl::Program program(context,sources);
    programBuild(device, program);
    
    //define Buffer names
    //1.on Host
    
    int width = (int)floor(sqrt(size));
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
    
    //2. on Device
    cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(float)*newSize);
    cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(float)*newSize);
    cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(float)*newSize);
    
    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,device);
    
    syr2kCopyInputHostArrayToDeviceArray(input_a,input_b,output,buffer_A,buffer_B,buffer_C,queue,newSize);
    
    cl::Kernel syr2k1=cl::Kernel(program,"syr2k");
    syr2kRunTheKernel(syr2k1,program,buffer_A,buffer_B,buffer_C,ALPHA,BETA,M,N,queue,newSize);
    
    syr2kCopyOutputDeviceArrayToHostArray(output,buffer_C,queue,newSize);
    
    syr2kPrintTheResult(input_a,input_b,ALPHA,BETA,M,N,output,newSize);
    
    free(input_a);
    free(input_b);
    free(output);
}


void syr2k(int size){
    std::cout<<" Run on one default device: \n";
    syr2kRunAKernelOnOneDevice(CL_DEVICE_TYPE_ALL,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on one CPU device: \n";
    syr2kRunAKernelOnOneDevice(CL_DEVICE_TYPE_CPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on one GPU device: \n";
    syr2kRunAKernelOnOneDevice(CL_DEVICE_TYPE_GPU,size);
    std::cout<<" \n\n";
}

//*****************************
//        Convolution
//*****************************
void convolutionRunAKernelOnOneDevice(cl_device_type type, int size){
    std::vector<cl::Platform> all_platforms;
    cl::Platform default_platform=getADefaultPlatform(all_platforms);
    
    std::vector<cl::Device> all_devices;
    cl::Device device=getOneDevice(default_platform,all_devices,type);
    
    cl::Context context(type);
    
    cl::Program::Sources sources;
    
    std::string kernel_code="__kernel void convolution(__global int* input, __constant int* mask, __global int* output, int num_elements, int width, int mask_width) {"
    "int gid = get_global_id(0);"
    "if (gid >= num_elements) return;"
    "int tx = gid % width;"
    "int ty = gid / width;"
    "int offset = mask_width/2;"
    "if (tx < offset || ty < offset || tx >= (width-offset) || ty >= (width-offset)) {"
    "output[gid] = 0;"
    "return;"
    "}"
    "int sum = 0;"
    
    "int tmpx = tx - offset;"
    "int tmpy = ty - offset;"
    "for (int r = 0; r < mask_width; ++r) {"
    "for (int c = 0; c < mask_width; ++c) {"
    "sum += mask[r * mask_width + c] * input[(tmpy + r ) * width + tmpx + c];"
    "}"
    "}"
    "output[gid] = sum;"
    "}";
    
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    
    cl::Program program(context,sources);
    programBuild(device, program);
    
    //define Buffer names
    //1.on Host
    
    int width = (int)floor(sqrt(size));
    int newSize = width * width;
    
    int mask_width = 22;
    int mask_size = mask_width * mask_width;
    
    int* input  = (int*)malloc(sizeof(int) * newSize);
    int* mask   = (int*) malloc(sizeof(int) * mask_size);
    int* output = (int*)malloc(sizeof(int) * newSize);
    
    for(int i=0; i < mask_size; ++i){
        mask[i] = 1;
    }
    mask[mask_size/2] = 0;
    
    for(int i=0; i < newSize; ++i) {
        input[i] = 1;//rand() % 10;
    }
    
    //2. on Device
    cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(int)*newSize);
    cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(int)*newSize);
    cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(int)*newSize);
    
    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,device);
    
    convolutionCopyInputHostArrayToDeviceArray(input,mask,output,buffer_A,buffer_B,buffer_C,queue,newSize,mask_size);
    
    cl::Kernel convolution1=cl::Kernel(program,"convolution");
    convolutionRunTheKernel(convolution1,program,buffer_A,buffer_B,buffer_C,width,mask_width,queue,newSize);
    
    convolutionCopyOutputDeviceArrayToHostArray(output,buffer_C,queue,newSize);
    
    convolutionPrintTheResult(input,mask,output,width,mask_width,newSize);
    
    free(input);
    free(mask);
    free(output);
}


void convolution(int size){
    
    std::cout<<" Run on one default device: \n";
    convolutionRunAKernelOnOneDevice(CL_DEVICE_TYPE_ALL,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on one CPU device: \n";
    convolutionRunAKernelOnOneDevice(CL_DEVICE_TYPE_CPU,size);
    std::cout<<" \n\n";
    
    std::cout<<" Run on one GPU device: \n";
    convolutionRunAKernelOnOneDevice(CL_DEVICE_TYPE_GPU,size);
    std::cout<<" \n\n";
    
}

//*****************************
// Main Functions
//*****************************

void sixKernels(int size, int programmNumber){
    //std::cout<<size<<" ";
    if (programmNumber==1)
       vecAdd(size);
    else if (programmNumber==2)
        bitCompression(size);
    else if (programmNumber==3)
       matMul(size);
    else if (programmNumber==4)
       linReg(size);
    else if (programmNumber==5)
       syr2k(size);
    else
      convolution(size);
    
    //TODO!!
    //sobelFilter(size);
}



int main(int argc, char* argv[]){
    
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
