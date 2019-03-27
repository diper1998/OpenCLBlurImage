#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 
#include <CL/cl.hpp> 
#include <iostream>
#include <string>
#include <fstream>
#include <Windows.h>
#include "FreeImage.h"
#pragma comment(lib, "FreeImage.lib")

class OpenCLHelper
{

protected:

	// Platforms
	std::vector<cl::Platform> allPlatforms;	  
											  
	std::vector<cl::Platform> allPlatformCPU; 
	std::vector<cl::Platform> allPlatformGPU; 
											  
	cl::Platform defaultPlatformCPU;		  
	cl::Platform defaultPlatformGPU;		  
										  
	// Devices
	std::vector<cl::Device> allDevices;

	std::vector<cl::Device> allDevicesCPU;
	std::vector<cl::Device> allDevicesGPU;

	cl::Device defaultDeviceCPU;
	cl::Device defaultDeviceGPU;

	cl::Context clContext;
	std::string clKernelCode;
    cl::Kernel clKernel;
    cl::Kernel clKernelFirst;
    cl::Kernel clKernelSecond;
    cl::Kernel clKernelThird;
	cl::Program::Sources clSources;
	cl::Program clProgram;
	
	cl::Buffer *clBuffers;
	cl_int numberArguments;
	cl_int* arraySizes;

	cl::CommandQueue commandQueueCPU;
	cl::CommandQueue commandQueueGPU;
	
	LARGE_INTEGER startTime, endTime, freq;
	double workTime;

    
    cl::NDRange globalSizeCPU;
    cl::NDRange globalSizeGPU;

	

public:

	OpenCLHelper();
	~OpenCLHelper();

	void SetDevices();
	void GetDevicesInfo();

	void SetContext();

	std::string ReadKernelCode(std::string myFile);

	void SetKernelCode(std::string myKernelCode);
	void GetKernelCodeInfo();

	void SetKernel(std::string functionName);
    void SetKernelFirst(std::string functionName);
    void SetKernelSecond(std::string functionName);
    void SetKernelThird(std::string functionName);

	void SetProgram();
	void GetProgramInfo();

	void SetCommandQueue();

	void SetBuffers( cl_int myCountArguments, cl_int* myArraySizes);

    void SetGlobalSizeCPU(cl_int* mySize);
    void SetGlobalSizeGPU(cl_int* mySize);

	void StartTime();
	void FinishTime();
	void GetTimeInfo();
};
