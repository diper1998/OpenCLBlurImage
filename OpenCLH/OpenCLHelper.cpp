#include "OpenCLHelper.h"

OpenCLHelper::OpenCLHelper() {
  SetDevices();
  SetContext();
  SetKernelCode(ReadKernelCode("kernelCode.cl"));
  SetProgram();
  SetCommandQueue();
}

OpenCLHelper::~OpenCLHelper() {}

void OpenCLHelper::SetDevices() {
  cl::Platform::get(&allPlatforms);

  for (int i = 0; i < allPlatforms.size(); i++) {
    allPlatforms[i].getDevices(CL_DEVICE_TYPE_GPU, &allDevicesGPU);
    allPlatforms[i].getDevices(CL_DEVICE_TYPE_CPU, &allDevicesCPU);

    if (allDevicesGPU.size() != 0 && allDevicesCPU.size() != 0) {
      if (i < allDevicesCPU.size() && i < allDevicesGPU.size()) {
        defaultDeviceGPU = allDevicesGPU[i];
        defaultDeviceCPU = allDevicesCPU[i];
        defaultPlatformGPU = allPlatforms[i];
        defaultPlatformCPU = allPlatforms[i];
        break;
      }
    } 
  }
}

void OpenCLHelper::GetDevicesInfo() {
  std::string myInfo;

  std::cout << std::endl << "CPU: " << std::endl;
  myInfo = defaultPlatformCPU.getInfo<CL_PLATFORM_NAME>();
  std::cout << std::endl << myInfo << std::endl;
  myInfo = defaultDeviceCPU.getInfo<CL_DEVICE_NAME>();
  std::cout << std::endl << myInfo << std::endl;

  std::cout << std::endl << "GPU: " << std::endl;
  myInfo = defaultPlatformGPU.getInfo<CL_PLATFORM_NAME>();
  std::cout << std::endl << myInfo << std::endl;
  myInfo = defaultDeviceGPU.getInfo<CL_DEVICE_NAME>();
  std::cout << std::endl << myInfo << std::endl;
}

void OpenCLHelper::SetContext() {
  cl::Context myContext({defaultDeviceCPU, defaultDeviceGPU});
  clContext = myContext;
}

std::string OpenCLHelper::ReadKernelCode(std::string myFileName) {
  std::ifstream myFile(myFileName);
  if (!myFile.is_open()) {
    std::cout << "ERROR:";
    std::cout << "ReadKernelCode():";
    std::cout << myFileName << " isn't open.";
    exit(1);
  }
  myFile.seekg(0, std::ios::end);
  size_t size = myFile.tellg();
  std::string myKernelCode(size, ' ');
  myFile.seekg(0);
  myFile.read(&myKernelCode[0], size);
  myFile.close();

  return myKernelCode;
}

void OpenCLHelper::SetKernelCode(std::string myKernelCode) {
  clKernelCode = myKernelCode;
  clSources.push_back({clKernelCode.c_str(), clKernelCode.length()});
}

void OpenCLHelper::GetKernelCodeInfo() {
  std::cout << clKernelCode << std::endl;
}

void OpenCLHelper::SetKernel(std::string myFunctionName) {
  cl::Kernel kernelFunction(clProgram, myFunctionName.c_str());
  clKernel = kernelFunction;
}

void OpenCLHelper::SetKernelFirst(std::string myFunctionName) {
  cl::Kernel kernelFunction(clProgram, myFunctionName.c_str());
  clKernelFirst = kernelFunction;
}

void OpenCLHelper::SetKernelSecond(std::string myFunctionName) {
  cl::Kernel kernelFunction(clProgram, myFunctionName.c_str());
  clKernelSecond = kernelFunction;
}

void OpenCLHelper::SetKernelThird(std::string myFunctionName) {
  cl::Kernel kernelFunction(clProgram, myFunctionName.c_str());
  clKernelThird = kernelFunction;
}

void OpenCLHelper::SetProgram() {
  cl::Program myProgram(clContext, clSources);
  clProgram = myProgram;

  if (clProgram.build({defaultDeviceCPU, defaultDeviceGPU}) != CL_SUCCESS) {
    std::cout << "ERROR SetProgram(): "
              << clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDeviceCPU)
              << std::endl
              << clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDeviceGPU)
              << std::endl;
    exit(1);
  }
}

void OpenCLHelper::GetProgramInfo() {
  std::cout << "CPU:" << std::endl
            << clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDeviceCPU)
            << std::endl
            << "GPU:" << std::endl
            << clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDeviceGPU)
            << std::endl;
}

void OpenCLHelper::SetCommandQueue() {
  cl::CommandQueue myCommandQueueCPU(clContext, defaultDeviceCPU);
  cl::CommandQueue myCommandQueueGPU(clContext, defaultDeviceGPU);

  commandQueueCPU = myCommandQueueCPU;
  commandQueueGPU = myCommandQueueGPU;
}

void OpenCLHelper::SetBuffers(cl_int myCountArguments, cl_int* myArraySizes) {
  numberArguments = myCountArguments;
  arraySizes = myArraySizes;

  clBuffers = new cl::Buffer[numberArguments];

  for (int i = 0; i < numberArguments; i++) {
    cl::Buffer myBuffer(clContext, CL_MEM_READ_WRITE, arraySizes[i]);
    clBuffers[i] = myBuffer;
  }
}

void OpenCLHelper::StartTime() {
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&startTime);
}

void OpenCLHelper::FinishTime() {
  QueryPerformanceCounter(&endTime);
  workTime = (endTime.QuadPart - startTime.QuadPart) / (double)freq.QuadPart;
}

void OpenCLHelper::GetTimeInfo() {
  std::cout << std::endl << "Time: " << workTime << std::endl;
}
