#pragma once
#include "OpenCLHelper.h"
#include "FreeImage.h"


class OpenCLImage :
	public OpenCLHelper
{
protected:

	struct Image {
		float* R;
		float* G;
		float* B;
		unsigned height;
		unsigned width;
		unsigned bitsPerPixel;
		Image() : R(NULL), G(NULL), B(NULL), height(0), width(0) {}
	};

	Image clImageIn;
	Image clImageOut;
	std::string clImageInName;
	std::string clImageOutName;
	
    int blurStep = 32;

public:

	OpenCLImage(std::string myImageInName, std::string myImageOutName): OpenCLHelper() {
		clImageInName = myImageInName;
		clImageOutName = myImageOutName;
		SetImageIn(myImageInName);
	};

	~OpenCLImage() {
		
	};

	Image ReadImage(std::string myImageName) {
		Image myImage;

		FIBITMAP* bitmap = FreeImage_Load(FIF_JPEG, myImageName.c_str());
		if (!bitmap) return myImage;

		FREE_IMAGE_TYPE imageType = FreeImage_GetImageType(bitmap);
		FREE_IMAGE_COLOR_TYPE colorType = FreeImage_GetColorType(bitmap);

		unsigned height = FreeImage_GetHeight(bitmap);
		unsigned width = FreeImage_GetWidth(bitmap);
		unsigned pitch = FreeImage_GetPitch(bitmap);
		unsigned bitsPerPixel = FreeImage_GetBPP(bitmap);

		if (colorType == FIC_RGB || colorType == FIC_RGBALPHA)
		{
			if (imageType == FIT_BITMAP)
			{
				myImage.R = new float[height*width];
				myImage.G = new float[height*width];
				myImage.B = new float[height*width];
				myImage.height = height;
				myImage.width = width;
				myImage.bitsPerPixel = bitsPerPixel;

				BYTE* bits = FreeImage_GetBits(bitmap);
				for (unsigned y = 0; y < height; ++y)
				{
					BYTE* pixel = (BYTE*)bits;
					for (unsigned x = 0; x < width; ++x)
					{
						myImage.R[y*width + x] = (float)pixel[FI_RGBA_RED];
						myImage.G[y*width + x] = (float)pixel[FI_RGBA_GREEN];
						myImage.B[y*width + x] = (float)pixel[FI_RGBA_BLUE];
						pixel += bitsPerPixel / 8;
					}
					bits += pitch;
				}
			}
		}

		FreeImage_Unload(bitmap);

		return myImage;
	}
	void SetImageIn(std::string myImageName) {
		clImageIn = ReadImage(myImageName);
	}
	void SetImageOut(std::string myImageName) {
		clImageOut = ReadImage(myImageName);
	}

	Image GetImageIn() {
		return clImageIn;
	}
	Image GetImageOut() {
		return clImageOut;
	}
	
	void GetImageInInfo() {
		std::cout << clImageIn.height << std::endl
			<< clImageIn.width << std::endl
			<< clImageIn.bitsPerPixel << std::endl;
	}
	void GetImageOutInfo() {
		std::cout << clImageIn.height << std::endl
			<< clImageIn.width << std::endl
			<< clImageIn.bitsPerPixel << std::endl;
	}

	bool WriteImage(Image myImage, std::string myImageName) {
		FIBITMAP* bitmap = FreeImage_Allocate(myImage.width, myImage.height, myImage.bitsPerPixel);
		if (!bitmap) return false;

		unsigned height = FreeImage_GetHeight(bitmap);
		unsigned width = FreeImage_GetWidth(bitmap);
		unsigned pitch = FreeImage_GetPitch(bitmap);
		unsigned bitsPerPixel = FreeImage_GetBPP(bitmap);

		BYTE* bits = FreeImage_GetBits(bitmap);
		for (unsigned y = 0; y < height; ++y)
		{
			BYTE* pixel = (BYTE*)bits;
			for (unsigned x = 0; x < width; ++x)
			{
				pixel[FI_RGBA_RED] = (BYTE)myImage.R[y*width + x];
				pixel[FI_RGBA_GREEN] = (BYTE)myImage.G[y*width + x];
				pixel[FI_RGBA_BLUE] = (BYTE)myImage.B[y*width + x];
				pixel += bitsPerPixel / 8;
			}
			bits += pitch;
		}

		BOOL isSaved = FreeImage_Save(FIF_JPEG, bitmap, myImageName.c_str());
		FreeImage_Unload(bitmap);

		return (isSaved == TRUE);
	}

    void BlurImageCPU() {
      clImageOut = clImageIn;

      cl_int myStatus = CL_SUCCESS;
      

      numberArguments = 6;
      arraySizes = new int[numberArguments];
      arraySizes[0] = clImageIn.height * clImageIn.width * sizeof(float);
      arraySizes[1] = clImageIn.height * clImageIn.width * sizeof(float);
      
      arraySizes[2] = clImageIn.height * clImageIn.width * sizeof(float);
      arraySizes[3] = clImageIn.height * clImageIn.width * sizeof(float);
      
      arraySizes[4] = clImageIn.height * clImageIn.width * sizeof(float);
      arraySizes[5] = clImageIn.height * clImageIn.width * sizeof(float);
      
      
      SetBuffers(numberArguments, arraySizes);
      SetKernel("blur");
     
      

      myStatus = clKernel.setArg(2, clImageIn.width );
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 2 " << myStatus << std::endl;
      }
      myStatus = clKernel.setArg(3, clImageIn.height );
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 3 " << myStatus << std::endl;
      }
      myStatus = clKernel.setArg(4, blurStep);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 4 " << myStatus << std::endl;
      }

     
      
      cl::NDRange globalSize(clImageIn.width, clImageIn.height);


      myStatus = commandQueueCPU.enqueueWriteBuffer(
          clBuffers[2], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(float), clImageIn.R);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: writing " << myStatus << std::endl;
      }

      myStatus = commandQueueCPU.enqueueWriteBuffer(
          clBuffers[0], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(float), clImageIn.G);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: writing " << myStatus << std::endl;
      }
      
      myStatus = commandQueueCPU.enqueueWriteBuffer(
          clBuffers[4], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(float), clImageIn.B);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: writing " << myStatus << std::endl;
      }


      commandQueueCPU.finish();

      StartTime();
      
      // R

      myStatus = clKernel.setArg(1, clBuffers[3]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 1 " << myStatus << std::endl;
      }

      myStatus = clKernel.setArg(0, clBuffers[2]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 0 " << myStatus << std::endl;
      }

      myStatus = commandQueueCPU.enqueueNDRangeKernel(
          clKernel, cl::NDRange(0, 0), globalSize);

      //G
      myStatus = clKernel.setArg(1, clBuffers[1]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 1 " << myStatus << std::endl;
      }
   
      myStatus = clKernel.setArg(0, clBuffers[0]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 0 " << myStatus << std::endl;
      }
      myStatus = commandQueueCPU.enqueueNDRangeKernel(
          clKernel, cl::NDRange(0, 0), globalSize);

    
      // B
      myStatus = clKernel.setArg(1, clBuffers[5]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 2 " << myStatus << std::endl;
      }
      
      myStatus = clKernel.setArg(0, clBuffers[4]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 0 " << myStatus << std::endl;
      }

      myStatus = commandQueueCPU.enqueueNDRangeKernel(
          clKernel, cl::NDRange(0, 0), globalSize);



      //Read
      commandQueueCPU.finish();
      FinishTime();

      myStatus = commandQueueCPU.enqueueReadBuffer(
          clBuffers[3], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(int), clImageOut.R);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: reading " << myStatus << std::endl;
      }

      myStatus = commandQueueCPU.enqueueReadBuffer(
          clBuffers[1], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(int), clImageOut.G);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: reading " << myStatus << std::endl;
      }

      myStatus = commandQueueCPU.enqueueReadBuffer(
          clBuffers[5], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(int), clImageOut.B);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: reading " << myStatus << std::endl;
      }

      commandQueueCPU.finish();
     
      GetTimeInfo();
    }

    void BlurImageGPU() {
      clImageOut = clImageIn;

      cl_int myStatus = CL_SUCCESS;

      numberArguments = 6;
      arraySizes = new int[numberArguments];
      arraySizes[0] = clImageIn.height * clImageIn.width * sizeof(float);
      arraySizes[1] = clImageIn.height * clImageIn.width * sizeof(float);

      arraySizes[2] = clImageIn.height * clImageIn.width * sizeof(float);
      arraySizes[3] = clImageIn.height * clImageIn.width * sizeof(float);

      arraySizes[4] = clImageIn.height * clImageIn.width * sizeof(float);
      arraySizes[5] = clImageIn.height * clImageIn.width * sizeof(float);

      SetBuffers(numberArguments, arraySizes);
      SetKernel("blur");
     

      myStatus = clKernel.setArg(2, clImageIn.width);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 2 " << myStatus << std::endl;
      }
      myStatus = clKernel.setArg(3, clImageIn.height);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 3 " << myStatus << std::endl;
      }
      myStatus = clKernel.setArg(4, blurStep);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 4 " << myStatus << std::endl;
      }

      cl::NDRange globalSize(clImageIn.width, clImageIn.height);

      myStatus = commandQueueGPU.enqueueWriteBuffer(
          clBuffers[2], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(float), clImageIn.R);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: writing " << myStatus << std::endl;
      }

      myStatus = commandQueueGPU.enqueueWriteBuffer(
          clBuffers[0], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(float), clImageIn.G);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: writing " << myStatus << std::endl;
      }

      myStatus = commandQueueGPU.enqueueWriteBuffer(
          clBuffers[4], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(float), clImageIn.B);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: writing " << myStatus << std::endl;
      }

      commandQueueGPU.finish();

      StartTime();

      // R

      myStatus = clKernel.setArg(1, clBuffers[3]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 1 " << myStatus << std::endl;
      }

      myStatus = clKernel.setArg(0, clBuffers[2]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 0 " << myStatus << std::endl;
      }

      myStatus = commandQueueGPU.enqueueNDRangeKernel(
          clKernel, cl::NDRange(0, 0), globalSize);

      // G
      myStatus = clKernel.setArg(1, clBuffers[1]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 1 " << myStatus << std::endl;
      }

      myStatus = clKernel.setArg(0, clBuffers[0]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 0 " << myStatus << std::endl;
      }
      myStatus = commandQueueGPU.enqueueNDRangeKernel(
          clKernel, cl::NDRange(0, 0), globalSize);

      // B
      myStatus = clKernel.setArg(1, clBuffers[5]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 2 " << myStatus << std::endl;
      }

      myStatus = clKernel.setArg(0, clBuffers[4]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 0 " << myStatus << std::endl;
      }

      myStatus = commandQueueGPU.enqueueNDRangeKernel(
          clKernel, cl::NDRange(0, 0), globalSize);

      // Read
      commandQueueGPU.finish();

      myStatus = commandQueueGPU.enqueueReadBuffer(
          clBuffers[3], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(int), clImageOut.R);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: reading " << myStatus << std::endl;
      }

      myStatus = commandQueueGPU.enqueueReadBuffer(
          clBuffers[1], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(int), clImageOut.G);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: reading " << myStatus << std::endl;
      }

      myStatus = commandQueueGPU.enqueueReadBuffer(
          clBuffers[5], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(int), clImageOut.B);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: reading " << myStatus << std::endl;
      }

      commandQueueGPU.finish();

      FinishTime();
      GetTimeInfo();
    }

    void BlurImageGPU70_CPU30() {
      clImageOut = clImageIn;

      cl_int myStatus = CL_SUCCESS;

      numberArguments = 6;
      arraySizes = new int[numberArguments];
      arraySizes[0] = clImageIn.height * clImageIn.width * sizeof(float);
      arraySizes[1] = clImageIn.height * clImageIn.width * sizeof(float);

      arraySizes[2] = clImageIn.height * clImageIn.width * sizeof(float);
      arraySizes[3] = clImageIn.height * clImageIn.width * sizeof(float);

      arraySizes[4] = clImageIn.height * clImageIn.width * sizeof(float);
      arraySizes[5] = clImageIn.height * clImageIn.width * sizeof(float);

      SetBuffers(numberArguments, arraySizes);
      SetKernel("blur");

      myStatus = clKernel.setArg(2, clImageIn.width);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 2 " << myStatus << std::endl;
      }
      myStatus = clKernel.setArg(3, clImageIn.height);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 3 " << myStatus << std::endl;
      }
      myStatus = clKernel.setArg(4, blurStep);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 4 " << myStatus << std::endl;
      }

      cl::NDRange globalSize(clImageIn.width, clImageIn.height);

      myStatus = commandQueueGPU.enqueueWriteBuffer(
          clBuffers[2], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(float), clImageIn.R);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: writing " << myStatus << std::endl;
      }

      myStatus = commandQueueGPU.enqueueWriteBuffer(
          clBuffers[0], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(float), clImageIn.G);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: writing " << myStatus << std::endl;
      }

      myStatus = commandQueueGPU.enqueueWriteBuffer(
          clBuffers[4], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(float), clImageIn.B);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: writing " << myStatus << std::endl;
      }

      commandQueueCPU.finish();
      commandQueueGPU.finish();

      StartTime();

      // R

      myStatus = clKernel.setArg(1, clBuffers[3]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 1 " << myStatus << std::endl;
      }

      myStatus = clKernel.setArg(0, clBuffers[2]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 0 " << myStatus << std::endl;
      }

      myStatus = commandQueueCPU.enqueueNDRangeKernel(
          clKernel, cl::NDRange(0, 0), globalSize);

      // G
      myStatus = clKernel.setArg(1, clBuffers[1]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 1 " << myStatus << std::endl;
      }

      myStatus = clKernel.setArg(0, clBuffers[0]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 0 " << myStatus << std::endl;
      }
      myStatus = commandQueueGPU.enqueueNDRangeKernel(
          clKernel, cl::NDRange(0, 0), globalSize);

      // B
      myStatus = clKernel.setArg(1, clBuffers[5]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 2 " << myStatus << std::endl;
      }

      myStatus = clKernel.setArg(0, clBuffers[4]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 0 " << myStatus << std::endl;
      }

      myStatus = commandQueueGPU.enqueueNDRangeKernel(
          clKernel, cl::NDRange(0, 0), globalSize);

      // Read
      commandQueueGPU.finish();
      commandQueueCPU.finish();
      FinishTime();

      myStatus = commandQueueCPU.enqueueReadBuffer(
          clBuffers[3], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(int), clImageOut.R);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: reading " << myStatus << std::endl;
      }

      myStatus = commandQueueGPU.enqueueReadBuffer(
          clBuffers[1], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(int), clImageOut.G);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: reading " << myStatus << std::endl;
      }

      myStatus = commandQueueGPU.enqueueReadBuffer(
          clBuffers[5], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(int), clImageOut.B);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: reading " << myStatus << std::endl;
      }

      commandQueueGPU.finish();
      commandQueueCPU.finish();
      
      GetTimeInfo();
    }

    void BlurImageGPU30_CPU70() {
      clImageOut = clImageIn;

      cl_int myStatus = CL_SUCCESS;

      numberArguments = 6;
      arraySizes = new int[numberArguments];
      arraySizes[0] = clImageIn.height * clImageIn.width * sizeof(float);
      arraySizes[1] = clImageIn.height * clImageIn.width * sizeof(float);

      arraySizes[2] = clImageIn.height * clImageIn.width * sizeof(float);
      arraySizes[3] = clImageIn.height * clImageIn.width * sizeof(float);

      arraySizes[4] = clImageIn.height * clImageIn.width * sizeof(float);
      arraySizes[5] = clImageIn.height * clImageIn.width * sizeof(float);

      SetBuffers(numberArguments, arraySizes);
      SetKernel("blur");

      myStatus = clKernel.setArg(2, clImageIn.width);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 2 " << myStatus << std::endl;
      }
      myStatus = clKernel.setArg(3, clImageIn.height);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 3 " << myStatus << std::endl;
      }
      myStatus = clKernel.setArg(4, blurStep);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 4 " << myStatus << std::endl;
      }

      cl::NDRange globalSize(clImageIn.width, clImageIn.height);

      myStatus = commandQueueGPU.enqueueWriteBuffer(
          clBuffers[2], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(float), clImageIn.R);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: writing " << myStatus << std::endl;
      }

      myStatus = commandQueueCPU.enqueueWriteBuffer(
          clBuffers[0], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(float), clImageIn.G);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: writing " << myStatus << std::endl;
      }

      myStatus = commandQueueCPU.enqueueWriteBuffer(
          clBuffers[4], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(float), clImageIn.B);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: writing " << myStatus << std::endl;
      }

      commandQueueCPU.finish();
      commandQueueGPU.finish();

      StartTime();

      // R

      myStatus = clKernel.setArg(1, clBuffers[3]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 1 " << myStatus << std::endl;
      }

      myStatus = clKernel.setArg(0, clBuffers[2]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 0 " << myStatus << std::endl;
      }

      myStatus = commandQueueGPU.enqueueNDRangeKernel(
          clKernel, cl::NDRange(0, 0), globalSize);

      // G
      myStatus = clKernel.setArg(1, clBuffers[1]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 1 " << myStatus << std::endl;
      }

      myStatus = clKernel.setArg(0, clBuffers[0]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 0 " << myStatus << std::endl;
      }
      myStatus = commandQueueCPU.enqueueNDRangeKernel(
          clKernel, cl::NDRange(0, 0), globalSize);

      // B
      myStatus = clKernel.setArg(1, clBuffers[5]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 2 " << myStatus << std::endl;
      }

      myStatus = clKernel.setArg(0, clBuffers[4]);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: SetArg 0 " << myStatus << std::endl;
      }

      myStatus = commandQueueCPU.enqueueNDRangeKernel(
          clKernel, cl::NDRange(0, 0), globalSize);

      // Read
      commandQueueGPU.finish();
      commandQueueCPU.finish();
      FinishTime();

      myStatus = commandQueueGPU.enqueueReadBuffer(
          clBuffers[3], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(int), clImageOut.R);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: reading " << myStatus << std::endl;
      }

      myStatus = commandQueueCPU.enqueueReadBuffer(
          clBuffers[1], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(int), clImageOut.G);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: reading " << myStatus << std::endl;
      }

      myStatus = commandQueueCPU.enqueueReadBuffer(
          clBuffers[5], CL_FALSE, 0,
          clImageIn.height * clImageIn.width * sizeof(int), clImageOut.B);
      if (myStatus != CL_SUCCESS) {
        std::cout << "ERROR: reading " << myStatus << std::endl;
      }

      commandQueueGPU.finish();
      commandQueueCPU.finish();
     
      GetTimeInfo();
    }

};



