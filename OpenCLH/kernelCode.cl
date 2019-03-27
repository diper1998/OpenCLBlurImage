kernel void blur(__global float* in,  __global float* out,  int width, int height, int step)
{ 
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int h = 0;
	
	int n = 1;

	out[gy*width+gx] = 0;
	
	
	for(h = 0; h < step; h++){
	
	//
	if(( (gy-h)*width+(gx-h) < width*height ) && ( (gy-h)*width+(gx-h) >= 0 ) ){
	out[gy*width+gx] += in[(gy-h)*width+(gx-h)];
	n++;
	}

	if(( (gy-h)*width+(gx) < width*height ) && ( (gy-h)*width+(gx) >= 0 ) ){
	out[gy*width+gx] += in[(gy-h)*width+(gx)];
	n++;
	}

	if(( (gy-h)*width+(gx+h) < width*height ) && ( (gy-h)*width+(gx+h) >= 0 ) ){
	out[gy*width+gx] += in[(gy-h)*width+(gx+h)];
	n++;
	}
	//
	//
	if(( (gy)*width+(gx-h) < width*height ) && ( (gy)*width+(gx-h) >= 0 ) ){
	out[gy*width+gx] += in[(gy)*width+(gx-h)];
	n++;
	}

	if(( (gy)*width+(gx+h)  < width*height) && ( (gy)*width+(gx+h) >= 0 ) ){
	out[gy*width+gx] += in[(gy)*width+(gx+h)];
	n++;
	}
	//
	//
	if(( (gy+h)*width+(gx-h) < width*height ) && ( (gy+h)*width+(gx-h) >= 0 ) ){
	out[gy*width+gx] += in[(gy+h)*width+(gx-h)];
	n++;
	}

	if(( (gy+h)*width+(gx) < width*height ) && ( (gy+h)*width+(gx) >= 0 ) ){
	out[gy*width+gx] += in[(gy+h)*width+(gx)];
	n++;
	}

	if(( (gy+h)*width+(gx+h) < width*height ) && ( (gy+h)*width+(gx+h)>= 0 ) ){
	out[gy*width+gx] += in[(gy+h)*width+(gx+h)];
	n++;
	}
    //


	}

	out[gy*width+gx] /= n;

	//printf("x: %d, y: %d, x+y: %d, in: %f \n", gx, gy, gy*width+gx , out[gy*width+gx] );

}


