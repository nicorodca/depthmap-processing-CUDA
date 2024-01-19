__constant__ float parameters[4]={3.1859233711350760e+002,2.4926027230995192e+002,5.2338812559311623e+002,5.2332543643433257e+002};  //cX cY fX fY
__constant__ float cameraToWorldMatrix[16];  
__constant__ float scale=10.0f; 

__global__ void convert2World(unsigned short* depthMap,float *x,float *y, float *z, int w, int h){
		float xW=NAN,yW=NAN,zW=NAN;
		unsigned short depthValue;
		for(int globalTidY=threadIdx.y+blockIdx.y*blockDim.y;globalTidY<h;globalTidY+=gridDim.y*blockDim.y){
			for(int globalTidX=threadIdx.x+blockIdx.x*blockDim.x;globalTidX<w;globalTidX+=gridDim.x*blockDim.x){
				depthValue=depthMap[globalTidY*w+globalTidX];
				if(depthValue>0){
					float xWTmp=(globalTidX-parameters[0])* depthValue / (parameters[2]*scale);
					float yWTmp=depthValue/scale;
					float zWTmp=-1.0f*(globalTidY-parameters[1])* depthValue / (parameters[3]*scale);

					xW = cameraToWorldMatrix[0]*xWTmp + cameraToWorldMatrix[1]*yWTmp
							+ cameraToWorldMatrix[2]*zWTmp + cameraToWorldMatrix[3];
					yW = cameraToWorldMatrix[4]*xWTmp + cameraToWorldMatrix[5]*yWTmp
							+ cameraToWorldMatrix[6]*zWTmp + cameraToWorldMatrix[7];
					zW = cameraToWorldMatrix[8]*xWTmp + cameraToWorldMatrix[9]*yWTmp
							+ cameraToWorldMatrix[10]*zWTmp + cameraToWorldMatrix[11];
				}

				x[globalTidY*w+globalTidX]=xW;
				y[globalTidY*w+globalTidX]=yW;
				z[globalTidY*w+globalTidX]=zW;
			}
		}
}