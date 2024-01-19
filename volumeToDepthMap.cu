__constant__ float parameters[4]={3.1859233711350760e+002,2.4926027230995192e+002,5.2338812559311623e+002,5.2332543643433257e+002};  //cX cY fX fY
__constant__ float scale=1.0f;
__constant__ float cameraPosition[16];//Matrix used to transform points on projection plane to camera position

__global__ void renderDepthMap( volumeType *volume,
									int width, int height, //Separation between pixels in length units, the width and size of the image.
								   depthMapType *depthMap){ //depthMap
	float eyeX,eyeY,eyeZ; //vector to the camera position
	float vX,vY,vZ; //vector from 'eye' to the point of the projection plane that each thread will process.
	float normV;
	float nX,nY,nZ; //normal vector to projection plane
	float stepX,stepY,stepZ;
	float posX,posY,posZ;
	float aux;
	const int maxSteps = 500; //maximum number of steps to march ray
	bool hit;

	for (int globalTidY=threadIdx.y+blockIdx.y*blockDim.y;globalTidY<height;globalTidY+=gridDim.y*blockDim.y){
		for(int globalTidX=threadIdx.x+blockIdx.x*blockDim.x;globalTidX<width;globalTidX+=gridDim.x*blockDim.x){

			//initialize pos to the point of the projection plane that each thread will process
			posX=(globalTidX-parameters[0]) / (parameters[2]*scale);
			posY=1;
			posZ=-1.0f*(globalTidY-parameters[1]) / (parameters[3]*scale);

			//Transform points of virtual projection plane to match the position and orientation of the camera.
			vX= cameraPosition[0]*posX + cameraPosition[1]*posY
				+ cameraPosition[2]*posZ + cameraPosition[3];
			vY= cameraPosition[4]*posX + cameraPosition[5]*posY
				+ cameraPosition[6]*posZ + cameraPosition[7];
			vZ=cameraPosition[8]*posX + cameraPosition[9]*posY
			   + cameraPosition[10]*posZ + cameraPosition[11];

			eyeX=cameraPosition[3];
			eyeY=cameraPosition[7];
			eyeZ=cameraPosition[11];

			vX-=eyeX;
			vY-=eyeY;
			vZ-=eyeZ;
			normV=sqrtf(vX*vX+vY*vY+vZ*vZ);
			vX/=normV; vY/=normV; vZ/=normV;

			nX=cameraPosition[1];
			nY=cameraPosition[5];
			nZ=cameraPosition[9];
			aux=sqrtf(nX*nX+nY*nY+nZ*nZ);
			nX/=aux; nY/=aux; nZ/=aux;

			posX=eyeX+vX*normV;
			posY=eyeY+vY*normV;
			posZ=eyeZ+vZ*normV;
			stepX=vX;
			stepY=vY;
			stepZ=vZ;
			hit=false;
			for (int i=0; i<maxSteps; i++){
				// remap position to grid coordinates
				int x=posX+X/2;
				int y=posY+Y/2;
				int z=posZ+Z/2;

				if(0<=x&&x<X && 0<=y&&y<Y && 0<=z&&z<Z){ //if in volume boundaries
					volumeType sample=volume[X*Y*z+X*y+x]; //sample volume
					if(sample>0){
						i=maxSteps;
						hit=true;
					}
					else{
						posX += stepX;
						posY += stepY;
						posZ += stepZ;
					}
				}
				else{
					i=maxSteps;
				}
			}
			depthMap[globalTidY*width+globalTidX]=hit?(depthMapType) sqrtf((posX-eyeX)*(posX-eyeX)+(posY-eyeY)*(posY-eyeY)+(posZ-eyeZ)*(posZ-eyeZ)):0;
			//depthMap[globalTidY*width+globalTidX]=hit?(depthMapType) ((posX-eyeX)*nX+(posY-eyeY)*nY+(posZ-eyeZ)*nZ):0; // distance to projection plane.
		}
	}
}