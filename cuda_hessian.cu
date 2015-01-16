/*
This is the CUDA implementation of hessian.cpp

*/

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <math.h>
#include <algorithm>

#include <cuda_runtime.h>


#include "responselayer.h"


using namespace std;

const int width = 640;
const int height = 466;

const int OCTAVES = 4;
const int INTERVALS = 4;
const int step = 1;
int scales[OCTAVES][INTERVALS] = {{9,15,21,27}, {15,27,39,51}, {27,51,75,99},
51,99,147,195};

float *cuda_img, **cuda_responses;



void loadImg(float *, char *, int, int);
__device__ float BoxIntegral(float *, int, int, int, int);


void loadImg(float *img, char* fname, int w, int h)
{
    
    ifstream readstream;
    readstream.open(fname);
    char comma;
    for(int i=0; i<w*h; ++i) {
        readstream >> img[i];
        readstream >> comma;
    }
    readstream.close();
}

/*void checkImg(float *img, int r, int c) */
/*{*/
    /*float res = img[r*width + c];*/
    /*printf("%f\n", res);*/
/*}*/



__global__ void buildResponseLayer(float* img, int height, int width, int step, int filter, float *
        responses)
{
    int b = (filter - 1) / 2 + 1;
    int l = filter / 3;
    int w = filter;
    float inverse_area = 1.f/(w*w);
    float Dxx, Dyy, Dxy;


    int idx = blockIdx.x * blockDim.x + blockIdx.y;

    for(int r, c, ar = 0, index = 0; ar < height; ++ar)
    {
        for(int ac = idx; ac < width; ac+=idx, index+=idx)
        {
            r = ar * step;
            c = ac * step;

            Dxx = BoxIntegral(img, r - l + 1, c - b, 2*l - 1, w)
                    - BoxIntegral(img, r - l + 1, c - l / 2, 2*l - 1, l)*3;
            Dyy = BoxIntegral(img, r - b, c - l + 1, w, 2*l - 1)
                    - BoxIntegral(img, r - l / 2, c - l + 1, l, 2*l - 1)*3;
            Dxy = + BoxIntegral(img, r - l, c + 1, l, l)
                    + BoxIntegral(img, r + 1, c - l, l, l)
                    - BoxIntegral(img, r - l, c - l, l, l)
                    - BoxIntegral(img, r + 1, c + 1, l, l);

            Dxx *= inverse_area;
            Dyy *= inverse_area;
            Dxy *= inverse_area;

            responses[index] = (Dxx * Dyy - 0.9f *Dxy * Dxy);
        }
    }
}

void checkResponse(ResponseLayer *rl)
{
    printf("response for layer w:%d h:%d s:%d oct:%d\n", rl->width, rl->height,
            rl->step, rl->filter);
    float * responses = rl->responses;
    
    for(int i=0; i<1; ++i)
    {
        for(int j=0; j<rl->width; ++j)
            printf("%f ",responses[i*rl->width+j]);
        printf("\n");
    }

}

__device__ float BoxIntegral(float* img, int row, int col, int rows, int cols)
{
    int r1 = min(row, height) - 1;
    int c1 = min(col, width) - 1;
    int r2 = min(row + rows, height) - 1;
    int c2 = min(col + cols, width) - 1;

    float A(0.0f), B(0.0f), C(0.0f), D(0.0f);
    if (r1 >=0 && c1 >= 0) A = img[r1*step + c1];
    if (r1 >=0 && c2 >= 0) B = img[r1*step + c2];
    if (r2 >=0 && c1 >= 0) C = img[r2*step + c1];
    if (r2 >=0 && c2 >= 0) D = img[r2*step + c2];

    return max(0.f, A - B - C + D);
}



int main()
{
    // load integral image
    float *img;
    img = new float[width*height];
    char* testfile = "integral.csv";
    loadImg(img, testfile, width, height);

    // build response map
    vector<ResponseLayer *> responseMap;
    
    for (int oct=0; oct<OCTAVES; ++oct) {
        for (int inter=0; inter<INTERVALS; ++inter) {
            responseMap.push_back(new ResponseLayer (width/(int) pow(2.0,oct),
                        height/(int) pow(2.0,oct), step*(int)pow(2.0,oct), scales[oct][inter]));
        }
    }

    dim3 grid(1,1);
    dim3 block(32);

    int img_size = width*height*sizeof(float);
    printf("img_size: %d\n", img_size);
    cuda_responses = new float*[responseMap.size()];

    cudaMalloc((void **) &cuda_img, img_size);
    cudaMemcpy(cuda_img, img, img_size, cudaMemcpyHostToDevice);

    for(int i=0; i<responseMap.size(); ++i) {
        ResponseLayer *tmp = responseMap[i];
        int response_size = (tmp->width) * (tmp->height) * sizeof(float);
        cudaMalloc((void **)&cuda_responses[i], response_size);
        buildResponseLayer<<<grid, block>>>(cuda_img, tmp->height, tmp->width, tmp->step, tmp->filter, cuda_responses[i]);
        cudaMemcpy(tmp->responses, cuda_responses[i], response_size,
                cudaMemcpyDeviceToHost);

    }

    cudaFree(cuda_img);
    for(int i=0; i<responseMap.size(); ++i) {
        cudaFree(cuda_responses[i]);
    }

    delete cuda_responses;



    // nonparallel implementation
    /*for(int i=0; i<responseMap.size(); ++i) {*/
        /*ResponseLayer *tmp = responseMap[i];*/
        /*[>printf("%d %d %d %d\n", tmp->width, tmp->height, tmp->step,<]*/
            /*[>tmp->filter);<]*/
        /*buildResponseLayer(img, responseMap[i]);*/
    /*}*/

    
    checkResponse(responseMap[0]);



    return 0;
}


