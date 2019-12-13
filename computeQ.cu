/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

#define BLOCK_SIZE 512
#define KV_SIZE 2048
__constant__ __device__ kValues KV[KV_SIZE];

__global__ void ComputePhiMagKernel(int numK, float *phiR, float *phiI,
                                    float *phiMag)
{
  int m = threadIdx.x + (blockIdx.x * blockDim.x);
  if (m < numK){
    float R = phiR[m];
    float I = phiI[m];
    phiMag[m] = (R * R) + (I * I);
  }
}

__global__ void ComputeQKernel(int numK, int numX,
                               float *x_d, float *y_d, float *z_d,
                               float *Qr_d, float *Qi_d)
{
  int m = threadIdx.x + (blockIdx.x * blockDim.x);
  if (m >= numX)
    return;
  float temp_x = x_d[m];
  float temp_y = y_d[m];
  float temp_z = z_d[m];
  float temp_Qr = 0.0f;
  float temp_Qi = 0.0f;
  float expArg;
  float temp_phi = 0.0f;
  for (int i=0; i < numK; i++) {
    expArg = PIx2 * (KV[i].Kx * temp_x +
                     KV[i].Ky * temp_y +
                     KV[i].Kz * temp_z);
    temp_phi = KV[i].PhiMag;
    temp_Qr += temp_phi * cos(expArg);
    temp_Qi += temp_phi * sin(expArg);
  }
  Qr_d[m] += temp_Qr;
  Qi_d[m] += temp_Qi;
}


void ComputePhiMagGPU(int numK, float* phiR_d, float* phiI_d,
                      float* phiMag_d)
{
  unsigned int numBlocks = ((numK - 1) / BLOCK_SIZE) + 1;
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  ComputePhiMagKernel<<<dimGrid, dimBlock>>>(numK, phiR_d, phiI_d, phiMag_d);
}

void ComputeQGPU(int numK, int numX, struct kValues *kVals,
                 float *x_d, float *y_d, float *z_d, float *Qr_d, float *Qi_d)
{
  unsigned int Q_size = KV_SIZE;
  unsigned int Q_n = ((numK - 1) / KV_SIZE) + 1;
  struct kValues *ptr = kVals;

  unsigned int numBlocks = ((numX - 1) / BLOCK_SIZE) + 1;
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  for (int i=0; i<Q_n; i++) {
    Q_size = MIN(KV_SIZE, numK - (i * KV_SIZE));
    if (Q_size) {
        cudaMemcpyToSymbol(KV, ptr, Q_size * sizeof(struct kValues), 0);
        ComputeQKernel<<<dimGrid, dimBlock>>>(Q_size, numX, x_d, y_d, z_d, Qr_d, Qi_d);
    }
    ptr += Q_size;
  }

}

/*
inline
void
ComputePhiMagCPU(int numK,
                 float* phiR, float* phiI,
                 float* __restrict__ phiMag) {
  int indexK = 0;
  for (indexK = 0; indexK < numK; indexK++) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}

inline
void
ComputeQCPU(int numK, int numX,
            struct kValues *kVals,
            float* x, float* y, float* z,
            float *__restrict__ Qr, float *__restrict__ Qi) {
  float expArg;
  float cosArg;
  float sinArg;

  int indexK, indexX;

  // Loop over the space and frequency domains.
  // Generally, numX > numK.
  // Since loops are not tiled, it's better that the loop with the smaller
  // cache footprint be innermost.
  for (indexX = 0; indexX < numX; indexX++) {

    // Sum the contributions to this point over all frequencies
    float Qracc = 0.0f;
    float Qiacc = 0.0f;
    for (indexK = 0; indexK < numK; indexK++) {
      expArg = PIx2 * (kVals[indexK].Kx * x[indexX] +
                       kVals[indexK].Ky * y[indexX] +
                       kVals[indexK].Kz * z[indexX]);

      cosArg = cosf(expArg);
      sinArg = sinf(expArg);

      float phi = kVals[indexK].PhiMag;
      Qracc += phi * cosArg;
      Qiacc += phi * sinArg;
    }
    Qr[indexX] = Qracc;
    Qi[indexX] = Qiacc;
  }
}
*/

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qr, 0, numX * sizeof(float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qi, 0, numX * sizeof(float));
}
