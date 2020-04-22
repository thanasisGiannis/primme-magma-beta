#ifndef auxiliary_eigs_gpu_H
#define auxiliary_eigs_gpu_H

#ifdef __NVCC__
void NumGPU_compute_residual_dprimme(int64_t n, double eval, double *x,double *Ax, double *r,primme_params *params);
void NumGPU_compute_residual_zprimme(int64_t n, PRIMME_COMPLEX_DOUBLE eval, PRIMME_COMPLEX_DOUBLE *x, PRIMME_COMPLEX_DOUBLE *Ax, 								  PRIMME_COMPLEX_DOUBLE *r,primme_params *params);
void NumGPU_compute_residual_sprimme(int64_t n, float eval, float *x, float *Ax, float *r, primme_params *params);

void NumGPU_compute_residual_cprimme(int64_t n, PRIMME_COMPLEX_FLOAT eval, PRIMME_COMPLEX_FLOAT *x,
							   PRIMME_COMPLEX_FLOAT *Ax, PRIMME_COMPLEX_FLOAT *r,primme_params *params);
#endif
#endif
