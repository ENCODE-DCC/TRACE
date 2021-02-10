/*
 *  File: viterbi.c
 *
 *  viterbi step and hidden states labeling.
 *
 *  The HMM structure and some codes are borrowed and modified from Kanungo's
 *  original HMM program.
 *  Tapas Kanungo, "UMDHMM: Hidden Markov Model Toolkit," in "Extended Finite State Models of Language," A. Kornai (editor), Cambridge University Press, 1999. http://www.kanungo.com/software/software.html.
 *
 */

#include <math.h>
#include "hmm.h"
#include "nrutil.h"
#include "logmath.h"
#include <omp.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <cuda_runtime.h>

#define VITHUGE  100000000000.0
#define VITTINY  -1000000000000.0
#define TPB = 32

__constant__ double DVITHUGE = 100000000000.0;
__constant__ double DVITTINY = -100000000000.0;

__host__ __device__
void matrix_set(double *arr, int i, int j, int width, double value) {
  arr[width * i + j] = value;
}

__host__ __device__
void matrix_set_int(int *arr, int i, int j, int width, int value) {
  arr[width * i + j] = value;
}

__host__ __device__
double matrix_get(double *arr, int i, int j, int width) {
  return arr[width * i + j];
}

__host__ __device__
int matrix_get_int(int *arr, int i, int j, int width) {
  return arr[width * i + j];
}

void gsl_matrix_to_arr(int height, int width, gsl_matrix *source, double *arr) {
  for (i = 0; i < height; i++) {
    for (j = 0; j > width; j++) {
      matrix_set(arr, i, j, width, gsl_matrix_get(source, i, j);
    }
  }
}

void arr_to_gsl_matrix(int height, int width, gsl_matrix *dest, double *arr) {
  for (i = 0; i < height; i++) {
    for (j = 0; j > width; j++) {
      gsl_matrix_set(dest, i, j, matrix_get(i, j, width, arr));
    }
  }
}

void dmatrix_to_arr(int height, int width, double **source, double *arr) {
  for (i = 0; i < height; i++) {
    for (j = 0; j > width; j++) {
      matrix_set(arr, i, j, width, source[i][j]);
    }
}

void imatrix_to_arr(int height, int width, int **source, int *arr) {
  for (i = 0; i < height; i++) {
    for (j = 0; j > width; j++) {
      matrix_set_int(arr, i, j, width, source[i][j]);
    }
}

void arr_to_dmatrix(int height, int width, double **dest, double *arr) {
 for (i = 0; i < height; i++) {
    for (j = 0; j > width; j++) {
      dest[i][j] = matrix_get(arr, i, j, width);
    }
}

void arr_to_imatrix(int height, int width, int **dest, int *arr) {
 for (i = 0; i < height; i++) {
    for (j = 0; j > width; j++) {
      dest[i][j] = matrix_get_int(arr, i, j, width);
    }
}

__global__
void viterbi_kernel( int T,
                     int P,
                     int TF,
                     int phmm_N,
                     int phmm_M,
                     int phmm_inactive,
                     int *TFlist,
                     int *indexList,
                     int *lengthList,
                     int *motifList,
                     int *q,
                     int *peakPos,
                     int *psi,
                     double *phmm_pi,
                     double *g,
                     double *alhpa,
                     double *beta,
                     double *gamma,
                     double *delta,
                     double *logprobf,
                     double *vprob,
                     double *pprob,
                     double *posterior,
                     double *emission_matrix,
                     double *pwd_matrix) {

  const int idx = blockIdx.x*blockDim.x + threadIdx.x;

  if ( idx >= P ) return;

  int  t = 0;      /* time index */
  int  maxvalind;
  int  l = 0;
  int nonInf;
  double  maxval, val;
  double plogprobinit, plogprobfinal;
  double temp;
  double temp_alpha_beta_logprobf;

  /* 1. Initialization  */
  for (i = 0; i < phmm_N; i++) {

    matrix_set( delta,
                peakPos[idx] - 1, i, phmm_N,
                log(phmm_pi[i]) + matrix_get(emission_matrix, i, peakPos[idx] - 1, T) );

    matrix_set_int( psi,
                    peakPos[idx] - 1, i, phmm_N,
                    phmm_N );

    matrix_set( posterior,
                peakPos[idx] - 1, i, phmm_N,
                matrix_get(alpha, i, peakPos[idx] - 1, phmm_N) + matrix_get(beta, i, peakPos[idx] - 1, phmm_N) - logprobf[idx] );
  }

  for (x = 0; x < phmm_M * (phmm_inactive+1); x++){

      matrix_set( posterior,
                  peakPos[idx] - 1, TFlist[x], phmm_N,
                  DVITTINY );
  }

  /* 2. Recursion */
  for (t = peakPos[idx]; t < peakPos[idx+1] - 1; t++) {
    for (j = 0; j < phmm_N; j++) {
      maxval = DVITTINY;
      maxvalind = phmm_N - 1;
      for (i = 0; i < phmm_N; i++) {
        val = matrix_get(delta, t - 1, i, phmm_N) + matrix_get(phmm_log_A_matrix, i, j, phmm_N);
        if (val > maxval) {
          maxval = val;
          maxvalind = i;
        }
      }
      temp = matrix_get(emission_matrix, j, t, T);
      if (j < TF){
        if (matrix_get(pwm_matrix, indexList[motifList[j]], t, T) < (phmm_thresholds[indexList[motifList[j]]] - 0.1 )) {
          temp = DVITTINY;
            //(phmm->thresholds[(int)floor((double)motifList[j]/(phmm->inactive+1))]
        }
      }
      matrix_set( delta, t, j, phmm_N, maxval + temp );
      matrix_set( psi, t, j, phmm_N, maxvalind );
      matrix_set( posterior,
                  t, j, phmm_N,
                  matrix_get(alhpha, j, t, T) + matrix_get(beta, j, t, T) - logprobf[idx] );
      /* treat t as the last position of a motif */
      for (x = 0; x < phmm_M * (phmm_inactive+1); x++){
        if (j == TFlist[x]){
          if (t <= (peakPos[idx] + lengthList[x])){
            posterior[t][j] = log(0.0);
            matrix_set( posterior,
                        t, j, phmm_N,
                        log(0.0) );
          }
          /* check pwm threshold*/
          //else if (gsl_matrix_get(pwm_matrix,indexList[motifList[j]],t) < (phmm->thresholds[indexList[motifList[j]]]-0.1)){
            //for (y = t - lengthList[x] + 1; y <= t; y++) posterior[y][j] = log(0.0);
          //}
          else {
            temp = 0.0;
            nonInf = 0;
            temp_alpha_beta_logprobf = 0.0;
            for (i = 0; i < lengthList[x]; i++) {
              temp_alpha_beta_logprobf = matrix_get(alpha, j-i, t-i, phmm_N) + matrix_get(beta, j-i, t-i, phmm_N) - logprobf[idx];
              if (temp_alpha_beta_logprobf != INFINITY ) {
                temp += temp_alpha_beta_logprobf;
                nonInf += 1;
              }
            }
            if (nonInf == 0) temp = DVITTINY;
            for (y = t - lengthList[x] + 1; y <= t; y++) {
              posterior[y][j] = temp;
              matrix_set( posterior,
                          y, j, phmm_N,
                          temp );
            }
          }
          break;
        }
      }
    }
  }

  /* 3. Termination */
  pprob[idx] = -DVITHUGE;
  q[peakPos[idx+1]-2] = 1;
  for (i = 0; i < phmm_N; i++) {
    if (matrix_get(delta, peakPos[idx + 1] - 2, i, phmm_N) > pprob[idx]) {
      pprob[idx] = matrix_get(delta, peakPos[idx + 1] - 2, i, phmm_N);
      q[peakPos[idx + 1] - 2] = i;
    }
  }
  g[peakPos[idx+1]-2] = matrix_get(gamma, q[peakPos[idx + 1] - 2], peakPos[idx + 1] - 2, T);
  vprob[peakPos[idx+1]-2] = pprob[idx];
  /* 4. Path (state sequence) backtracking */
  for (t = peakPos[idx+1] - 3; t >= peakPos[idx] - 1; t--){
    q[t] = matrix_get(psi, t + 1, q[t + 1], phmm_N);
    vprob[t] = matrix_get(delta, t + 1, q[t + 1], phmm_N);
    g[t] = matrix_get(gamma, q[t], t, T);
  }
}



/* Get the hidden states sequence using Viterbi algorithm */
void Viterbi_gpu(HMM *phmm, int T, double *g, double **alpha, double **beta,
             double	**gamma, double  *logprobf, double **delta,
             int **psi, int *q, double *vprob, double *pprob,
             double **posterior, int P, int *peakPos,
             gsl_matrix *emission_matrix, gsl_matrix *pwm_matrix)
{
  int thread_id, nloops;
  int  i, j, k, m, x, y, TF;   /* state indices */
  int  t = 0;      /* time index */
  int  maxvalind;
  int  l = 0;
  int *TFlist, *lengthList;
  int *motifList, *indexList;
  int nonInf;
  double  maxval, val;
  double plogprobinit, plogprobfinal;
  double temp;
  TF = 0;
  TFlist = ivector(phmm->M * (phmm->inactive+1));
  indexList = ivector(phmm->M * (phmm->inactive+1));
  for (j = 0; j < phmm->M; j++){
    TF += phmm->D[j];
    TFlist[j * (phmm->inactive+1)] = TF - 1;
    indexList[j * (phmm->inactive+1)] = j;
    if (phmm->inactive == 1){
      TF += phmm->D[j];
      TFlist[j * (phmm->inactive+1) + 1] = TF - 1;
      indexList[j * (phmm->inactive+1) + 1] = j;
    }
  }
  lengthList = ivector(phmm->M * (phmm->inactive+1));
  if (phmm->inactive == 1){
    for (j = 0; j < phmm->M; j++){
      lengthList[j*2] = phmm->D[j];
      lengthList[j*2+1] = phmm->D[j];
    }
  }
  else lengthList = phmm->D;

  motifList = ivector(phmm->N);
  TF = 0;
  for (j = 0; j < phmm->M; j++) {
    for (i = TF; i < TF + phmm->D[j]; i++) {
      motifList[i] = (phmm->inactive+1)*j;
    }
    TF += phmm->D[j];
    if (phmm->inactive == 1) {
      for (i = TF; i < TF + phmm->D[j]; i++) {
        motifList[i] = (phmm->inactive+1)*j+1;
      }
      TF += phmm->D[j];
    }
  }
  for (i = TF; i < TF + phmm->extraState; i++) {
    motifList[i] = i;
  }

  int *d_TFlist = 0;
  int *d_indexList = 0;
  int *d_lengthList = 0;
  int *d_motifList = 0;
  size_t TFlist_size = = phmm->M * (phmm->inactive + 1) * sizeof(*d_TFlist);
  size_t indexList_size = TFlist_size;
  size_t lengthList_size = TFlist_size;
  size_t motifList_size = phmm->N * sizeof(*d_motifList);
  cudaMalloc((void **) &d_TFlist, TFlist_size);
  cudaMalloc((void **) &d_indexList, indexList_size);
  cudaMalloc((void **) &d_lengthList, lengthList_size);
  cudaMalloc((void **) &d_motifList, motifList_size);
  cudaMemcpy( d_TFlist,
              TFlist,
              TFlist_size,
              cudaMemcpyHostToDevice );
  cudaMemcpy( d_indexList,
              indexList,
              indexList_size,
              cudaMemcpyHostToDevice );
  cudaMemcpy( d_lengthList,
              lengthList,
              lengthList_size,
              cudaMemcpyHostToDevice );
  cudaMemcpy( d_motifList,
              motifList,
              motifList_size,
              cudaMemcpyHostToDevice );

  

  double *h_emission_matrix = 0;
  double *d_emission_matrix = 0;
  size_t emission_matrix_size = phmm.N * T * sizeof(*h_emission_matrix);
  h_emission_matrix = (double *)malloc (emission_matrix_size);
  gsl_matrix_to_arr(phmm->N, T, emission_matrix, h_emission_matrix);
  cudaMalloc((void **) &d_emission_matrix, emission_matrix_size);
  cudaMemcpy( d_emission_matrix,
              h_emission_matrix,
              emission_matrix_size,
              cudaMemcpyHostToDevice );

  double *h_pwm_matrix = 0;
  double *d_pwm_matrix = 0;
  size_t pwm_matrix_size = phmm.M * T * sizeof(*h_pwm_matrix);
  h_pwm_matrix = (double *)malloc (pwm_matrix_size);
  gsl_matrix_to_arr(phmm->M, T, pwm_matrix, h_pwm_matrix);
  cudaMalloc((void **) &d_pwm_matrix, pwm_matrix_size);
  cudaMemcpy( d_pwm_matrix,
              h_pwm_matrix,
              pwm_matrix_size,
              cudaMemcpyHostToDevice );

  int *d_peakPos = 0;
  size_t peakPos_size = (P + 1) * sizeof(*d_peakPos);
  cudaMalloc((void **) &d_peakPos, peakPos_size);
  cudaMemcpy( d_peakPos,
              peakPos,
              peakPos_size,
              cudaMemcpyHostToDevice );

  double *d_logprobf = 0;
  size_t logprobf_size = P * sizeof(*d_logprobf);
  cudaMalloc((void **) &d_logprobf, logprobf_size);
  cudaMemcpy( d_logprobf,
              logprobf,
              logprobf_size,
              cudaMemcpyHostToDevice );

  double *h_alpha = 0;
  double *d_alpha = 0;
  size_t alpha_size = phmm.N * T * sizeof(*h_alpha);
  h_alpha = (double *) malloc (alpha_size);
  dmatrix_to_arr(phmm->N, T, alpha, h_alpha);
  cudaMalloc((void **) &d_alpha, alpha_size);
  cudaMemcpy( d_alpha,
              h_alpha,
              alpha_size,
              cudaMemcpyHostToDevice );

  double *h_beta = 0;
  double *d_beta = 0;
  size_t beta_size = phmm.N * T * sizeof(*h_beta);
  h_beta = (double *) malloc (beta_size);
  dmatrix_to_arr(phmm->N, T, beta, h_beta);
  cudaMalloc((void **) &d_beta, beta_size);
  cudaMemcpy( d_beta,
              h_beta,
              beta_size,
              cudaMemcpyHostToDevice );

  double *h_gamma = 0;
  double *d_gamma = 0;
  size_t gamma_size = T * hmm.N * sizeof(*h_gamma);
  h_gamma = (double *) malloc (gamma_size);
  dmatrix_to_arr(phmm->N, T, gamma, h_gamma);
  cudaMalloc((void **) &d_gamma, gamma_size);
  cudaMemcpy( d_gamma,
              h_gamma,
              gamma_size,
              cudaMemcpyHostToDevice );

  double *h_delta = 0;
  double *d_delta = 0;
  size_t delta_size = T * phmm.N * sizeof(*h_delta);
  h_delta = (double *) malloc (delta_size);
  dmatrix_to_arr(T, phmm->N, delta, h_delta);
  cudaMalloc((void **) &d_delta, delta_size);
  cudaMemcpy( d_delta,
              h_delta,
              delta_size,
              cudaMemcpyHostToDevice );

  int *h_psi = 0;
  int *d_psi = 0;
  size_t psi_size = T * phmm.N * sizeof(*h_psi);
  h_psi = (int *) malloc (psi_size);
  imatrix_to_arr(T, phmm->N, psi, h_psi);
  cudaMalloc((void **) &d_psi, psi_size);
  cudaMemcpy( d_psi,
              h_psi,
              psi_size,
              cudaMemcpyHostToDevice );

  double *h_posterior = 0;
  double *d_posterior = 0;
  size_t posterior_size = T * phmm.N * sizeof(*h_posterior);
  h_posterior = (double *) malloc (posterior_size);
  dmatrix_to_arr(T, phmm->N, posterior, h_posterior);
  cudaMalloc((void **) &d_posterior, posterior_size);
  cudaMemcpy( d_posterior,
              h_posterior,
              posterior_size,
              cudaMemcpyHostToDevice );

  double *d_phmm_pi = 0;
  size_t phmm_pi_size = phmm->N * sizeof(*d_phmm_pi);
  cudaMalloc((void **) &d_phmm_pi, phmm_pi_size);
  cudaMemcpy( d_phmm_pi,
              phmm->pi,
              phmm_pi_size,
              cudaMemcpyHostToDevice );

  double *d_g = 0;
  size_t g_size = T * sizeof(*d_g);
  cudaMalloc((void **) &d_g, g_size);
  cudaMemcpy( d_g,
              g,
              g_size,
              cudaMemcpyHostToDevice );

  double *d_vprob = 0;
  size_t vprob_size = T * sizeof(*vprob_g);
  cudaMalloc((void **) &d_vprob, vprob_size);
  cudaMemcpy( d_vprob,
              vprob,
              vprob_size,
              cudaMemcpyHostToDevice );

  double *d_pprob = 0;
  size_t pprob_size = P * sizeof(*pprob_g);
  cudaMalloc((void **) &d_pprob, pprob_size);
  cudaMemcpy( d_pprob,
              pprob,
              pprob_size,
              cudaMemcpyHostToDevice );

  int *d_q = 0;
  size_t q_size = T * sizeof(*d_q);
  cudaMalloc((void**) &d_q, q_size);
  cudaMemcpy( d_q,
              q,
              q_size,
              cudaMemcpyHostToDevice );

  int phmm_N = phmm->N;
  int phmm_M = phmm->M;
  int phmm_inactive = phmm->inactive;

  double *h_phmm_log_A_matrix = 0;
  double *d_phmm_log_A_matrix = 0;
  size_t phmm_log_A_matrix_size = phmm->N * phmm->N * sizeof(*h_phmm_log_A_matrix);
  h_phmm_log_A_matrix = (double *) malloc (phmm_log_A_matrix_size);
  gsl_matrix_to_arr(phmm->N, phmm->N, phmm->log_A_matrix, d_phmm_log_A_matrix);
  cudaMalloc((void **) &d_phmm_log_A_matrix, phmm_log_A_matrix_size);
  cudaMemcpy( d_phmm_log_A_matrix,
              h_phmm_log_A_matrix,
              phmm_log_A_matrix_size,
              cudaMemcpyHostToDevice );

  double *d_phmm_thresholds = 0;
  size_t phmm_thresholds_size = phmm->M * sizeof(*d_phmm_thresholds);
  cudaMalloc((void **) &d_phmm_thresholds, phmm_thresholds_size);
  cudaMemcpy( d_phmm_thresholds,
              phmm->thresholds,
              phmm_thresholds_size,
              cudaMemcpyHostToDevice );
  

  viterbi_kernel<<<(P + TPB - 1)/TPB, TPB>>>( T,
                                              P,
                                              TF,
                                              phmm_N,
                                              phmm_M,
                                              phmm_inactive,
                                              d_TFlist,
                                              d_indexList,
                                              d_lengthList,
                                              d_motifList,
                                              q,
                                              d_peakPos,
                                              d_psi,
                                              d_phmm_pi,
                                              d_g,
                                              d_alpha,
                                              d_beta,
                                              d_gamma,
                                              d_delta,
                                              d_logprobf,
                                              d_vprobf,
                                              d_pprob,
                                              d_posterior,
                                              d_emission_matrix,
                                              d_pwd_matrix );
  cudaMemcpy( h_posterior,
              d_posterior,
              posterior_size,
              cudaMemcpyDeviceToHost );
  arr_to_dmatrix(T, phmm->N, h_posterior, posterior);

  cudaMemcpy( vprob,
              d_vprob,
              vprob_size,
              cudaMemcpyDeviceToHost );

  cudaMemcpy( q,
              d_q,
              q_size,
              cudaMemcpyDeviceToHost );
    
  free(TFlist);
  free(indexList);
  free(lengthList);
  free(motifList);
  free(h_emission_matrix);
  free(h_pwd_matrix);
  free(h_alpha);
  free(h_beta);
  free(h_gamma);
  free(h_delta);
  free(h_psi);
  free(h_phmm_log_A_matrix);

  cudaFree(d_TFlist);
  cudaFree(d_indexList);
  cudaFree(d_lengthList);
  cudaFree(d_motifList);
  cudaFree(d_emission_matrix);
  cudaFree(d_pwd_matrix);
  cudaFree(d_peakPos);
  cudaFree(d_logprobf);
  cudaFree(d_alpha);
  cudaFree(d_beta);
  cudaFree(d_gamma);
  cudaFree(d_delta);
  cudaFree(d_psi);
  cudaFree(d_posterior);
  cudafree(d_phmm_pi);
  cudaFree(d_g);
  cudaFree(d_vprob);
  cudaFree(d_pprob);
  cudaFree(d_q);
  cudaFree(d_phmm_log_A_matrix);
  cudaFree(d_phmm_thresholds);                              

}

/* Get the hidden states sequence using Viterbi algorithm */
void Viterbi(HMM *phmm, int T, double *g, double **alpha, double **beta,
             double	**gamma, double  *logprobf, double **delta,
             int **psi, int *q, double *vprob, double *pprob,
             double **posterior, int P, int *peakPos,
             gsl_matrix *emission_matrix, gsl_matrix *pwm_matrix)
{
  int thread_id, nloops;
  int  i, j, k, m, x, y, TF;   /* state indices */
  int  t = 0;      /* time index */
  int  maxvalind;
  int  l = 0;
  int *TFlist, *lengthList;
  int *motifList, *indexList;
  int nonInf;
  double  maxval, val;
  double plogprobinit, plogprobfinal;
  double temp;
  TF = 0;
  TFlist = ivector(phmm->M * (phmm->inactive+1));
  indexList = ivector(phmm->M * (phmm->inactive+1));
  for (j = 0; j < phmm->M; j++){
    TF += phmm->D[j];
    TFlist[j * (phmm->inactive+1)] = TF - 1;
    indexList[j * (phmm->inactive+1)] = j;
    if (phmm->inactive == 1){
      TF += phmm->D[j];
      TFlist[j * (phmm->inactive+1) + 1] = TF - 1;
      indexList[j * (phmm->inactive+1) + 1] = j;
    }
  }
  lengthList = ivector(phmm->M * (phmm->inactive+1));
  if (phmm->inactive == 1){
    for (j = 0; j < phmm->M; j++){
      lengthList[j*2] = phmm->D[j];
      lengthList[j*2+1] = phmm->D[j];
    }
  }
  else lengthList = phmm->D;

  motifList = ivector(phmm->N);
  TF = 0;
  for (j = 0; j < phmm->M; j++) {
    for (i = TF; i < TF + phmm->D[j]; i++) {
      motifList[i] = (phmm->inactive+1)*j;
    }
    TF += phmm->D[j];
    if (phmm->inactive == 1) {
      for (i = TF; i < TF + phmm->D[j]; i++) {
        motifList[i] = (phmm->inactive+1)*j+1;
      }
      TF += phmm->D[j];
    }
  }
  for (i = TF; i < TF + phmm->extraState; i++) {
    motifList[i] = i;
  }
#pragma omp parallel num_threads(THREAD_NUM) \
  private(thread_id, nloops, val, maxval, maxvalind, t, j, i, temp, x, y, nonInf)
  {
    nloops = 0;
#pragma omp for
  for (k = 0; k < P; k++){
    ++nloops;
    /* 1. Initialization  */
    for (i = 0; i < phmm->N; i++) {
      delta[peakPos[k]-1][i] = log(phmm->pi[i]) +
                               gsl_matrix_get(emission_matrix, i, peakPos[k]-1);
      psi[peakPos[k]-1][i] = phmm->N;
      posterior[peakPos[k]-1][i] = alpha[i][peakPos[k]-1] +
                                   beta[i][peakPos[k]-1] - logprobf[k];
    }
    for (x = 0; x < phmm->M * (phmm->inactive+1); x++){
        posterior[peakPos[k]-1][TFlist[x]] = VITTINY;
    }
    /* 2. Recursion */
    for (t = peakPos[k]; t < peakPos[k+1] - 1; t++) {
      for (j = 0; j < phmm->N; j++) {
        maxval = VITTINY;
        maxvalind = phmm->N-1;
        for (i = 0; i < phmm->N; i++) {
          val = delta[t-1][i] + gsl_matrix_get(phmm->log_A_matrix, i, j);
          if (val > maxval) {
            maxval = val;
            maxvalind = i;
          }
        }
        temp = gsl_matrix_get(emission_matrix, j, t);
        if (j < TF){
          if (gsl_matrix_get(pwm_matrix,indexList[motifList[j]],t) < (phmm->thresholds[indexList[motifList[j]]]-0.1)){
            temp = VITTINY;
              //(phmm->thresholds[(int)floor((double)motifList[j]/(phmm->inactive+1))]
          }
        }
        delta[t][j] = maxval + temp;
        psi[t][j] = maxvalind;

        posterior[t][j] = alpha[j][t] + beta[j][t] - logprobf[k];
        /* treat t as the last position of a motif */
        for (x = 0; x < phmm->M * (phmm->inactive+1); x++){
          if (j == TFlist[x]){
            if (t <= (peakPos[k] + lengthList[x])){
              posterior[t][j] = log(0.0);
            }
            /* check pwm threshold*/
            //else if (gsl_matrix_get(pwm_matrix,indexList[motifList[j]],t) < (phmm->thresholds[indexList[motifList[j]]]-0.1)){
              //for (y = t - lengthList[x] + 1; y <= t; y++) posterior[y][j] = log(0.0);
            //}
            else {
              temp=0.0;
              nonInf = 0;
              for (i = 0; i < lengthList[x]; i++) {
                if ((alpha[j-i][t-i] + beta[j-i][t-i] - logprobf[k]) != -INFINITY){
                  temp += (alpha[j-i][t-i] + beta[j-i][t-i] - logprobf[k]);
                  nonInf += 1;
                }
              }
              if (nonInf == 0) temp = VITTINY;
              for (y = t - lengthList[x] + 1; y <= t; y++) posterior[y][j] = temp;
            }

            break;
          }
        }
      }
    }

    /* 3. Termination */
    pprob[k] = -VITHUGE;
    q[peakPos[k+1]-2] = 1;
    for (i = 0; i < phmm->N; i++) {
      if (delta[peakPos[k+1]-2][i] > pprob[k]) {
        pprob[k] = delta[peakPos[k+1]-2][i];
        q[peakPos[k+1]-2] = i;
      }
    }
    g[peakPos[k+1]-2] = gamma[q[peakPos[k+1]-2]][peakPos[k+1]-2];
    vprob[peakPos[k+1]-2] = pprob[k];
    /* 4. Path (state sequence) backtracking */
  	for (t = peakPos[k+1] - 3; t >= peakPos[k] - 1; t--){
	  	q[t] = psi[t+1][q[t+1]];
      vprob[t] = delta[t+1][q[t+1]];
      g[t] = gamma[q[t]][t];
    }
  }
  }

}


/* Motif-centric approach. This function will calculate marginal posterior probabilities
 * for all motif sites provided  */
int getPosterior_motif(FILE *fpIn, FILE *fpOut, int T, int *peakPos,
                       double **posterior, HMM *phmm, int *q, double *vprob)
{
  int start, end, TFstart, TFend, length, init, t, j, m, n;
  int old_start = -1;
  int i;
  int TF, maxTF, indexTF_end, state, pos, motif;
  int half;
  int ifProb;
  double prob;
  char chr[8];
  char chr_[8];
  int *lengthList = ivector(phmm->M * (phmm->inactive+1));
  int *motifList = ivector(phmm->N);
  indexTF_end = -1;
  if (phmm->inactive == 1){
    for (j = 0; j < phmm->M; j++){
      lengthList[j*2] = phmm->D[j];
      lengthList[j*2+1] = phmm->D[j];
      indexTF_end += phmm->D[j] * 2;
    }
  }
  else {
    lengthList = phmm->D;
    for (j = 0; j < phmm->M; j++){
      indexTF_end += phmm->D[j];
    }
  }

  TF = 0;
  for (j = 0; j < phmm->M; j++) {
    for (i = TF; i < TF + phmm->D[j]; i++) {
      motifList[i] = (phmm->inactive+1)*j;
    }
    TF += phmm->D[j];
    if (phmm->inactive == 1) {
      for (i = TF; i < TF + phmm->D[j]; i++) {
        motifList[i] = (phmm->inactive+1)*j+1;
      }
      TF += phmm->D[j];
    }
  }
  for (i = TF; i < TF + phmm->extraState; i++) {
    motifList[i] = i;
  }
  TF -= 1;
  i = -1;
  fprintf(stdout,"scanning motif sites and calculating posterior probabilities \n");
  while (fscanf(fpIn, "%s\t%d\t%d\t%s\t%d\t%d\t%d", chr, &start, &end, chr_, &TFstart, &TFend, &length) != EOF) {
    /* Skip repetitive regions */
    if (start != old_start) {
      i++;
    }
    if (TFstart != -1){
      if (length >= (TFend-TFstart) && TFend <= end) {
        init = peakPos[i] - 1;
        /*
        state = 100000;
        motif = 100000;
        // Get the state signed for the motif site from viterbi
        for (m = init + TFend - start - 1; m > init + TFstart - start - 1; m--) {
          if (motif > motifList[q[m]]) {
            motif = motifList[q[m]];
            state = q[m];
            pos = m;
          }
        }
        ifProb = -1;
        TF = -1;
        half = length / 2;
        t = init + TFstart - start - 1 + half; // middle position of motif site
        // If the assigned state is in one of the motif, check which motif it is
        if (state <= indexTF_end){
          maxTF = motif + 1;
          prob = posterior[pos][state];
          ifProb = 1;
        }
        // If the model doesn't have motif information (Boyle method), get the
         // state and posterior probability of middle position of motif site
        else if (indexTF_end != -1){
          maxTF = q[t];
          prob = -1000000000.0;
        }
        // If the state is not in a motif, get the state and posterior probability
         // of middle position of motif site
        else{
          maxTF = q[t];
          prob = posterior[t][maxTF];
        }
        */

        /* Print posterior probabilities of being every active and inactive
         * motif states and generic footprints states for all motif sites */
        fprintf(fpOut,"%s\t%d\t%d", chr, TFstart, TFend);
        TF = -1;
        for (j = 0; j < phmm->M * (phmm->inactive+1); j++){
          TF += lengthList[j];
          prob = -INFINITY;
          for (m = 0; m <= length+MIN(lengthList[j]-1,end-TFend); m ++) prob = MAX(prob, posterior[(init + TFstart - start + m - 1)][TF]);
          fprintf(fpOut,"\t%e", prob);
        }
        /* posterior probability of being a generic footprint */
        prob = 0.0;
        for (m = 0; m < length; m ++) prob += posterior[(init + TFstart - start + m)][phmm->N-4];
        fprintf(fpOut,"\t%e", prob/length);
        prob = 0.0;
        for (m = 0; m < length; m ++) prob += posterior[(init + TFstart - start + m)][phmm->N-3];
        fprintf(fpOut,"\t%e", prob/length);
        fprintf(fpOut,"\n");
      }
    }
    old_start = start;
	}
	return indexTF_end;
}

/* Get all binding sites predictions from viterbi */
void getPosterior_all(FILE *fpIn, FILE *fpOut, int T, int *q,
                         int *peakPos, double **posterior, HMM *phmm)
{
  int start, end, TFstart, TFend, length, init, t, i, j;
  int stateStart, stateEnd, dataStart, dataEnd, stateLength;
  int old_start = -1;
  int TF, maxTF;
  double prob;
  char chr[20];
  fprintf(stdout,"scanning peak file and calculating posterior probabilities for all positions\n");

  int * stateList = ivector(phmm->N);
  TF = 0;
  for (j = 0; j < phmm->M; j++){
    for (i = TF; i < TF + phmm->D[j]; i++) {
      stateList[i] = j * (phmm->inactive + 1);
    }
    TF += phmm->D[j];

    if (phmm->inactive == 1){
      for (i = TF; i < TF + phmm->D[j]; i++) {
        stateList[i] = j * (phmm->inactive + 1) + 1;
      }
      TF += phmm->D[j];
    }
  }
  TF -= 1;
  for (j = phmm->N - phmm->extraState; j < phmm->N; j++){
    stateList[j] = j;
  }
  i = -1;
  while (fscanf(fpIn, "%s\t%d\t%d", chr, &start, &end) != EOF) {
    if (start != old_start){
      i++;
      dataStart = peakPos[i] - 1;
      dataEnd = peakPos[i+1] - 2;
      stateStart = 0;
      stateEnd = stateStart;
      t = dataStart;
      if (posterior[t][q[t]] != -INFINITY) {
        prob = posterior[t][q[t]];
        stateLength = 1;
      }
      else {
        prob = 0;
        stateLength = 0;
      }
      for (t = dataStart+1; t <= dataEnd; t++){
        if (stateList[q[t]] == stateList[q[t-1]]){
          stateEnd ++;
          if (posterior[t][q[t]] != -INFINITY){
            prob += posterior[t][q[t]];
            stateLength ++;
          }
          maxTF = stateList[q[t]] + 1;
          if (t == dataEnd)
            fprintf(fpOut,"%s\t%d\t%d\t%d\t%e\t%e\n", chr, start + stateStart,
                    start + stateEnd, maxTF, prob/stateLength, prob/stateLength);
        }
        else {
          maxTF = stateList[q[t-1]] + 1;
          if (maxTF <= phmm->M * (phmm->inactive+1) && phmm->M != 0) {
            if (maxTF % 2 == 0) {
              fprintf(fpOut, "%s\t%d\t%d\t%d\t%e\t%e\n", chr, start + stateStart,
                      start + stateEnd, maxTF, posterior[t - 1][q[t - 1]],
                      posterior[t - 1][q[t - 1] - phmm->D[maxTF / 2 - 1]]);
            }
            else {
              fprintf(fpOut, "%s\t%d\t%d\t%d\t%e\t%e\n", chr, start + stateStart,
                      start + stateEnd, maxTF, posterior[t - 1][q[t - 1]],
                      posterior[t - 1][q[t - 1] + phmm->D[(maxTF - 1) / 2]]);
            }
          }
          else fprintf(fpOut,"%s\t%d\t%d\t%d\t%e\t%e\n", chr, start + stateStart,
                       start + stateEnd, maxTF, prob/stateLength, prob/stateLength);
          stateEnd ++;
          stateStart = stateEnd;
          if (posterior[t][q[t]] != -INFINITY) {
            prob = posterior[t][q[t]];
            stateLength = 1;
          }
          else {
            prob = 0;
            stateLength = 0;
          }
          if (t == dataEnd)
            fprintf(fpOut,"%s\t%d\t%d\t%d\t%e\t%e\n", chr, start + stateStart,
                    start + stateEnd, stateList[q[t]] + 1, prob/stateLength,
                    prob/stateLength);
        }
      }
    }
    old_start = start;
  }
}


