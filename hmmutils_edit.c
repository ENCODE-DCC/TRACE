#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nrutil.h"
#include "hmm_edit.h"

static char rcsid[] = "$Id: hmmutils.c,v 1.4 1998/02/23 07:51:26 kanungo Exp kanungo $";
/*
void ReadHMM(FILE *fp, HMM *phmm)
{
  //printf("pass1");
	int i, j, k, n;
  double *PWM;
  double sum;

	fscanf(fp, "M= %d\n", &(phmm->M)); 

	fscanf(fp, "N= %d\n", &(phmm->N)); 

	fscanf(fp, "A:\n");
	phmm->A = (double **) dmatrix(1, phmm->N, 1, phmm->N);
	for (i = 1; i <= phmm->N; i++) { 
		for (j = 1; j <= phmm->N; j++) {
			fscanf(fp, "%lf", &(phmm->A[i][j])); 
		}
		fscanf(fp,"\n");
	}
  //printf("pass2");
	fscanf(fp, "PWM: n=%d\n", &(phmm->D));
	phmm->pwm = (double **) dmatrix(1, phmm->D, 1, 4);
  PWM = (double *) dvector(1, 4);
	for (j = 1; j <= phmm->D; j++) { 
    sum = 0.0;
		for (k = 1; k <= 4; k++) {
			fscanf(fp, "%lf", &(PWM[k])); 
      PWM[k] += 1.0;
      sum += PWM[k];
		}
    for (k = 1; k <= 4; k++) {
      phmm->pwm[j][k] = PWM[k]/sum;
    }
		fscanf(fp,"\n");
	}
  //printf("pass3");
	fscanf(fp, "pi:\n");
	phmm->pi = (double *) dvector(1, phmm->N);
  phmm->mu = (double *) dvector(1, (phmm->N - phmm->D + 1));
  phmm->sigma = (double *) dvector(1, (phmm->N - phmm->D + 1));
  phmm->CG = (double *) dvector(1, 4);
	for (i = 1; i <= phmm->N; i++) 
		fscanf(fp, "%lf\t", &(phmm->pi[i])); 
  fscanf(fp, "mu:\n");
  for (i = 1; i <= (phmm->N - phmm->D + 1); i++) {
    fscanf(fp, "%lf", &(phmm->mu[i])); 
  }
  fscanf(fp,"\n");
  fscanf(fp, "sigma:\n");
  for (i = 1; i <= (phmm->N - phmm->D + 1); i++) {
    fscanf(fp, "%lf", &(phmm->sigma[i])); 
  }

}
*/
void ReadOutHMM(FILE *fp, HMM *phmm)
{
  
	int i, j, k, n;
  
  double sum;

  fscanf(fp, "M= %d\n", &(phmm->M)); 
  fscanf(fp, "N= %d\n", &(phmm->N)); 
  phmm->K = 1 + phmm->M; /*number of types of data used. slop and pwm scrore here*/
             /*XXX: change to take input later*/ 
  fscanf(fp, "A:\n");
  phmm->A = (double **) dmatrix(1, phmm->N, 1, phmm->N);
  for (i = 1; i <= phmm->N; i++) { 
    for (j = 1; j <= phmm->N; j++) {
      fscanf(fp, "%lf", &(phmm->A[i][j])); 
    }
    fscanf(fp,"\n");
  }
  phmm->D = (int *) ivector(1, phmm->M);
  //double **PWM[phmm->M];
  phmm->pwm = (double ***)malloc(phmm->M*sizeof(double**));
  //double **phmm->pwm[phmm->M];
  for (i = 1; i <= phmm->M; i++) { 
    fscanf(fp, "PWM: n=%d\n", &(phmm->D[i]));
    //PWM[i] = (double **) dmatrix(1, phmm->D[i], 1, 4);
    phmm->pwm[i] = (double **) dmatrix(1, phmm->D[i], 1, 4);
    for (j = 1; j <= phmm->D[i]; j++) { 
      for (k = 1; k <= 4; k++) {
        fscanf(fp, "%lf", &(phmm->pwm[i][j][k])); 
      }
      fscanf(fp,"\n");
    }
  }

  //fprintf(stdout, "phmm->pwm: %lf \n", phmm->pwm[1][2][1]);  
  
  phmm->pi = (double *) dvector(1, phmm->N);
  phmm->mu = (double **) dmatrix(1, phmm->K, 1, phmm->N);
  //phmm->sigma = (double ***)malloc(phmm->K*sizeof(double**));
  phmm->sigma = (double **) dmatrix(1, phmm->K, 1, phmm->N);
  phmm->rho = (double **) dmatrix(1, phmm->K*(phmm->K-1)/2, 1, phmm->N);
  phmm->bg = (double *) dvector(1, 4);
  fscanf(fp, "pi:\n");
  for (i = 1; i <= phmm->N; i++){
    fscanf(fp, "%lf\t", &(phmm->pi[i])); 
  }
  fscanf(fp, "mu:\n");
  for (i = 1; i <= phmm->K; i++) {
    for (j = 1; j <= phmm->N; j++) {
      fscanf(fp, "%lf", &(phmm->mu[i][j])); 
    }
    fscanf(fp,"\n");
  }
  fscanf(fp, "sigma:\n");
  for (i = 1; i <= phmm->K; i++) {
    for (j = 1; j <= phmm->N ; j++) {
      fscanf(fp, "%lf", &(phmm->sigma[i][j])); 
    }
    fscanf(fp,"\n");
  }
  
  fscanf(fp, "rho:\n");
  for (i = 1; i <= phmm->K*(phmm->K-1)/2; i++) {
    for (j = 1; j <= phmm->N ; j++) {
      fscanf(fp, "%lf", &(phmm->rho[i][j])); 
    }
    fscanf(fp,"\n");
  }
}

/*
void InitHMMwithInput(HMM *phmm, int seed, double *gc)
{

  phmm->mu = (double *) dvector(1, (phmm->N - phmm->D + 1));
  phmm->sigma = (double *) dvector(1, (phmm->N - phmm->D + 1));
  phmm->CG = (double *) dvector(1, 4);
  hmmsetseed(seed);	
  for (int i = 1; i <= (phmm->N - phmm->D + 1); i++) {
    phmm->mu[i] = hmmgetrand(); 
    phmm->sigma[i] = hmmgetrand(); 
  }
  phmm->CG[2] = phmm->CG[3] = gc[2];
  phmm->CG[1] = phmm->CG[4] = gc[1];
}
*/
void FreeHMM(HMM *phmm)
{
  int i;
  free_dmatrix(phmm->A, 1, phmm->N, 1, phmm->N);
  for (i = 1; i <= phmm->M; i++) { 
    free_dmatrix(phmm->pwm[i], 1, phmm->D[i], 1, 4);
  }
  free_dvector(phmm->pi, 1, phmm->N);
  free_dmatrix(phmm->mu, 1, phmm->N, 1, phmm->M+1);
  free_dmatrix(phmm->sigma, 1, phmm->N, 1, phmm->M+1);
  free_dvector(phmm->bg, 1, 4);
}

/*
** InitHMM() This function initializes matrices A, B and vector pi with
**	random values. Not doing so can result in the BaumWelch behaving
**	quite weirdly.
*/ 
/*
void InitHMM(HMM *phmm, int N, int M, int D, int seed)
{
	int i, j, k;
	double sum;


	// initialize random number generator 


	hmmsetseed(seed);	

       	phmm->M = M;
 
        phmm->N = N;
        phmm->D = D;
 
        phmm->A = (double **) dmatrix(1, phmm->N, 1, phmm->N);

        for (i = 1; i <= phmm->N; i++) {
		sum = 0.0;
                for (j = 1; j <= phmm->N; j++) {
                        phmm->A[i][j] = hmmgetrand(); 
			sum += phmm->A[i][j];
		}
                for (j = 1; j <= phmm->N; j++) 
			 phmm->A[i][j] /= sum;
	}
 
        phmm->pwm = (double **) dmatrix(1, phmm->D, 1, 4);

        for (j = 1; j <= phmm->D; j++) {
		sum = 0.0;	
                for (k = 1; k <= 4; k++) {
                        phmm->B[j][k] = hmmgetrand();
			sum += phmm->B[j][k];
		}
                for (k = 1; k <= 4; k++) 
			phmm->B[j][k] /= sum;
	}
 
        phmm->pi = (double *) dvector(1, phmm->N);
	sum = 0.0;
        for (i = 1; i <= phmm->N; i++) {
                phmm->pi[i] = hmmgetrand(); 
		sum += phmm->pi[i];
	}
        for (i = 1; i <= phmm->N; i++) 
		phmm->pi[i] /= sum;
}
*/

/*
void CopyHMM(HMM *phmm1, HMM *phmm2)
{
  int i, j, k;
  phmm2->M = phmm1->M;
  phmm2->N = phmm1->N;
  phmm2->A = (double **) dmatrix(1, phmm2->N, 1, phmm2->N);
  for (i = 1; i <= phmm2->N; i++)
    for (j = 1; j <= phmm2->N; j++)
      phmm2->A[i][j] = phmm1->A[i][j];
  phmm2->pwm = (double **) dmatrix(1, phmm2->D, 1, 4);
  for (j = 1; j <= phmm2->D; j++)
    for (k = 1; k <=4; k++)
      phmm2->pwm[j][k] = phmm1->pwm[j][k];
 
   phmm2->pi = (double *) dvector(1, phmm2->N);
   for (i = 1; i <= phmm2->N; i++)
     phmm2->pi[i] = phmm1->pi[i]; 
 
}
*/
void PrintHMM(FILE *fp, HMM *phmm)
{
  int i, j, k;
  
	fprintf(fp, "M= %d\n", phmm->M); 
	fprintf(fp, "N= %d\n", phmm->N); 
 
	fprintf(fp, "A:\n");
  for (i = 1; i <= phmm->N; i++) {
    for (j = 1; j <= phmm->N; j++) {
      fprintf(fp, "%f ", phmm->A[i][j] );
    }
    fprintf(fp, "\n");
	}
  for (i = 1; i <= phmm->M; i++) {
	  fprintf(fp, "PWM: n=%d\n", (phmm->D[i]));
    for (j = 1; j <= phmm->D[i]; j++) {
      for (k = 1; k <= 4; k++){
        fprintf(fp, "%f ", phmm->pwm[i][j][k]);
	    }
      fprintf(fp, "\n");
    }
	}
 
	fprintf(fp, "pi:\n");
  for (i = 1; i <= phmm->N; i++) {
	  fprintf(fp, "%f ", phmm->pi[i]);
	}
  fprintf(fp, "\n");
  
  fprintf(fp, "mu:\n");
  for (i = 1; i <= phmm->K; i++) {
    for (j = 1; j <= phmm->N; j++) {
      fprintf(fp, "%lf ", phmm->mu[i][j]);
    }
    fprintf(fp, "\n");
  }
  fprintf(fp, "sigma:\n");
  for (i = 1; i <= phmm->K; i++) {
    for (j = 1; j <= phmm->N ; j++) {
      fprintf(fp, "%lf ", phmm->sigma[i][j]);
    }
    fprintf(fp, "\n");
  }
  
  fprintf(fp, "rho:\n");
  for (i = 1; i <= phmm->K*(phmm->K-1)/2; i++) {
    for (j = 1; j <= phmm->N ; j++) {
      fprintf(fp, "%lf ", phmm->rho[i][j]);
    }
  }
  fprintf(fp, "\n\n");
}

/*print a matrix of (size1 x size2)*/
void printMatrix(double **matrix, int size1, int size2){
  int i,j;
  for (i = 1; i <= size1; i++){
    for (j = 1; j <= size2; j++){
      fprintf(stdout, "%lf ", matrix[i][j]);   
    }
    fprintf(stdout, "\n");
  }
}

