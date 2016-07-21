/*
 * rbf_train.c
 *
 *  Created on: 14 Jun 2016
 *      Author: luism_000
 */

#include <stdlib.h>
#include "kmeans_cluster.h"
#include "LU_Inverse_Matrix.h"
#include "rbf_train.h"
#include "rbf_kernel.h"
#include "Flags.h"



tuple* new_tuple(int K){
	tuple* t = malloc(sizeof(tuple));

	double **MU_p;
	double ***SIGMA_p;

	MU_p = (double **)malloc(sizeof(double *) * K);
	SIGMA_p = (double ***)malloc(sizeof(double*) * K);

	if (MU_p == NULL) {
		perror("tuple memory allocation failure");
		exit(EXIT_FAILURE);
	}
	int i;
	for(i=0; i<K; i++){
		MU_p[i] = malloc(sizeof(double)*2);
		if (MU_p[i] == NULL) {
			perror("tuple memory allocation failure");
			exit(EXIT_FAILURE);
		}

		SIGMA_p[i] = (double **)malloc(sizeof(double*)*2);
		if (SIGMA_p[i] == NULL) {
			perror("tuple memory allocation failure");
			exit(EXIT_FAILURE);
		}
		SIGMA_p[i][0] = (double *)malloc(sizeof(double)*2);
		SIGMA_p[i][1] = (double *)malloc(sizeof(double)*2);

	}

	t->MU = MU_p;
	t->SIGMA = SIGMA_p;

	return t;
}

void free_tuple(tuple* t, int K){
	// Fields are transfered to triple
	//	int i;
	//	for(i=0; i<K; i++){
	//		free(t->MU[i]);
	//
	//		free(t->SIGMA[i][0]);
	//		free(t->SIGMA[i][1]);
	//
	//		free(t->SIGMA[i]);
	//
	//	}
	//	free(t->MU);
	//	free(t->SIGMA);
	free(t);
}

void free_triple(triple* t, int K){
	// Fields are transfered to triple
	int i;
	for(i=0; i<K; i++){
		free(t->MU[i]);

		free(t->SIGMA[i][0]);
		free(t->SIGMA[i][1]);

		free(t->SIGMA[i]);

	}
	free(t->MU);
	free(t->SIGMA);
	free(t->W);
	free(t);
}

triple* rbf_train(int F_size, double F[2][F_size], double C[F_size], int K, int KMI){

	//printf("RBF_TRAIN, call K-Means\n");
	tuple *ms = kmeans_cluster(F_size,F,K,KMI);

	//	printf("Value from tuple %lf\n",ms->MU[0][0]);
	//	printf("Value from tuple %lf\n",ms->MU[0][1]);
	//	printf("Value from tuple %lf\n",ms->MU[1][0]);
	//	printf("Value from tuple %lf\n",ms->MU[1][1]);
	//	printf("SIGMA[%i][0][1] = %lf\n",0,ms->SIGMA[0][0][1]);
	//	printf("SIGMA[%i][0][0] = %lf\n",1,ms->SIGMA[1][0][0]);

	double **H= (double **)malloc(sizeof(double *)*K);
	if((H==NULL)){
		exit(EXIT_FAILURE);
	}

	double **A = (double **)malloc(sizeof(double *)*K);
	if((A==NULL)){
		exit(EXIT_FAILURE);
	}

	double **A_M = (double **)malloc(sizeof(double *)*K);
	if((A==NULL)){
		exit(EXIT_FAILURE);
	}

	int i;
	int j;
	int k;

	for(k=0; k<K; k++){
		A_M[k] = (double *)calloc(F_size,sizeof(double));
		if((A_M[k]==NULL)){
			exit(EXIT_FAILURE);
		}
	}


	for(k=0;k<K;k++){
		H[k] = (double *)rbf_kernel(F_size, F, ms->MU, ms->SIGMA, k);
		A[k] = (double *)calloc(K,sizeof(double));
		//		printf("H[%i] = %lf\n",k,H[k][1]);
	}

	for(k=0; k<K; k++){
		for(j=0; j<K; j++){
			for(i=0; i<F_size; i++){
				A[k][j] += H[k][i]*H[j][i];
			}
		}
	}
	//
	//		printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A[0][0],A[0][1],A[0][2],A[0][3],A[0][4],A[0][5],A[0][6],A[0][7]);
	//		printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A[1][0],A[1][1],A[1][2],A[1][3],A[1][4],A[1][5],A[1][6],A[1][7]);
	//		printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A[2][0],A[2][1],A[2][2],A[2][3],A[2][4],A[2][5],A[2][6],A[2][7]);
	//		printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A[3][0],A[3][1],A[3][2],A[3][3],A[3][4],A[3][5],A[3][6],A[3][7]);
	//		printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A[4][0],A[4][1],A[4][2],A[4][3],A[4][4],A[4][5],A[4][6],A[4][7]);
	//		printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A[5][0],A[5][1],A[5][2],A[5][3],A[5][4],A[5][5],A[5][6],A[5][7]);
	//		printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A[6][0],A[6][1],A[6][2],A[6][3],A[6][4],A[6][5],A[6][6],A[6][7]);
	//		printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n\n\n\n",A[7][0],A[7][1],A[7][2],A[7][3],A[7][4],A[7][5],A[7][6],A[7][7]);


	inverse(A,K);

	//		printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A[0][0],A[0][1],A[0][2],A[0][3],A[0][4],A[0][5],A[0][6],A[0][7]);
	//		printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A[1][0],A[1][1],A[1][2],A[1][3],A[1][4],A[1][5],A[1][6],A[1][7]);
	//		printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A[2][0],A[2][1],A[2][2],A[2][3],A[2][4],A[2][5],A[2][6],A[2][7]);
	//		printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A[3][0],A[3][1],A[3][2],A[3][3],A[3][4],A[3][5],A[3][6],A[3][7]);
	//		printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A[4][0],A[4][1],A[4][2],A[4][3],A[4][4],A[4][5],A[4][6],A[4][7]);
	//		printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A[5][0],A[5][1],A[5][2],A[5][3],A[5][4],A[5][5],A[5][6],A[5][7]);
	//		printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A[6][0],A[6][1],A[6][2],A[6][3],A[6][4],A[6][5],A[6][6],A[6][7]);
	//		printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n\n",A[7][0],A[7][1],A[7][2],A[7][3],A[7][4],A[7][5],A[7][6],A[7][7]);

	for(i=0; i<K; i++){
		for(j=0; j<F_size; j++){
			for(k=0; k<K; k++){
				A_M[i][j] += A[k][i] * H[k][j];
			}
		}
	}

//	printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A_M[0][0],A_M[0][1],A_M[0][2],A_M[0][3],A_M[0][4],A_M[0][5],A_M[0][6],A_M[0][7]);
//	printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A_M[1][0],A_M[1][1],A_M[1][2],A_M[1][3],A_M[1][4],A_M[1][5],A_M[1][6],A_M[1][7]);
//	printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A_M[2][0],A_M[2][1],A_M[2][2],A_M[2][3],A_M[2][4],A_M[2][5],A_M[2][6],A_M[2][7]);
//	printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A_M[3][0],A_M[3][1],A_M[3][2],A_M[3][3],A_M[3][4],A_M[3][5],A_M[3][6],A_M[3][7]);
//	printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A_M[4][0],A_M[4][1],A_M[4][2],A_M[4][3],A_M[4][4],A_M[4][5],A_M[4][6],A_M[4][7]);
//	printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A_M[5][0],A_M[5][1],A_M[5][2],A_M[5][3],A_M[5][4],A_M[5][5],A_M[5][6],A_M[5][7]);
//	printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",A_M[6][0],A_M[6][1],A_M[6][2],A_M[6][3],A_M[6][4],A_M[6][5],A_M[6][6],A_M[6][7]);
//	printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n\n",A_M[7][0],A_M[7][1],A_M[7][2],A_M[7][3],A_M[7][4],A_M[7][5],A_M[7][6],A_M[7][7]);


	double *W = (double *)calloc(K,sizeof(double));
	if((W==NULL)){
		exit(EXIT_FAILURE);
	}

	for(i=0; i<K; i++){
		for(j=0; j<F_size; j++){
			for(k=0; k<K; k++){
				W[i] += A_M[i][j] * C[j];
				//printf(" W[%i] += A_M[%i][%i] = %lf * C[%i] = %lf\n",i,i,j,A_M[i][j],j,C[j]);
			}
		}
	}

	//printf("[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]\n",W[0],W[1],W[2],W[3],W[4],W[5],W[6],W[7]);

	triple *tpl = malloc(sizeof(triple));
	tpl->MU = ms->MU;
	tpl->SIGMA = ms->SIGMA;
	tpl->W = W;

	free_tuple(ms,K);

	//Free vectors
	for(i=0; i<K; i++){
		free(H[i]);
		free(A[i]);
		free(A_M[i]);
	}

	free(H);
	free(A);
	free(A_M);

	return tpl;
}
