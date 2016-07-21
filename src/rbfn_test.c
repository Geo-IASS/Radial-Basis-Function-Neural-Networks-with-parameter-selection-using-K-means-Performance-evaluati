/*
 * rbfn_test.c

 *
 *  Created on: 12 Jul 2016
 *      Author: luism_000
 */

#include <stdlib.h>
#include "rbf_train.h"
#include "rbf_kernel.h"
#include "rbfn_test.h"

double* rbfn_test(int F_size, double F[2][F_size], int K, triple *tpl){

	double **H = (double **)malloc(sizeof(double *) * K);
	if (H == NULL) {
		exit(EXIT_FAILURE);
	}

	double *Y = (double *)calloc(F_size,sizeof(double));
	if (Y == NULL) {
		exit(EXIT_FAILURE);
	}

	int i;
	int k;

	for(i=0; i<K; i++){
		H[i] = (double *)calloc(F_size,sizeof(double));
		if (H[i] == NULL) {
			exit(EXIT_FAILURE);
		}
	}

	for(i=0;i<K;i++){
		H[i] = (double *)rbf_kernel(F_size, F, tpl->MU, tpl->SIGMA, i);
	}

	for(i=0; i<F_size; i++){
		for(k=0; k<K; k++){
			Y[i] += H[k][i] * tpl->W[k];
		}
	}

	for(i=0; i<F_size; i++){
		if(Y[i] > 1){
			Y[i] = 1;
		}else if(Y[i] < 0){
			Y[i] = 0;
		}else if(Y[i] > 0.5){
			Y[i] = 1;
		}else{
			Y[i] = 0;
		}
	}

	//Free vector
	for(i=0; i<K; i++){
		free(H[i]);
	}

	free(H);

	return Y;
}
