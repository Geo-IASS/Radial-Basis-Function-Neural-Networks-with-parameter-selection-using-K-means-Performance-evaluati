/*

 * kmeans_cluster.c
 *
 *  Created on: 4 Jul 2016
 *      Author: luism_000
 */
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include "float.h"
#include "rbf_train.h"
#include "Flags.h"

//Creates array of n point between 0 and 1, multiplies a factor
//Ceil the results


void rand_vals(double *values, int n, int factor){
	int i;
	srand(time( NULL ) );
	for(i=0;i < n;i++){
		values[i] = ceil(((double)(rand()) / (double)((unsigned)RAND_MAX + 1))*factor);
		//printf("values[%i] = %lf \n",i,values[i]);
	}
}

void vector_norm(double *vector, int vector_size, double *result){
	int i;
	double aux = 0;

	for(i=0;i<vector_size;i++){
		aux += vector[i]*vector[i];
		//printf("Vector norm: vector[%i] = %lf\n",i,vector[i]);
	}

	*result = sqrt(aux);
}

tuple* kmeans_cluster(int F_size, double F[2][F_size], int K, int KMI){

	double values[K];

	//Reserve array space
	double **CENTS;
	double **DAL;

	CENTS = (double **)malloc(sizeof(double*)*K);

	int p;
	for(p=0; p<K; p++){
		CENTS[p] = (double *)malloc(sizeof(double)*2);
	}

	DAL = (double **)malloc(sizeof(double*)*F_size);
	for(p=0; p<F_size; p++){
		DAL[p] = (double *)malloc(sizeof(double)*(K+2));
	}

	//Start KMEANS

	rand_vals(values,K,(int)F_size);

	int i;
#if defined(LOOP_UNROLING)
	int i_roll = 0;
	for(i=0;i<K;i+=2){
		i_roll = i+1;
		int v1 = (int)values[i];
		int v2 = (int)values[i_roll];

		CENTS[i][0] = F[0][v1];
		CENTS[i][1] = F[1][v1];
		CENTS[i_roll][0] = F[0][v2];
		CENTS[i_roll][1] = F[1][v2];
	}
#else
	for(i=0;i<K;i++){
		CENTS[i][0] = F[0][(int)values[i]];
		CENTS[i][1] = F[1][(int)values[i]];
	}
#endif


	int n;
	for(n=0; n<KMI; n++){
		int j;
#if defined(LOOP_UNROL_AND_JAM_AND_DATA_LAYOUT)
		int K_plus_1 = K + 1;
		int i_plus_1 = 0;
		for(i=0;i<F_size;i+=2){

			i_plus_1 = i + 1;
			double F_0_i1 = F[0][i];
			double F_1_i1 = F[1][i];
			double F_0_i2 = F[0][i_plus_1];
			double F_1_i2 = F[1][i_plus_1];

			int j_plus_1 = 0;
			for(j=0;j<K;j+=2){
				j_plus_1 = j + 1;
				double norm;
				double aux[2] = {0};

				aux[0] = F_0_i1 - CENTS[j][0];
				aux[1] = F_1_i1 - CENTS[j][1];
				vector_norm(aux,2,&norm);

				DAL[i][j] = norm;

				aux[0] = F_0_i1 - CENTS[j_plus_1][0];
				aux[1] = F_1_i1 - CENTS[j_plus_1][1];
				vector_norm(aux,2,&norm);

				DAL[i][j_plus_1] = norm;

				aux[0] = F_0_i2 - CENTS[j][0];
				aux[1] = F_1_i2 - CENTS[j][1];
				vector_norm(aux,2,&norm);

				DAL[i_plus_1][j] = norm;

				aux[0] = F_0_i2 - CENTS[j_plus_1][0];
				aux[1] = F_1_i2 - CENTS[j_plus_1][1];
				vector_norm(aux,2,&norm);

				DAL[i_plus_1][j_plus_1] = norm;

			}

			double distance1 = DBL_MAX;
			double distance2 = DBL_MAX;
			int index1;
			int index2;

			for(j=0;j<K;j++){
				if(DAL[i][j] < distance1){
					distance1 = DAL[i][j];
					index1 = j;
				}
				if(DAL[i_plus_1][j] < distance2){
					distance2 = DAL[i_plus_1][j];
					index2 = j;
				}
			}
			DAL[i][K] = index1;
			DAL[i][K_plus_1] = distance1;

			DAL[i_plus_1][K] = index2;
			DAL[i_plus_1][K_plus_1] = distance2;
		}

		i_plus_1 = 0;
		for(i=0;i<K;i+=2){
			i_plus_1 = i + 1;
			double aux_mean_1_1 = 0;
			double aux_mean_1_2 = 0;
			int aux_size1 = 0;

			double aux_mean_2_1 = 0;
			double aux_mean_2_2 = 0;
			int aux_size2 = 0;

			int m;
			for(m=0; m<F_size; m++){
				if((int)DAL[m][K] == i){
					aux_mean_1_1 += F[0][m];
					aux_mean_1_2 += F[1][m];
					aux_size1 ++;
				}

				if((int)DAL[m][K] == i+1){
					aux_mean_2_1 += F[0][m];
					aux_mean_2_2 += F[1][m];
					aux_size2 ++;
				}
			}

			CENTS[i][0] = aux_mean_1_1/aux_size1;
			CENTS[i][1] = aux_mean_1_2/aux_size1;

			CENTS[i_plus_1][0] = aux_mean_2_1/aux_size2;
			CENTS[i_plus_1][1] = aux_mean_2_2/aux_size2;

			int y;
			int y_plus_1;
			y_plus_1 = 0;
			for(y=0; y<K; y+=2){
				y_plus_1 = y +1;
				if(isnan(CENTS[y][0])){
					printf("Is NaN on KMI:%i, y:%i, CENTS:%lf \n",n,y,CENTS[y][0]);
					CENTS[y][0] = F[0][rand() % ((K_plus_1-y)+y)];
				}
				if(isnan(CENTS[y][1])){
					printf("Is NaN on KMI:%i, k:%i, CENTS:%lf \n",n,y,CENTS[y][1]);
					CENTS[y][1] = F[0][rand() % ((K_plus_1-y)+y)];
				}
				if(isnan(CENTS[y_plus_1][0])){
					printf("Is NaN on KMI:%i, y:%i, CENTS:%lf \n",n,y_plus_1,CENTS[y_plus_1][0]);
					CENTS[y_plus_1][0] = F[0][rand() % ((K_plus_1-y_plus_1)+y_plus_1)];
				}
				if(isnan(CENTS[y_plus_1][1])){
					printf("Is NaN on KMI:%i, k:%i, CENTS:%lf \n",n,y,CENTS[y_plus_1][1]);
					CENTS[y_plus_1][1] = F[0][rand() % ((K_plus_1-y_plus_1)+y_plus_1)];
				}
			}
		}

#elif defined(LOOP_UNROLING_AND_JAM)
		int i_plus_1 = 0;
		for(i=0;i<F_size;i+=2){
			i_plus_1 = i +1;
			double F_0_i1 = F[0][i];
			double F_1_i1 = F[1][i];
			double F_0_i2 = F[0][i_plus_1];
			double F_1_i2 = F[1][i_plus_1];

			for(j=0;j<K;j++){
				double norm;
				double aux[2] = {0};

				aux[0] = F_0_i1 - CENTS[j][0];
				aux[1] = F_1_i1 - CENTS[j][1];
				vector_norm(aux,2,&norm);

				DAL[i][j] = norm;

				aux[0] = F_0_i2 - CENTS[j][0];
				aux[1] = F_1_i2 - CENTS[j][1];
				vector_norm(aux,2,&norm);

				DAL[i_plus_1][j] = norm;

			}

			double distance1 = DBL_MAX;
			double distance2 = DBL_MAX;
			int index1;
			int index2;

			for(j=0;j<K;j++){
				if(DAL[i][j] < distance1){
					distance1 = DAL[i][j];
					index1 = j;
				}
				if(DAL[i_plus_1][j] < distance2){
					distance2 = DAL[i_plus_1][j];
					index2 = j;
				}
			}
			DAL[i][K] = index1;
			DAL[i][K+1] = distance1;

			DAL[i+1][K] = index2;
			DAL[i+1][K+1] = distance2;
		}

		for(i=0;i<K;i+=2){

			double aux_mean_1_1 = 0;
			double aux_mean_1_2 = 0;
			int aux_size1 = 0;

			double aux_mean_2_1 = 0;
			double aux_mean_2_2 = 0;
			int aux_size2 = 0;

			int m;
			for(m=0; m<F_size; m++){
				if((int)DAL[m][K] == i){
					aux_mean_1_1 += F[0][m];
					aux_mean_1_2 += F[1][m];
					aux_size1 ++;
				}

				if((int)DAL[m][K] == i+1){
					aux_mean_2_1 += F[0][m];
					aux_mean_2_2 += F[1][m];
					aux_size2 ++;
				}
			}

			CENTS[i][0] = aux_mean_1_1/aux_size1;
			CENTS[i][1] = aux_mean_1_2/aux_size1;

			CENTS[i+1][0] = aux_mean_2_1/aux_size2;
			CENTS[i+1][1] = aux_mean_2_2/aux_size2;

			int y;
			for(y=0; y<K; y++){
				if(isnan(CENTS[y][0])){
					printf("Is NaN on KMI:%i, y:%i, CENTS:%lf \n",n,y,CENTS[y][0]);
					CENTS[y][0] = F[0][rand() % ((K+1-y)+y)];
				}
				if(isnan(CENTS[y][1])){
					printf("Is NaN on KMI:%i, k:%i, CENTS:%lf \n",n,y,CENTS[y][1]);
					CENTS[y][1] = F[0][rand() % ((K+1-y)+y)];
				}
			}
		}
#elif defined(LOOP_UNROLING)
		int i_plus_1 = 0;
		for(i=0;i<F_size;i+=2){
			i_plus_1 = i + 1;
			double F_0_i1 = F[0][i];
			double F_1_i1 = F[1][i];
			double F_0_i2 = F[0][i_plus_1];
			double F_1_i2 = F[1][i_plus_1];

			for(j=0;j<K;j++){
				double norm;
				double aux[2] = {0};
				aux[0] = F_0_i1 - CENTS[j][0];
				aux[1] = F_1_i1 - CENTS[j][1];
				vector_norm(aux,2,&norm);
				DAL[i][j] = norm;
			}

			for(j=0;j<K;j++){
				double norm;
				double aux[2] = {0};
				aux[0] = F_0_i2 - CENTS[j][0];
				aux[1] = F_1_i2 - CENTS[j][1];
				vector_norm(aux,2,&norm);

				DAL[i_plus_1][j] = norm;

			}

			double distance = DBL_MAX;
			int index;

			for(j=0;j<K;j++){
				if(DAL[i][j] < distance){
					distance = DAL[i][j];
					index = j;
				}
			}
			DAL[i][K] = index;
			DAL[i][K+1] = distance;

			distance = DBL_MAX;
			index = 0;

			for(j=0;j<K;j++){
				if(DAL[i_plus_1][j] < distance){
					distance = DAL[i_plus_1][j];
					index = j;
				}
			}
			DAL[i_plus_1][K] = index;
			DAL[i_plus_1][K+1] = distance;
		}

		i_plus_1 = 0;
		for(i=0;i<K;i+=2){
			i_plus_1 = i + 1;
			double aux_mean_1_1 = 0;
			double aux_mean_1_2 = 0;
			int aux_size1 = 0;

			double aux_mean_2_1 = 0;
			double aux_mean_2_2 = 0;
			int aux_size2 = 0;

			int m;
			for(m=0; m<F_size; m++){
				if((int)DAL[m][K] == i){
					aux_mean_1_1 += F[0][m];
					aux_mean_1_2 += F[1][m];
					aux_size1 ++;
				}
			}

			CENTS[i][0] = aux_mean_1_1/aux_size1;
			CENTS[i][1] = aux_mean_1_2/aux_size1;

			for(m=0; m<F_size; m++){
				if((int)DAL[m][K] == i+1){
					aux_mean_2_1 += F[0][m];
					aux_mean_2_2 += F[1][m];
					aux_size2 ++;
				}
			}

			CENTS[i_plus_1][0] = aux_mean_2_1/aux_size2;
			CENTS[i_plus_1][1] = aux_mean_2_2/aux_size2;

			int y;
			for(y=0; y<K; y++){
				if(isnan(CENTS[y][0])){
					printf("Is NaN on KMI:%i, y:%i, CENTS:%lf \n",n,y,CENTS[y][0]);
					CENTS[y][0] = F[0][rand() % ((K+1-y)+y)];
				}
				if(isnan(CENTS[y][1])){
					printf("Is NaN on KMI:%i, k:%i, CENTS:%lf \n",n,y,CENTS[y][1]);
					CENTS[y][1] = F[0][rand() % ((K+1-y)+y)];
				}
			}
		}
#else
		for(i=0;i<F_size;i++){
			int j;
			for(j=0;j<K;j++){
				double norm;
				double aux[2] = {0};
				aux[0]=F[0][i] - CENTS[j][0];
				aux[1]=F[1][i] - CENTS[j][1];

				vector_norm(aux,2,&norm);

				DAL[i][j] = norm;

			}

			double distance = DBL_MAX;
			int index;

			for(j=0;j<K;j++){
				if(DAL[i][j] < distance){
					distance = DAL[i][j];
					index = j;
				}
			}
			DAL[i][K] = index;
			DAL[i][K+1] = distance;
		}

		for(i=0;i<K;i++){

			double aux_mean_1 = 0;
			double aux_mean_2 = 0;
			int aux_size = 0;

			int m;
			for(m=0; m<F_size; m++){
				if((int)DAL[m][K] == i){
					aux_mean_1 += F[0][m];
					aux_mean_2 += F[1][m];
					aux_size ++;
				}
			}

			CENTS[i][0] = aux_mean_1/aux_size;
			CENTS[i][1] = aux_mean_2/aux_size;

			int y;
			for(y=0; y<K; y++){
				if(isnan(CENTS[y][0])){
					printf("Is NaN on KMI:%i, y:%i, CENTS:%lf \n",n,y,CENTS[y][0]);
					CENTS[y][0] = F[0][rand() % ((K+1-y)+y)];
				}
				if(isnan(CENTS[y][1])){
					printf("Is NaN on KMI:%i, k:%i, CENTS:%lf \n",n,y,CENTS[y][1]);
					CENTS[y][1] = F[0][rand() % ((K+1-y)+y)];
				}
			}
		}
#endif

	}

	//n ends here

	tuple* result = new_tuple(K);

	if(result == NULL){
		perror("tuple is null");
		exit(EXIT_FAILURE);
	}

#if defined(LOOP_UNROLING_AND_JAM)
	int i_plus_1 = 0;
	for(i=0; i<K; i+=2){
		i_plus_1 = i + 1;
		double mean_1_1 = 0;
		double mean_1_2 = 0;
		double std_1_1 = 0;
		double std_1_2 = 0;
		int counter1 = 0;
		double mean_2_1 = 0;
		double mean_2_2 = 0;
		double std_2_1 = 0;
		double std_2_2 = 0;
		int counter2 = 0;

		int j;
		for(j=0; j<F_size; j++){
			if(DAL[j][K] == i){
				counter1 ++;
				mean_1_1 += F[0][j];
				mean_1_2 += F[1][j];
			}
			if(DAL[j][K] == i+1){
				counter2 ++;
				mean_2_1 += F[0][j];
				mean_2_2 += F[1][j];
			}
		}

		mean_1_1 = mean_1_1/counter1;
		mean_1_2 = mean_1_2/counter1;

		mean_2_1 = mean_2_1/counter2;
		mean_2_2 = mean_2_2/counter2;

		result->MU[i][0] = mean_1_1;
		result->MU[i][1] = mean_1_2;

		result->MU[i_plus_1][0] = mean_2_1;
		result->MU[i_plus_1][1] = mean_2_2;

		counter1 = 0;
		counter2 = 0;
		for(j=0; j<F_size; j++){
			double f_0_j = F[0][j];
			double f_1_j = F[1][j];
			if(DAL[j][K] == i){

				double f_0_j_mean_1_1 = f_0_j-mean_1_1;
				f_0_j_mean_1_1 = f_0_j_mean_1_1 * f_0_j_mean_1_1;
				std_1_1 += abs(f_0_j_mean_1_1);

				double f_1_j_mean_1_2 = f_1_j-mean_1_2;
				f_0_j_mean_1_1 = f_1_j_mean_1_2 * f_1_j_mean_1_2;
				std_1_2 += abs(f_1_j_mean_1_2);
				counter1++;
			}else if(DAL[j][K] == i+1){
				double f_0_j_mean_2_1 = f_0_j-mean_2_1;
				f_0_j_mean_2_1 = f_0_j_mean_2_1 * f_0_j_mean_2_1;
				std_2_1 += abs(f_0_j_mean_2_1);

				double f_1_j_mean_2_2 = f_1_j-mean_2_2;
				f_1_j_mean_2_2 = f_1_j_mean_2_2 * f_1_j_mean_2_2;
				std_2_2 += abs(f_1_j_mean_2_2);
				counter2++;
			}
		}

		std_1_1 = sqrt(std_1_1/counter1);
		std_1_2 = sqrt(std_1_2/counter1);

		std_2_1 = sqrt(std_2_1/counter2);
		std_2_2 = sqrt(std_2_2/counter2);

		result->SIGMA[i][0][0] = std_1_1;
		result->SIGMA[i][0][1] = 0;
		result->SIGMA[i][1][0] = 0;
		result->SIGMA[i][1][1] = std_1_2;

		result->SIGMA[i_plus_1][0][0] = std_2_1;
		result->SIGMA[i_plus_1][0][1] = 0;
		result->SIGMA[i_plus_1][1][0] = 0;
		result->SIGMA[i_plus_1][1][1] = std_2_2;

	}

#elif defined(LOOP_UNROLING)
	int i_plus_1 = 0;
	for(i=0; i<K; i+=2){
		i_plus_1 = i + 1;
		double mean_1_1 = 0;
		double mean_1_2 = 0;
		double std_1_1 = 0;
		double std_1_2 = 0;
		int counter1 = 0;
		double mean_2_1 = 0;
		double mean_2_2 = 0;
		double std_2_1 = 0;
		double std_2_2 = 0;
		int counter2 = 0;

		int j;
		for(j=0; j<F_size; j++){
			if(DAL[j][K] == i){
				counter1 ++;
				mean_1_1 += F[0][j];
				mean_1_2 += F[1][j];
			}
		}

		mean_1_1 = mean_1_1/counter1;
		mean_1_2 = mean_1_2/counter1;

		result->MU[i][0] = mean_1_1;
		result->MU[i][1] = mean_1_2;

		counter1 = 0;
		for(j=0; j<F_size; j++){
			if(DAL[j][K] == i){
				std_1_1 += abs((F[0][j]-mean_1_1)*(F[0][j]-mean_1_1));
				std_1_2 += abs((F[1][j]-mean_1_2)*(F[1][j]-mean_1_2));
				counter1++;
			}
		}

		std_1_1 = sqrt(std_1_1/counter1);
		std_1_2 = sqrt(std_1_2/counter1);

		result->SIGMA[i][0][0] = std_1_1;
		result->SIGMA[i][0][1] = 0;
		result->SIGMA[i][1][0] = 0;
		result->SIGMA[i][1][1] = std_1_2;

		for(j=0; j<F_size; j++){
			if(DAL[j][K] == i_plus_1){
				counter2 ++;
				mean_2_1 += F[0][j];
				mean_2_2 += F[1][j];
			}
		}

		mean_2_1 = mean_2_1/counter2;
		mean_2_2 = mean_2_2/counter2;

		result->MU[i_plus_1][0] = mean_2_1;
		result->MU[i_plus_1][1] = mean_2_2;

		counter2 = 0;
		for(j=0; j<F_size; j++){
			if(DAL[j][K] == i+1){
				std_2_1 += abs((F[0][j]-mean_2_1)*(F[0][j]-mean_2_1));
				std_2_2 += abs((F[1][j]-mean_2_2)*(F[1][j]-mean_2_2));
				counter2++;
			}
		}

		std_2_1 = sqrt(std_2_1/counter2);
		std_2_2 = sqrt(std_2_2/counter2);

		result->SIGMA[i_plus_1][0][0] = std_2_1;
		result->SIGMA[i_plus_1][0][1] = 0;
		result->SIGMA[i_plus_1][1][0] = 0;
		result->SIGMA[i_plus_1][1][1] = std_2_2;
	}

#else
	for(i=0; i<K; i++){
		double mean_1 = 0;
		double mean_2 = 0;
		double std_1 = 0;
		double std_2 = 0;
		int counter = 0;

		int j;
		for(j=0; j<F_size; j++){
			if(DAL[j][K] == i){
				counter ++;
				mean_1 += F[0][j];
				mean_2 += F[1][j];
			}
		}

		mean_1 = mean_1/counter;
		mean_2 = mean_2/counter;

		result->MU[i][0] = mean_1;
		result->MU[i][1] = mean_2;

		counter = 0;
		for(j=0; j<F_size; j++){
			if(DAL[j][K] == i){
				std_1 += abs((F[0][j]-mean_1)*(F[0][j]-mean_1));
				std_2 += abs((F[1][j]-mean_2)*(F[1][j]-mean_2));
				counter++;
			}
		}

		std_1 = sqrt(std_1/counter);
		std_2 = sqrt(std_2/counter);

		result->SIGMA[i][0][0] = std_1;
		result->SIGMA[i][0][1] = 0;
		result->SIGMA[i][1][0] = 0;
		result->SIGMA[i][1][1] = std_2;
	}

	//Free arrays
	for(p=0; p<K; p++){
		free(CENTS[p]);
	}
	free(CENTS);

	for(p=0; p<F_size; p++){
		free(DAL[p]);
	}
	free(DAL);
#endif

#ifdef CLOCK_K_MEANS_CLUSTER
	// Stop clock:
	stop_time();

	// Ticks elapsed:
	CORE_TICKS ticksElapsed = get_core_ticks();
	printf("==== Number of ticks elapsed for KMEANS CLUSTER: %d\n", ticksElapsed);
	kmeans_ticks += ticksElapsed;
	printf("==== Number of total ticks elapsed for KMEANS CLUSTER: %d\n", kmeans_ticks);

	// Time elapsed:
	timerepr timeElapsed = time_in_secs(ticksElapsed);
	printf("Elapsed time (s): %f\n", (float)timeElapsed);
	timeElapsed = time_in_msecs(ticksElapsed);
	printf("Elapsed time (ms): %f\n", (float)timeElapsed);
	timeElapsed = time_in_usecs(ticksElapsed);
	printf("Elapsed time (us): %f\n", (float)timeElapsed);
	timeElapsed = time_in_usecs(kmeans_ticks);
	printf("Total elapsed time (us): %f\n", (float)timeElapsed);
#endif

	return result;
}


