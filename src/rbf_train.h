/*
 * rbf_train.h
 *
 *  Created on: 14 Jun 2016
 *      Author: luism_000
 */

#ifndef RBF_TRAIN_H_
#define RBF_TRAIN_H_


//Need to change to a dynamic allocation
typedef struct tuple_struct {
	//double MU[8][2];
	//double SIGMA[8][2][2];
	double **MU;
	double ***SIGMA;

} tuple;

typedef struct triple_struct {
//	double MU[8][2];
//	double SIGMA[8][2][2];
//	double W[8];

	double **MU;
	double ***SIGMA;
	double *W;

} triple;

tuple* new_tuple(int K);
void free_tuple(tuple* t, int K);

void free_triple(triple* t, int K);

triple* rbf_train(int F_size, double F[2][F_size], double C[F_size], int K, int KMI);

#endif /* RBF_TRAIN_H_ */
