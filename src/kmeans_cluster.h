/*
 * kmeans_cluster.h
 *
 *  Created on: 4 Jul 2016
 *      Author: luism_000
 */

#ifndef KMEANS_CLUSTER_H_
#define KMEANS_CLUSTER_H_

#include "rbf_train.h"

void rand_vals(double *values, int n, int factor);
void vector_norm(double *vector, size_t vector_size, double *result);
tuple* kmeans_cluster(int F_size, double F[2][F_size], int K, int KMI);

#endif /* KMEANS_CLUSTER_H_ */
