/*
 * rbf_kernel.h
 *
 *  Created on: 14 Jun 2016
 *      Author: luism_000
 */


#ifndef RBF_KERNEL_H_
#define RBF_KERNEL_H_

double *rbf_kernel(int F_size, double F[2][F_size], double **MU, double ***SIGMA, int KMI);

#endif /* MVRND_H_ */
