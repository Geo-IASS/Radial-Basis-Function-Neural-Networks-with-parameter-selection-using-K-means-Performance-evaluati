/*
 * LU_Inverse_Matrix.h
 *
 *  Created on: 12 Jul 2016
 *      Author: luism_000
 */

#ifndef LU_INVERSE_MATRIX_H_
#define LU_INVERSE_MATRIX_H_

void inverse(double**,int);
void ludcmp(double**, int, int*, double*);
void lubksb(double**, int, int*, double*);
void  free_matrix(double**,int,int,int,int);
void  free_vector(double*,int,int);
double **matrix(double init_x,int dim_x, double init_y, int dim_y);
double *vector(int init, int size);

#endif /* LU_INVERSE_MATRIX_H_ */
