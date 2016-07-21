/*
C inverse.c
C Source Code for "GRPP, A Scientific Programming Language
C Processor Designed for Lex and Yacc."
C Author: James Kent Blackburn
C Goddard Space Flight Center, Code 664.0, Greenbelt, MD. 20771
C Computers in Physics, Journal Section, Jan/Feb 1994

   Matrix Inversion using
   LU Decomposition from
   Numerical Recipes in C
   Chapter 2
 */

//http://bbs.dartmouth.edu/~fangq/MATH/download/source/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "MATRIX.H"
#include "LU_Inverse_Matrix.h"

#define   TINY 1.0e-20

double** matrix(double init_x,int dim_x, double init_y, int dim_y){
	double **matrix = (double **)malloc(sizeof(double *)*dim_x);
	int i;
	for(i=0; i<dim_x; i++){
		matrix[i] = (double *)malloc(sizeof(double) * dim_y);
	}

	return matrix;
}

double* vector(int init, int size){
	double *col = (double *)malloc(sizeof(double) * size);

	return col;
}

void  free_matrix(double **matrix,int init_x,int dim_x,int init_y,int dim_y){
	int i;
	for(i=0; i<dim_y; i++){
		free(matrix[i]);
	}
	free(matrix);
}
void  free_vector(double* vector,int init,int size){
	free(vector);
}

void inverse(double **mat, int dim)
{
	int i,j,*indx;
	double **y,d,*col;

	y = matrix(0,dim,0,dim);
	indx = (int *)malloc((unsigned)(dim*sizeof(int)));
	col = vector(0,dim-1);
	ludcmp(mat,dim,indx,&d);
	for (j=0;j<dim;j++)
	{
		for (i=0;i<dim;i++) col[i] = 0.0;
		col[j] = 1.0;
		lubksb(mat,dim,indx,col);
		for (i=0;i<dim;i++) y[i][j] = col[i];
	}
	for (i=0;i<dim;i++)
		for (j=0;j<dim;j++)
			mat[i][j] = y[i][j];
	free_matrix(y,0,dim,0,dim);
	free_vector(col,0,dim);
	free(indx);
}

void ludcmp(double **a, int n, int *indx, double *d)
{
	int i,imax,j,k;
	double   big,dum,sum,temp;
	double   *vv;

	vv = (double*)malloc((unsigned)(n*sizeof(double)));
	if (!vv)
	{
		fprintf(stderr,"Error Allocating Vector Memory\n");
		exit(1);
	}
	*d = 1.0;
	for (i=0;i<n;i++)
	{
		big = 0.0;
		for (j=0;j<n;j++)
		{
			if ((temp=fabs(a[i][j])) > big) big = temp;
		}
		if (big == 0.0)
		{
			fprintf(stderr,"Singular Matrix in Routine LUDCMP\n");
			for (j=0;j<n;j++) printf(" %f ",a[i][j]); printf("/n");
			exit(1);
		}
		vv[i] = 1.0/big;
	}
	for (j=0;j<n;j++)
	{
		for (i=0;i<j;i++)
		{
			sum = a[i][j];
			for (k=0;k<i;k++) sum -= a[i][k] * a[k][j];
			a[i][j] = sum;
		}
		big = 0.0;
		for (i=j;i<n;i++)
		{
			sum = a[i][j];
			for (k=0;k<j;k++) sum -= a[i][k] * a[k][j];
			a[i][j] = sum;
			if ((dum=vv[i]*fabs(sum)) >= big)
			{
				big = dum;
				imax = i;
			}
		}
		if (j != imax)
		{
			for (k=0;k<n;k++)
			{
				dum = a[imax][k];
				a[imax][k] = a[j][k];
				a[j][k] = dum;
			}
			*d = -(*d);
			vv[imax] = vv[j];
		}
		indx[j] = imax;
		if (a[j][j] == 0.0) a[j][j] = TINY;
		if (j != n-1)
		{
			dum = 1.0 / a[j][j];
			for (i=j+1;i<n;i++) a[i][j] *= dum;
		}
	}
	free(vv);
}

void lubksb(double **a, int n, int *indx, double *b)
{
	int i,ip,j,ii=-1;
	double   sum;

	for (i=0;i<n;i++)
	{
		ip = indx[i];
		sum = b[ip];
		b[ip] = b[i];
		if (ii>=0)
			for (j=ii;j<i;j++) sum -= a[i][j] * b[j];
		else if (sum) ii = i;
		b[i] = sum;
	}
	for (i=n-1;i>=0;i--)
	{
		sum = b[i];
		for (j=i+1;j<n;j++) sum -= a[i][j] * b[j];
		b[i] = sum / a[i][i];
	}
}

