/*
 * rbf_kernel.c
 *
 *  Created on: 14 Jun 2016
 *      Author: luism_000
 */


//function H = RBFKernel(F,MU,SIGMA,N,k)
//%% RBF Kernel
//%For CTranspose ' see:http://www.mathworks.com/help/matlab/ref/ctranspose.html
//if k == 1
// H         = ones(N,1);
//else
// H         = zeros(N,1);
//  for i    = 1:N
//    H(i,1) = exp((-1/2)*(F(i,:)'-MU(:,k))'*(inv(SIGMA(:,:,k)))*(F(i,:)'-MU(:,k)));
//  end
//end
//end

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

double *rbf_kernel(int F_size, double F[2][F_size], double **MU, double ***SIGMA, int k){

	double *h;
	h = (double *)malloc(sizeof(double)*F_size);

	if(k == 1){
		int i;
		for(i=0; i<F_size;i++){
			h[i] = 1;
		}
		return h;
	}else{

		int i;
		for(i=0; i<F_size;i++){

			//double f_i[2] = {F[0][i],F[1][i]};//F(i,:)'
			//MU[kmi_iteration]; <=> MU(:,k)
			double f_minus_mu[2] =
			{F[0][i]-MU[k][0],F[1][i]-MU[k][1]};//(F(1,:)'-MU(:,1))'

			double sigma[2][2] = {{SIGMA[k][0][0],SIGMA[k][0][1]},{SIGMA[k][1][0],SIGMA[k][1][1]}};//SIGMA(:,:,k)
			//			printf("sigma[%i][0][0] = %lf\n",k,SIGMA[k][0][0]);
			//			printf("sigma[%i][0][1] = %lf\n",k,SIGMA[k][0][1]);
			//			printf("sigma[%i][1][0] = %lf\n",k,SIGMA[k][1][0]);
			//			printf("sigma[%i][1][1] = %lf\n",k,SIGMA[k][1][1]);
			double factor = 1/((sigma[0][0]*sigma[1][1])-(sigma[1][0]*sigma[0][1]));
			//			printf("FACTOR: %lf\n",factor);
			double inv_sigma[2][2] = {{(sigma[1][1]*factor),-(sigma[1][0]*factor)},
					{-(sigma[0][1]*factor),(sigma[0][0]*factor)}};//inv(SIGMA(:,:,k))

//			printf("[%lf,%lf]\n[%lf,%lf]\n\n",inv_sigma[0][0],inv_sigma[0][1],inv_sigma[1][0],inv_sigma[1][1]);

			double mul_sigma[2] = {f_minus_mu[0]*inv_sigma[0][0]+f_minus_mu[1]*inv_sigma[0][1],f_minus_mu[0]*inv_sigma[1][0]+f_minus_mu[1]*inv_sigma[1][1]};//(F(i,:)'-MU(:,k))'*(inv(SIGMA(:,:,k)))

			double result = mul_sigma[0]*f_minus_mu[0]+mul_sigma[1]*f_minus_mu[1];

			result = exp((-0.5)*result);

			h[i] = result;
		}

		return h;
	}
}
