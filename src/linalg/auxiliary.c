 /*******************************************************************************
 * Copyright (c) 2016, College of William & Mary
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the College of William & Mary nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COLLEGE OF WILLIAM & MARY BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * PRIMME: https://github.com/primme/primme
 * Contact: Andreas Stathopoulos, a n d r e a s _at_ c s . w m . e d u
 *******************************************************************************
 * File: auxiliary.c
 *
 * Purpose - Miscellanea functions to copy and permuting matrices.
 *
 ******************************************************************************/


#include <stdlib.h>   /* free */
#include <string.h>   /* memmove */
#include <assert.h>
#include <math.h>
#include "template.h"
#include "auxiliary.h"
#include "blaslapack.h"

/******************************************************************************
 * Function Num_copy_matrix - Copy the matrix x into y
 *
 * PARAMETERS
 * ---------------------------
 * x           The source matrix
 * m           The number of rows of x
 * n           The number of columns of x
 * ldx         The leading dimension of x
 * y           On output y = x
 * ldy         The leading dimension of y
 *
 * NOTE: x and y *can* overlap
 *
 ******************************************************************************/
#ifdef __NVCC__
TEMPLATE_PLEASE
void NumGPU_copy_matrix_Sprimme(SCALAR *x, PRIMME_INT m, PRIMME_INT n, PRIMME_INT
      ldx, SCALAR *y, PRIMME_INT ldy, primme_params *primme) {

   PRIMME_INT i,j;

   assert(m == 0 || n == 0 || (ldx >= m && ldy >= m));

   /* Do nothing if x and y are the same matrix */
   if (x == y && ldx == ldy) return;
   if (x == NULL || y == NULL) return;
   /* Copy a contiguous memory region */

if (m == 0 || n == 0 ) return;

   if (ldx == ldy && ldx == m) {

//	SCALAR *tmp;
//	magma_malloc((void**)&tmp,sizeof(SCALAR)*m*n);

	#ifdef USE_DOUBLE
//	magmablas_dlacpy(MagmaFull,m,n,x,ldx,tmp,m,primme->queue);
//	magmablas_dlacpy(MagmaFull,m,n,tmp,m,y,ldy,primme->queue);
	magmablas_dlacpy(MagmaFull,m,n,x,ldx,y,ldy,primme->queue);
	#elif USE_DOUBLECOMPLEX
	magmablas_zlacpy(MagmaFull,m,n,x,ldx,y,ldy,primme->queue);
	#elif USE_FLOAT
	magmablas_slacpy(MagmaFull,m,n,x,ldx,y,ldy,primme->queue);
	#elif USE_FLOATCOMPLEX
	magmablas_clacpy(MagmaFull,m,n,x,ldx,y,ldy,primme->queue);
	#endif

//	magma_free(tmp);
   }

   /* Copy the matrix some rows back or forward */
   else if (ldx == ldy && (y > x ? y-x : x-y) < ldx) {
	 SCALAR *tmp;
	 magma_malloc((void**)&tmp,sizeof(SCALAR)*m);

	 for (i=0; i<n; i++){
		 magma_copyvector(m,sizeof(SCALAR),x+i*ldx,1,tmp,1,primme->queue);
		 magma_copyvector(m,sizeof(SCALAR),tmp,1,y+i*ldy,1,primme->queue);

	 }


	 magma_free(tmp);
   }

   /* Copy the matrix some columns forward */
   else if (ldx == ldy && y > x && y-x > ldx) {
	 SCALAR *tmp;
         magma_malloc((void**)&tmp,sizeof(SCALAR)*m);

	 for (i=n-1; i>=0; i--){
		magma_copyvector(m,sizeof(SCALAR),x+i*ldx,1,tmp,1,primme->queue);
		magma_copyvector(m,sizeof(SCALAR),tmp,1,y+i*ldy,1,primme->queue);
	 }
	 magma_free(tmp);
   }

   /* Copy the matrix some columns backward, and other cases */
   else {
      SCALAR *tmp;
      magma_malloc((void**)&tmp,sizeof(SCALAR)*m);

      for (i=0; i<n; i++){
		magma_copyvector(m,sizeof(SCALAR),x+i*ldx,1,tmp,1,primme->queue);
		magma_copyvector(m,sizeof(SCALAR),tmp,1,y+i*ldy,1,primme->queue);
	 }
	 magma_free(tmp);

   }

}

#endif

TEMPLATE_PLEASE
void Num_copy_matrix_Sprimme(SCALAR *x, PRIMME_INT m, PRIMME_INT n, PRIMME_INT
      ldx, SCALAR *y, PRIMME_INT ldy) {
   PRIMME_INT i,j;


   assert(m == 0 || n == 0 || (ldx >= m && ldy >= m));

   /* Do nothing if x and y are the same matrix */
   if (x == y && ldx == ldy) return;

   /* Copy a contiguous memory region */
   if (ldx == ldy && ldx == m) {
	 
      memmove(y, x, sizeof(SCALAR)*m*n);
   }

   /* Copy the matrix some rows back or forward */
   else if (ldx == ldy && (y > x ? y-x : x-y) < ldx) {
	 for (i=0; i<n; i++)
           memmove(&y[i*ldy], &x[i*ldx], sizeof(SCALAR)*m);
   }

   /* Copy the matrix some columns forward */
   else if (ldx == ldy && y > x && y-x > ldx) {
	 for (i=n-1; i>=0; i--)
           for (j=0; j<m; j++)
             y[i*ldy+j] = x[i*ldx+j];
   }

   /* Copy the matrix some columns backward, and other cases */
   else {
	 /* TODO: assert x and y don't overlap */
      for (i=0; i<n; i++)
         for (j=0; j<m; j++)
            y[i*ldy+j] = x[i*ldx+j];
   }

}

/******************************************************************************
 * Function Num_copy_matrix_columns - Copy the matrix x(xin) into y(yin)
 *
 * PARAMETERS
 * ---------------------------
 * x           The source matrix
 * m           The number of rows of x
 * xin         The column indices to copy
 * n           The number of columns of x
 * ldx         The leading dimension of x
 * y           On output y(yin) = x(xin)
 * yin         The column indices of y to be modified
 * ldy         The leading dimension of y
 *
 * NOTE: x(xin) and y(yin) *cannot* overlap
 * WARNING: compilers weren't able to optimize xin or yin being NULL;
 *    please use Num_copy_matrix as much as possible.
 *
 ******************************************************************************/
#ifdef __NVCC__

TEMPLATE_PLEASE
void NumGPU_copy_matrix_columns_Sprimme(SCALAR *x, PRIMME_INT m, int *xin, int n,
      PRIMME_INT ldx, SCALAR *y, int *yin, PRIMME_INT ldy,primme_params *primme) {
   int i;
   PRIMME_INT j;

   /* TODO: assert x and y don't overlap */
    SCALAR *tmp;
    magma_malloc((void**)&tmp,m*sizeof(SCALAR));
    for(i=0;i<n;i++){
       #ifdef USE_DOUBLE
       magma_dcopy(m,x + (xin?xin[i]:i)*ldx,1,tmp,1,primme->queue);
       magma_dcopy(m,tmp,1,y+(yin?yin[i]:i)*ldy,1,primme->queue);
       #elif USE_DOUBLECOMPLEX
       magma_zcopy(m,(magmaDoubleComplex_const_ptr)(x + (xin?xin[i]:i)*ldx),1,
           (magmaDoubleComplex_ptr)(tmp),1,primme->queue);
       magma_zcopy(m,(magmaDoubleComplex_const_ptr)(tmp),1,
           (magmaDoubleComplex_ptr)(y+(yin?yin[i]:i)*ldy),1,primme->queue);
       #elif USE_FLOAT
       magma_scopy(m,x + (xin?xin[i]:i)*ldx,1,tmp,1,primme->queue);
       magma_scopy(m,tmp,1,y+(yin?yin[i]:i)*ldy,1,primme->queue);
       #elif USE_FLOATCOMPLEX
       magma_ccopy(m,(magmaFloatComplex_const_ptr)(x + (xin?xin[i]:i)*ldx),1,
           (magmaFloatComplex_ptr)(tmp),1,primme->queue);
       magma_ccopy(m,(magmaFloatComplex_const_ptr)(tmp),1,
           (magmaFloatComplex_ptr)(y+(yin?yin[i]:i)*ldy),1,primme->queue);
       #endif
    }
       magma_free(tmp);
}

#endif
TEMPLATE_PLEASE
void Num_copy_matrix_columns_Sprimme(SCALAR *x, PRIMME_INT m, int *xin, int n,
      PRIMME_INT ldx, SCALAR *y, int *yin, PRIMME_INT ldy) {

   int i;
   PRIMME_INT j;

   /* TODO: assert x and y don't overlap */
   for (i=0; i<n; i++)
      for (j=0; j<m; j++)
         y[(yin?yin[i]:i)*ldy+j] = x[(xin?xin[i]:i)*ldx+j];
}

/******************************************************************************
 * Function Num_zero_matrix - Zero the matrix
 *
 * PARAMETERS
 * ---------------------------
 * x           The matrix
 * m           The number of rows of x
 * n           The number of columns of x
 * ldx         The leading dimension of x
 *
 ******************************************************************************/
#ifdef __NVCC__

TEMPLATE_PLEASE
void NumGPU_zero_matrix_Sprimme(SCALAR *x, PRIMME_INT m, PRIMME_INT n,
      PRIMME_INT ldx,primme_params *primme) {

   PRIMME_INT i,j;

//   for (i=0; i<n; i++)
//      magma_dscal(m,MAGMA_D_ZERO,x+i*ldx,1,primme->queue);
//      for (j=0; j<m; j++)
//         x[i*ldx+j] = 0.0;

         #ifdef USE_DOUBLE
	 magmablas_dlaset(MagmaFull,m,n,MAGMA_D_ZERO,MAGMA_D_ZERO,x,ldx,primme->queue);
	 #elif USE_DOUBLECOMPLEX
	 magmablas_zlaset(MagmaFull,m,n,MAGMA_Z_ZERO,MAGMA_Z_ZERO,(magmaDoubleComplex_ptr)x,ldx,primme->queue);
	 #elif USE_FLOAT
	 magmablas_slaset(MagmaFull,m,n,MAGMA_S_ZERO,MAGMA_S_ZERO,x,ldx,primme->queue);
	 #elif USE_FLOATCOMPLEX
	 magmablas_claset(MagmaFull,m,n,MAGMA_C_ZERO,MAGMA_C_ZERO,(magmaFloatComplex_ptr)x,ldx,primme->queue);
	 #endif
} 

#endif
TEMPLATE_PLEASE
void Num_zero_matrix_Sprimme(SCALAR *x, PRIMME_INT m, PRIMME_INT n,
      PRIMME_INT ldx) {

   PRIMME_INT i,j;

   for (i=0; i<n; i++)
      for (j=0; j<m; j++)
         x[i*ldx+j] = 0.0;
} 


/******************************************************************************
 * Function Num_copy_trimatrix - Copy the upper/lower triangular part of the
 *    matrix x into y
 *
 * PARAMETERS
 * ---------------------------
 * x           The source matrix
 * m           The number of rows of x
 * n           The number of columns of x
 * ldx         The leading dimension of x
 * ul          if 0, copy the upper part; otherwise copy the lower part 
 * i0          The row index that diagonal starts from
 * y           On output y = x
 * ldy         The leading dimension of y
 * zero        If nonzero, zero the triangular part not copied
 *
 * NOTE: the distance between x and y can be less than ldx, or
 *       x and y *cannot* overlap at all
 *
 ******************************************************************************/
#ifdef __NVCC__
TEMPLATE_PLEASE
void NumGPU_copy_trimatrix_Sprimme(SCALAR *x, int m, int n, int ldx, int ul,
      int i0, SCALAR *y, int ldy, int zero, primme_params *primme) {

   int i, j, jm;

  
   assert(m == 0 || n == 0 || (ldx >= m && ldy >= m));
   if (x == y) return;


   SCALAR *tmp;
   magma_malloc((void**)&tmp,m*sizeof(SCALAR));
   if (ul == 0) {
      /* Copy upper part */

      if (ldx == ldy && (y > x ? y-x : x-y) < ldx) {
         /* x and y overlap */
        for (i=0; i<n; i++) {
 		magma_copyvector(min(i0+i+1, m),sizeof(SCALAR),x+i*ldx,1,tmp,1,primme->queue);
		magma_copyvector(min(i0+i+1, m),sizeof(SCALAR),tmp,1,y+i*ldy,1,primme->queue);

      		if (zero){ //for (j=min(i0+i+1, m); j<m; j++) y[i*ldy+j] = 0.0;
   			   #ifdef USE_DOUBLE
   			   magma_dscal(m-min(i0+i+1,m),0.0,(magmaDouble_ptr)y+i*ldy+min(i0+i+1, m),1,primme->queue);
   			   #elif USE_DOUBLECOMPLEX
   			   magma_zdscal(m-min(i0+i+1,m),0.0,(magmaDoubleComplex_ptr)y+i*ldy+min(i0+i+1, m),1,primme->queue);
   			   #elif USE_FLOAT
   			   magma_sscal(m-min(i0+i+1,m),0.0,(magmaFloat_ptr)y+i*ldy+min(i0+i+1, m),1,primme->queue);
   		           #elif USE_FLOATCOMPLEX
   			   magma_csscal(m-min(i0+i+1,m),0.0,(magmaFloatComplex_ptr)y+i*ldy+min(i0+i+1, m),1,primme->queue);
   			   #endif
                }
		
         }   
      }
      else {
         /* x and y don't overlap */
         for (i=0; i<n; i++) {
   		  magma_copyvector(min(i0+i+1, m),sizeof(SCALAR),x+i*ldx,1,tmp,1,primme->queue);
   		  magma_copyvector(min(i0+i+1, m),sizeof(SCALAR),tmp,1,y+i*ldy,1,primme->queue);
		  if (zero){ //for (j=min(i0+i+1, m); j<m; j++) y[i*ldy+j] = 0.0;
      		 	#ifdef USE_DOUBLE
      			magma_dscal(m-min(i0+i+1,m),0.0,(magmaDouble_ptr)y+i*ldy+min(i0+i+1, m),1,primme->queue);
      			#elif USE_DOUBLECOMPLEX
      			magma_zdscal(m-min(i0+i+1,m),0.0,(magmaDoubleComplex_ptr)y+i*ldy+min(i0+i+1, m),1,primme->queue);
      			#elif USE_FLOAT
      			magma_sscal(m-min(i0+i+1,m),0.0,(magmaFloat_ptr)y+i*ldy+min(i0+i+1, m),1,primme->queue);
      		  	#elif USE_FLOATCOMPLEX
      			magma_csscal(m-min(i0+i+1,m),0.0,(magmaFloatComplex_ptr)y+i*ldy+min(i0+i+1, m),1,primme->queue);
      			#endif
   		   }
         }
      }
   }
   else {
      /* Copy lower part */

      if (ldx == ldy && (y > x ? y-x : x-y) < ldx) {
         /* x and y overlap */
         for (i=0; i<n; i++) {
   		  magma_copyvector((m-min(i0+i, m)),sizeof(SCALAR),x+i*ldx+i0+i,1,tmp,1,primme->queue);
   		  magma_copyvector((m-min(i0+i, m)),sizeof(SCALAR),tmp,1,y+i*ldy+i0+i,1,primme->queue);

   		  if (zero){ //for (j=min(i0+i+1, m); j<m; j++) y[i*ldy+j] = 0.0;
      			#ifdef USE_DOUBLE
      			magma_dscal(min(i0+i,m),0.0,(magmaDouble_ptr)y+i*ldy,1,primme->queue);
      			#elif USE_DOUBLECOMPLEX
      			magma_zdscal(min(i0+i,m),0.0,(magmaDoubleComplex_ptr)y+i*ldy,1,primme->queue);
      			#elif USE_FLOAT
      			magma_sscal(min(i0+i,m),0.0,(magmaFloat_ptr)y+i*ldy,1,primme->queue);
      		  	#elif USE_FLOATCOMPLEX
      			magma_csscal(min(i0+i,m),0.0,(magmaFloatComplex_ptr)y+i*ldy,1,primme->queue);
      			#endif
      		}
		
         }
      }
      else {
         /* x and y don't overlap */
         for (i=0; i<n; i++) {
   		  magma_copyvector(min(i0+i, m),sizeof(SCALAR),x+i*ldx+i0+i,1,tmp,1,primme->queue);
   		  magma_copyvector(min(i0+i, m),sizeof(SCALAR),tmp,1,y+i*ldy+i0+i,1,primme->queue);
   		  if (zero){ //for (j=min(i0+i+1, m); j<m; j++) y[i*ldy+j] = 0.0;
      			#ifdef USE_DOUBLE
      			magma_dscal(min(i0+i,m),0.0,(magmaDouble_ptr)y+i*ldy,1,primme->queue);
      			#elif USE_DOUBLECOMPLEX
      			magma_zdscal(min(i0+i,m),0.0,(magmaDoubleComplex_ptr)y+i*ldy,1,primme->queue);
      			#elif USE_FLOAT
      			magma_sscal(min(i0+i,m),0.0,(magmaFloat_ptr)y+i*ldy,1,primme->queue);
      		  	#elif USE_FLOATCOMPLEX
      			magma_csscal(min(i0+i,m),0.0,(magmaFloatComplex_ptr)y+i*ldy,1,primme->queue);
      			#endif
   		  }
         }
      }
   }
}

#endif

TEMPLATE_PLEASE
void Num_copy_trimatrix_Sprimme(SCALAR *x, int m, int n, int ldx, int ul,
      int i0, SCALAR *y, int ldy, int zero) {

   int i, j, jm;

   assert(m == 0 || n == 0 || (ldx >= m && ldy >= m));
   if (x == y) return;
   if (ul == 0) {
      /* Copy upper part */

      if (ldx == ldy && (y > x ? y-x : x-y) < ldx) {
         /* x and y overlap */
         for (i=0; i<n; i++) {
            memmove(&y[i*ldy], &x[i*ldx], sizeof(SCALAR)*min(i0+i+1, m));
            /* zero lower part*/
            if (zero) for (j=min(i0+i+1, m); j<m; j++) y[i*ldy+j] = 0.0;
         }
      }
      else {
         /* x and y don't overlap */
         for (i=0; i<n; i++) {
            for (j=0, jm=min(i0+i+1, m); j<jm; j++)
               y[i*ldy+j] = x[i*ldx+j];
            /* zero lower part*/
            if (zero) for (j=min(i0+i+1, m); j<m; j++) y[i*ldy+j] = 0.0;
         }
      }
   }
   else {
      /* Copy lower part */

      if (ldx == ldy && (y > x ? y-x : x-y) < ldx) {
         /* x and y overlap */
         for (i=0; i<n; i++) {
            memmove(&y[i*ldy+i0+i], &x[i*ldx+i0+i], sizeof(SCALAR)*(m-min(i0+i, m)));
            /* zero upper part*/
            if (zero) for (j=0, jm=min(i0+i, m); j<jm; j++) y[i*ldy+j] = 0.0;
         }
      }
      else {
         /* x and y don't overlap */
         for (i=0; i<n; i++) {
            for (j=i+i0; j<m; j++)
               y[i*ldy+j] = x[i*ldx+j];
            /* zero upper part*/
            if (zero) for (j=0, jm=min(i0+i, m); j<jm; j++) y[i*ldy+j] = 0.0;
         }
      }
   }
}


/******************************************************************************
 * Function Num_copy_trimatrix_compact - Copy the upper triangular part of the matrix x
 *    into y contiguously, i.e., y has all columns of x row-stacked
 *
 * PARAMETERS
 * ---------------------------
 * x           The source upper triangular matrix
 * m           The number of rows of x
 * n           The number of columns of x
 * ldx         The leading dimension of x
 * i0          The row index that diagonal starts from
 * y           On output y = x and nonzero elements of y are contiguous
 * ly          Output the final length of y
 *
 * NOTE: x and y *cannot* overlap
 *
 ******************************************************************************/
#ifdef __NVCC__
TEMPLATE_PLEASE
void NumGPU_copy_trimatrix_compact_Sprimme(SCALAR *x, PRIMME_INT m, int n,
      PRIMME_INT ldx, int i0, SCALAR *y, int *ly,primme_params *primme) {

   int i, j, k=0;

   assert(m == 0 || n == 0 || ldx >= m);
   
   for(i=0;i<n;i++){
	magma_copyvector(i+i0+1,sizeof(SCALAR),x,1,y,1,primme->queue); 
  	y += i+i0+1;
	x += i*ldx;
	k += i+i0+1;
   }

   if(ly) *ly = k;
}

#endif

TEMPLATE_PLEASE
void Num_copy_trimatrix_compact_Sprimme(SCALAR *x, PRIMME_INT m, int n,
      PRIMME_INT ldx, int i0, SCALAR *y, int *ly) {

   int i, j, k;

   assert(m == 0 || n == 0 || ldx >= m);

   for (i=0, k=0; i<n; i++)
      for (j=0; j<=i+i0; j++)
         y[k++] = x[i*ldx+j];
   if (ly) *ly = k;
}

/******************************************************************************
 * Function Num_copy_compact_trimatrix - Copy y into the upper triangular part of the
 *    matrix x
 *
 * PARAMETERS
 * ---------------------------
 * x           The source vector
 * m           The number of rows of y
 * n           The number of columns of y
 * i0          The row index that diagonal starts from
 * y           On output the upper triangular part of y has x
 * ldy         The leading dimension of y
 *
 * NOTE: x and y *cannot* overlap
 *
 ******************************************************************************/
#ifdef __NVCC__
TEMPLATE_PLEASE
void NumGPU_copy_compact_trimatrix_Sprimme(SCALAR *x, PRIMME_INT m, int n, int i0,
      SCALAR *y, int ldy,primme_params *primme){

   int i, j, k;

   assert(m == 0 || n == 0 || (ldy >= m && m >= n));
   SCALAR *startY = y;

   k = k=(n+1)*n/2+i0*n-1;
   magma_copyvector(i+i0+1,sizeof(SCALAR),x+k,1,y+i*ldy,1,primme->queue);

   for(i=n-2; i>=0; i--){
	k-=i+i0+1;
	y = startY+ i*ldy;
	magma_copyvector(i+i0+1,sizeof(SCALAR),x+k,1,y,1,primme->queue);

   }
/*
   for (j=i+i0; j>=0; j--)
        y[i*ldy+j] = x[k--];
*/
}
#endif

TEMPLATE_PLEASE
void Num_copy_compact_trimatrix_Sprimme(SCALAR *x, PRIMME_INT m, int n, int i0,
      SCALAR *y, int ldy) {

   int i, j, k;

   assert(m == 0 || n == 0 || (ldy >= m && m >= n));

   for (i=n-1, k=(n+1)*n/2+i0*n-1; i>=0; i--)
      for (j=i+i0; j>=0; j--)
         y[i*ldy+j] = x[k--];
}


/******************************************************************************
 * Subroutine permute_vecs - This routine permutes a set of vectors according
 *            to a permutation array perm.
 *
 * INPUT ARRAYS AND PARAMETERS
 * ---------------------------
 * m, n, ld    The number of rows and columns and the leading dimension of vecs
 * perm        The permutation of the columns
 * rwork       Temporary space of size the number of rows
 * iwork       Temporary space of size the number of columns
 *
 * INPUT/OUTPUT ARRAYS
 * -------------------
 * vecs        The matrix whose columns will be reordered
 *
 ******************************************************************************/
#ifdef __NVCC__
TEMPLATE_PLEASE
void permuteGPU_vecs_Sprimme(SCALAR *vecs, PRIMME_INT m, int n, PRIMME_INT ld,
      int *perm_, SCALAR *rwork, int *iwork, primme_params *primme) {

   int currentIndex;     /* Index of vector in sorted order                   */
   int sourceIndex;      /* Position of out-of-order vector in original order */
   int destinationIndex; /* Position of out-of-order vector in sorted order   */
   int tempIndex;        /* Used to swap                                      */
   int *perm=iwork;      /* A copy of perm_                                   */



   SCALAR *cpu_vecs = malloc(m*ld*sizeof(SCALAR));
   SCALAR *cpu_rwork = malloc(m*sizeof(SCALAR));

   /* Check that perm_ and iwork do not overlap */

   assert((perm_>iwork?perm_-iwork:iwork-perm_) >= n);

   /* Check perm_ is a permutation */

#ifndef NDEBUG
   for (tempIndex=0; tempIndex<n; tempIndex++) perm[tempIndex] = 0;
   for (tempIndex=0; tempIndex<n; tempIndex++) {
      assert(0 <= perm_[tempIndex] && perm_[tempIndex] < n);
      perm[perm_[tempIndex]] = 1;
   }
   for (tempIndex=0; tempIndex<n; tempIndex++) assert(perm[tempIndex] == 1);
#endif

   /* Copy of perm_ into perm, to avoid to modify the input permutation */

   for (tempIndex=0; tempIndex<n; tempIndex++)
      perm[tempIndex] = perm_[tempIndex];

   /* Continue until all vectors are in the sorted order */

   currentIndex = 0;
   while (1) {

      /* Find a vector that does not belong in its original position */
      while ((currentIndex < n) && (perm[currentIndex] == currentIndex)) {
         currentIndex++;
      }

      /* Return if they are in the sorted order */
      if (currentIndex >= n) {
         return;
      }

      /* Copy the vector to a buffer for swapping */
      NumGPU_copy_Sprimme(m, &vecs[currentIndex*ld], 1, rwork, 1, primme);

      destinationIndex = currentIndex;
      /* Copy vector perm[destinationIndex] into position destinationIndex */

      while (perm[destinationIndex] != currentIndex) {

         sourceIndex = perm[destinationIndex];

         NumGPU_copy_Sprimme(m, &vecs[sourceIndex*ld], 1, 
	            &vecs[destinationIndex*ld], 1, primme);
         tempIndex = perm[destinationIndex];
         perm[destinationIndex] = destinationIndex;
         destinationIndex = tempIndex;
      }

      /* Copy the vector from the buffer to where it belongs */

      NumGPU_copy_Sprimme(m, rwork, 1, &vecs[destinationIndex*ld], 1, primme);
      perm[destinationIndex] = destinationIndex;

      currentIndex++;
   }

   /* Check permutation */
   for (currentIndex=0; currentIndex < n; currentIndex++)
      assert(perm[currentIndex] == currentIndex);

   free(cpu_vecs);
   free(cpu_rwork);

}
#endif

TEMPLATE_PLEASE
void permute_vecs_Sprimme(SCALAR *vecs, PRIMME_INT m, int n, PRIMME_INT ld,
      int *perm_, SCALAR *rwork, int *iwork) {

   int currentIndex;     /* Index of vector in sorted order                   */
   int sourceIndex;      /* Position of out-of-order vector in original order */
   int destinationIndex; /* Position of out-of-order vector in sorted order   */
   int tempIndex;        /* Used to swap                                      */
   int *perm=iwork;      /* A copy of perm_                                   */

   /* Check that perm_ and iwork do not overlap */

   assert((perm_>iwork?perm_-iwork:iwork-perm_) >= n);

   /* Check perm_ is a permutation */

#ifndef NDEBUG
   for (tempIndex=0; tempIndex<n; tempIndex++) perm[tempIndex] = 0;
   for (tempIndex=0; tempIndex<n; tempIndex++) {
      assert(0 <= perm_[tempIndex] && perm_[tempIndex] < n);
      perm[perm_[tempIndex]] = 1;
   }
   for (tempIndex=0; tempIndex<n; tempIndex++) assert(perm[tempIndex] == 1);
#endif

   /* Copy of perm_ into perm, to avoid to modify the input permutation */

   for (tempIndex=0; tempIndex<n; tempIndex++)
      perm[tempIndex] = perm_[tempIndex];

   /* Continue until all vectors are in the sorted order */

   currentIndex = 0;
   while (1) {

      /* Find a vector that does not belong in its original position */
      while ((currentIndex < n) && (perm[currentIndex] == currentIndex)) {
         currentIndex++;
      }

      /* Return if they are in the sorted order */
      if (currentIndex >= n) {
         return;
      }

      /* Copy the vector to a buffer for swapping */
      Num_copy_Sprimme(m, &vecs[currentIndex*ld], 1, rwork, 1);

      destinationIndex = currentIndex;
      /* Copy vector perm[destinationIndex] into position destinationIndex */

      while (perm[destinationIndex] != currentIndex) {

         sourceIndex = perm[destinationIndex];
         Num_copy_Sprimme(m, &vecs[sourceIndex*ld], 1, 
            &vecs[destinationIndex*ld], 1);
         tempIndex = perm[destinationIndex];
         perm[destinationIndex] = destinationIndex;
         destinationIndex = tempIndex;
      }

      /* Copy the vector from the buffer to where it belongs */
      Num_copy_Sprimme(m, rwork, 1, &vecs[destinationIndex*ld], 1);
      perm[destinationIndex] = destinationIndex;

      currentIndex++;
   }

   /* Check permutation */
   for (currentIndex=0; currentIndex < n; currentIndex++)
      assert(perm[currentIndex] == currentIndex);

}

#ifdef USE_DOUBLE
TEMPLATE_PLEASE
void permute_vecs_iprimme(int *vecs, int n, int *perm_, int *iwork) {

   int currentIndex;     /* Index of vector in sorted order                   */
   int sourceIndex;      /* Position of out-of-order vector in original order */
   int destinationIndex; /* Position of out-of-order vector in sorted order   */
   int tempIndex;        /* Used to swap                                      */
   int *perm=iwork;      /* A copy of perm_                                   */
   int aux;

   /* Check that perm_ and iwork do not overlap */

   assert((perm_>iwork?perm_-iwork:iwork-perm_) >= n);

   /* Check perm_ is a permutation */

#ifndef NDEBUG
   for (tempIndex=0; tempIndex<n; tempIndex++) perm[tempIndex] = 0;
   for (tempIndex=0; tempIndex<n; tempIndex++) {
      assert(0 <= perm_[tempIndex] && perm_[tempIndex] < n);
      perm[perm_[tempIndex]] = 1;
   }
   for (tempIndex=0; tempIndex<n; tempIndex++) assert(perm[tempIndex] == 1);
#endif

   /* Copy of perm_ into perm, to avoid to modify the input permutation */

   for (tempIndex=0; tempIndex<n; tempIndex++)
      perm[tempIndex] = perm_[tempIndex];

   /* Continue until all vectors are in the sorted order */

   currentIndex = 0;
   while (1) {

      /* Find a vector that does not belong in its original position */
      while ((currentIndex < n) && (perm[currentIndex] == currentIndex)) {
         currentIndex++;
      }

      /* Return if they are in the sorted order */
      if (currentIndex >= n) {
         return;
      }

      /* Copy the vector to a buffer for swapping */
      aux = vecs[currentIndex];

      destinationIndex = currentIndex;
      /* Copy vector perm[destinationIndex] into position destinationIndex */

      while (perm[destinationIndex] != currentIndex) {

         sourceIndex = perm[destinationIndex];
         vecs[destinationIndex] = vecs[sourceIndex];
         tempIndex = perm[destinationIndex];
         perm[destinationIndex] = destinationIndex;
         destinationIndex = tempIndex;
      }

      /* Copy the vector from the buffer to where it belongs */
      vecs[destinationIndex] = aux;
      perm[destinationIndex] = destinationIndex;

      currentIndex++;
   }

   /* Check permutation */
   for (currentIndex=0; currentIndex < n; currentIndex++)
      assert(perm[currentIndex] == currentIndex);

}
#endif


/******************************************************************************
 * Subroutine Num_compact_vecs - copy certain columns of matrix into another
 *       matrix, i.e., work = vecs(perm). If avoidCopy and perm indices are
 *       consecutive the routine returns a reference in vecs and doesn't copy.
 *            
 *
 * PARAMETERS
 * ---------------------------
 * 
 * vecs        The matrix
 * m           The number of rows of vecs
 * n           The number of columns of to copy
 * ld          The leading dimension of vecs
 * perm        The indices of columns to copy
 * work        The columns are copied to this matrix
 * ldwork      The leading dimension of work
 * avoidCopy   If nonzero, the copy is avoid
 *
 * return      Reference of a matrix with the columns ordered as perm
 *
 ******************************************************************************/
#ifdef __NVCC__
TEMPLATE_PLEASE
SCALAR* NumGPU_compact_vecs_Sprimme(SCALAR *vecs, PRIMME_INT m, int n, 
      PRIMME_INT ld, int *perm, SCALAR *work, PRIMME_INT ldwork,
      int avoidCopy,primme_params *primme) {

   int i;

   if (avoidCopy) {
      for (i=0; i<n-1 && perm[i]+1 == perm[i+1]; i++);
      if (i >= n-1) return vecs + ld*perm[0];
	    
   }

   for (i=0; i < n; i++) {
      NumGPU_copy_matrix_Sprimme(&vecs[perm[i]*ld], m, 1, ld, &work[i*ldwork], ld,primme);
   }
   return work;
}
#endif

TEMPLATE_PLEASE
SCALAR* Num_compact_vecs_Sprimme(SCALAR *vecs, PRIMME_INT m, int n, 
      PRIMME_INT ld, int *perm, SCALAR *work, PRIMME_INT ldwork,
      int avoidCopy) {

   int i;

   if (avoidCopy) {
      for (i=0; i<n-1 && perm[i]+1 == perm[i+1]; i++);
      if (i >= n-1) return &vecs[ld*perm[0]];
   }

   for (i=0; i < n; i++) {
      Num_copy_matrix_Sprimme(&vecs[perm[i]*ld], m, 1, ld, &work[i*ldwork], ld);
   }
   return work;
}

/*******************************************************************************
 * Subroutine compute_submatrix - This subroutine computes the nX x nX submatrix
 *    R = X'*H*X, where H stores the upper triangular part of a symmetric matrix.
 *    
 * Input parameters
 * ----------------
 * X        The coefficient vectors retained from the previous iteration
 *
 * nX       Number of columns of X
 *
 * H        Matrix
 *
 * nH       Dimension of H
 *
 * ldH      Leading dimension of H
 *
 * rwork    Work array.  Must be of size nH x nX
 *
 * lrwork   Length of the work array
 *
 * ldR      Leading dimension of R
 *
 * Output parameters
 * -----------------
 * R - nX x nX matrix computed 
 *
 ******************************************************************************/
#ifdef __NVCC__
TEMPLATE_PLEASE
int computeGPU_submatrix_Sprimme(SCALAR *X, int nX, int ldX, 
   SCALAR *H, int nH, int ldH, SCALAR *R, int ldR,
   SCALAR *rwork, size_t *lrwork,primme_params* primme) {

   /* Return memory requirement */
   if (X == NULL) {
      *lrwork = max(*lrwork, (size_t)nH*(size_t)nX);
      return 0;
   }

   if (nH == 0 || nX == 0) return 0;

   assert(*lrwork >= (size_t)nH*(size_t)nX);

   NumGPU_hemm_Sprimme("L", "U", nH, nX, 1.0, H, ldH, X, ldX, 0.0, rwork, nH, primme);
   
   NumGPU_gemm_Sprimme("C", "N", nX, nX, nH, 1.0, X, ldX, rwork, nH, 0.0, R, 
      ldR, primme);

   return 0;
}

#endif

TEMPLATE_PLEASE
int compute_submatrix_Sprimme(SCALAR *X, int nX, int ldX, 
   SCALAR *H, int nH, int ldH, SCALAR *R, int ldR,
   SCALAR *rwork, size_t *lrwork) {

   /* Return memory requirement */
   if (X == NULL) {
      *lrwork = max(*lrwork, (size_t)nH*(size_t)nX);
      return 0;
   }

   if (nH == 0 || nX == 0) return 0;

   assert(*lrwork >= (size_t)nH*(size_t)nX);

   Num_hemm_Sprimme("L", "U", nH, nX, 1.0, H, ldH, X, ldX, 0.0, rwork, nH);
   
   Num_gemm_Sprimme("C", "N", nX, nX, nH, 1.0, X, ldX, rwork, nH, 0.0, R, 
      ldR);

   return 0;
}
