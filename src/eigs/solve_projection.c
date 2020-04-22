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
 * File: solve_H.c
 * 
 * Purpose - Solves the eigenproblem for the matrix V'*A*V.
 *
 ******************************************************************************/

#include <math.h>
#include <assert.h>
#include "const.h"
#include "numerical.h"
#include "solve_projection.h"
#include "ortho.h"
#include "globalsum.h"

static int solve_H_RR_Sprimme(SCALAR *H, int ldH, SCALAR *hVecs,
   int ldhVecs, PRIMME_REAL *hVals, int basisSize, int numConverged, size_t *lrwork,
   SCALAR *rwork, int liwork, int *iwork, primme_params *primme);

static int solve_H_Harm_Sprimme(SCALAR *H, int ldH, SCALAR *QtV, int ldQtV,
   SCALAR *R, int ldR, SCALAR *hVecs, int ldhVecs, SCALAR *hU, int ldhU,
   PRIMME_REAL *hVals, int basisSize, int numConverged, double machEps,
   size_t *lrwork, SCALAR *rwork, int liwork, int *iwork,
   primme_params *primme);

static int solve_H_Ref_Sprimme(SCALAR *H, int ldH, SCALAR *hVecs,
   int ldhVecs, SCALAR *hU, int ldhU, PRIMME_REAL *hSVals, SCALAR *R, int ldR,
   PRIMME_REAL *hVals, int basisSize, int targetShiftIndex, size_t *lrwork,
   SCALAR *rwork, int liwork, int *iwork, primme_params *primme);

static int solve_H_brcast_Sprimme(int basisSize, SCALAR *hU, int ldhU,
      SCALAR *hVecs, int ldhVecs, PRIMME_REAL *hVals, PRIMME_REAL *hSVals, size_t *lrwork,
      SCALAR *rwork, primme_params *primme);


/*******************************************************************************
 * Subroutine solve_H - This procedure solves the project problem and return
 *       the projected vectors (hVecs) and values (hVals) in the order according
 *       to primme.target.
 *        
 * INPUT ARRAYS AND PARAMETERS
 * ---------------------------
 * H              The matrix V'*A*V
 * basisSize      The dimension of H, R, QtV and hU
 * ldH            The leading dimension of H
 * R              The factor R for the QR decomposition of (A - target*I)*V
 * ldR            The leading dimension of R
 * QtV            Q'*V
 * ldQtV          The leading dimension of QtV
 * numConverged   Number of eigenvalues converged to determine ordering shift
 * lrwork         Length of the work array rwork
 * primme         Structure containing various solver parameters
 * 
 * INPUT/OUTPUT ARRAYS
 * -------------------
 * hU             The left singular vectors of R or the eigenvectors of QtV/R
 * ldhU           The leading dimension of hU
 * hVecs          The coefficient vectors such as V*hVecs will be the Ritz vectors
 * ldhVecs        The leading dimension of hVecs
 * hVals          The Ritz values
 * hSVals         The singular values of R
 * rwork          Workspace
 * iwork          Workspace in integers
 *
 * Return Value
 * ------------
 * int -  0 upon successful return
 *     - -1 Num_dsyev/zheev was unsuccessful
 ******************************************************************************/

TEMPLATE_PLEASE
int solve_H_Sprimme(SCALAR *H, int basisSize, int ldH, SCALAR *R, int ldR,
   SCALAR *QtV, int ldQtV, SCALAR *hU, int ldhU, SCALAR *hVecs, int ldhVecs,
   PRIMME_REAL *hVals, PRIMME_REAL *hSVals, int numConverged, double machEps, size_t *lrwork,
   SCALAR *rwork, int liwork, int *iwork, primme_params *primme) {

   int i;
   /* In parallel (especially with heterogeneous processors/libraries) ensure */
   /* that every process has the same hVecs and hU. Only processor 0 solves   */
   /* the projected problem and broadcasts the resulting matrices to the rest */
 if (primme->procID == 0) {
      switch (primme->projectionParams.projection) {
         case primme_proj_RR:
                  #if 0 
def __NVCC__
                  if(primme!=NULL && primme->data_in_gpu==1){
                      SCALAR *cpu_H = primme->cpu_H;
                      SCALAR *cpu_hVecs = primme->cpu_hVecs;
                      SCALAR *cpu_rwork = malloc(*lrwork*sizeof(SCALAR));

                      magma_getvector(primme->maxBasisSize*primme->maxBasisSize,sizeof(SCALAR),H,1,cpu_H,1,primme->queue);
                      magma_getvector(primme->maxBasisSize*primme->maxBasisSize,sizeof(SCALAR),hVecs,1,cpu_hVecs,1,primme->queue);
 
                      primme->data_in_gpu=0;

                      CHKERR(solve_H_RR_Sprimme(cpu_H, ldH, cpu_hVecs, ldhVecs, hVals, basisSize,
                          numConverged, lrwork, cpu_rwork, liwork, iwork, primme), -1);

                      primme->data_in_gpu=1;

                      magma_setvector(primme->maxBasisSize*primme->maxBasisSize,sizeof(SCALAR),cpu_H,1,H,1,primme->queue);
                      magma_setvector(primme->maxBasisSize*primme->maxBasisSize,sizeof(SCALAR),cpu_hVecs,1,hVecs,1,primme->queue);

                      free(cpu_rwork);
                  }else
                  #endif
		  CHKERR(solve_H_RR_Sprimme(H, ldH, hVecs, ldhVecs, hVals, basisSize,
                     numConverged, lrwork, rwork, liwork, iwork, primme), -1);
	    break;

         case primme_proj_harmonic:
                  #if 0 
def __NVCC__
                  if(primme!=NULL && primme->data_in_gpu==1){
                      SCALAR *cpu_H = primme->cpu_H;
                      SCALAR *cpu_QtV = primme->cpu_QtV;
                      SCALAR *cpu_R = primme->cpu_R;
                      SCALAR *cpu_hVecs = primme->cpu_hVecs;
                      SCALAR *cpu_hU = primme->cpu_hU;

                      SCALAR *cpu_rwork = malloc(*lrwork*sizeof(SCALAR));

                      magma_getvector(primme->maxBasisSize*primme->maxBasisSize,sizeof(SCALAR),H,1,cpu_H,1,primme->queue);
                      magma_getvector(primme->maxBasisSize*ldQtV,sizeof(SCALAR),QtV,1,cpu_QtV,1,primme->queue);
                      magma_getvector(primme->maxBasisSize*ldR,sizeof(SCALAR),R,1,cpu_R,1,primme->queue);
                      magma_getvector(primme->maxBasisSize*primme->maxBasisSize,sizeof(SCALAR),hVecs,1,cpu_hVecs,1,primme->queue);
                      magma_getvector(primme->maxBasisSize*ldhU,sizeof(SCALAR),hU,1,cpu_hU,1,primme->queue);

 
                      primme->data_in_gpu=0;

                      CHKERR(solve_H_Harm_Sprimme(cpu_H, ldH, cpu_QtV, ldQtV, cpu_R, ldR, cpu_hVecs,
                            ldhVecs, cpu_hU, ldhU, hVals, basisSize, numConverged, machEps,
                            lrwork, cpu_rwork, liwork, iwork, primme), -1);

                      primme->data_in_gpu=1;

                      magma_setvector(primme->maxBasisSize*primme->maxBasisSize,sizeof(SCALAR),cpu_H,1,H,1,primme->queue);
                      magma_setvector(primme->maxBasisSize*ldQtV,sizeof(SCALAR),cpu_QtV,1,QtV,1,primme->queue);
                      magma_setvector(primme->maxBasisSize*ldR,sizeof(SCALAR),cpu_R,1,R,1,primme->queue);
                      magma_setvector(primme->maxBasisSize*primme->maxBasisSize,sizeof(SCALAR),cpu_hVecs,1,hVecs,1,primme->queue);
                      magma_setvector(primme->maxBasisSize*ldhU,sizeof(SCALAR),cpu_hU,1,hU,1,primme->queue);

                      free(cpu_rwork);
                  }else
                  #endif
		  CHKERR(solve_H_Harm_Sprimme(H, ldH, QtV, ldQtV, R, ldR, hVecs,
                     ldhVecs, hU, ldhU, hVals, basisSize, numConverged, machEps,
                     lrwork, rwork, liwork, iwork, primme), -1);
	    break;

         case primme_proj_refined:
                  #if 0 
def __NVCC__
                  if(primme!=NULL && primme->data_in_gpu==1){
                      SCALAR *cpu_H = primme->cpu_H;
                      SCALAR *cpu_hVecs = primme->cpu_hVecs;
                      SCALAR *cpu_hU = primme->cpu_hU;
                      SCALAR *cpu_R = primme->cpu_R;

                      SCALAR *cpu_rwork = malloc(*lrwork*sizeof(SCALAR));

                      magma_getvector(primme->maxBasisSize*primme->maxBasisSize,sizeof(SCALAR),H,1,cpu_H,1,primme->queue);
                      magma_getvector(primme->maxBasisSize*primme->maxBasisSize,sizeof(SCALAR),hVecs,1,cpu_hVecs,1,primme->queue);
                      magma_getvector(primme->maxBasisSize*ldR,sizeof(SCALAR),R,1,cpu_R,1,primme->queue);
                      magma_getvector(primme->maxBasisSize*ldhU,sizeof(SCALAR),hU,1,cpu_hU,1,primme->queue);
 
                      primme->data_in_gpu=0;

                      CHKERR(solve_H_Ref_Sprimme(cpu_H, ldH, cpu_hVecs, ldhVecs, cpu_hU, ldhU, hSVals, 
                         cpu_R, ldR, hVals, basisSize, numConverged, lrwork, rwork,
                         liwork, iwork, primme), -1);

                      primme->data_in_gpu=1;

                      magma_setvector(primme->maxBasisSize*primme->maxBasisSize,sizeof(SCALAR),cpu_H,1,H,1,primme->queue);
                      magma_setvector(primme->maxBasisSize*primme->maxBasisSize,sizeof(SCALAR),cpu_hVecs,1,hVecs,1,primme->queue);
                      magma_setvector(primme->maxBasisSize*ldR,sizeof(SCALAR),cpu_R,1,R,1,primme->queue);
                      magma_setvector(primme->maxBasisSize*ldhU,sizeof(SCALAR),cpu_hU,1,hU,1,primme->queue);

                      free(cpu_rwork);
                  }else
                  #endif
		 CHKERR(solve_H_Ref_Sprimme(H, ldH, hVecs, ldhVecs, hU, ldhU, hSVals, 
                     R, ldR, hVals, basisSize, numConverged, lrwork, rwork,
                     liwork, iwork, primme), -1);
		break;

         default:
            assert(0);
      }
   }


   /* Broadcast hVecs, hU, hVals, hSVals */
   CHKERR(solve_H_brcast_Sprimme(basisSize, hU, ldhU, hVecs, ldhVecs, hVals,
            hSVals, lrwork, rwork, primme), -1);

   /* Return memory requirements */

   if (H == NULL) {
      return 0;
   }

   /* -------------------------------------------------------- */
   /* Update the leftmost and rightmost Ritz values ever seen  */
   /* -------------------------------------------------------- */

   for (i=0; i<basisSize; i++) {
      primme->stats.estimateMinEVal = min(primme->stats.estimateMinEVal,
            hVals[i]); 
      primme->stats.estimateMaxEVal = max(primme->stats.estimateMaxEVal,
            hVals[i]); 
   }
   primme->stats.estimateLargestSVal = max(fabs(primme->stats.estimateMinEVal),
                                           fabs(primme->stats.estimateMaxEVal));

   return 0;
}


/*******************************************************************************
 * Subroutine solve_H_RR - This procedure solves the eigenproblem for the
 *            matrix H.
 *        
 * INPUT ARRAYS AND PARAMETERS
 * ---------------------------
 * H              The matrix V'*A*V
 * basisSize      The dimension of H, R, hU
 * ldH            The leading dimension of H
 * numConverged   Number of eigenvalues converged to determine ordering shift
 * lrwork         Length of the work array rwork
 * primme         Structure containing various solver parameters
 * 
 * INPUT/OUTPUT ARRAYS
 * -------------------
 * hVecs          The eigenvectors of H or the right singular vectors
 * ldhVecs        The leading dimension of hVecs
 * hVals          The Ritz values
 * hSVals         The singular values of R
 * rwork          Workspace
 * iwork          Workspace in integers
 *
 * Return Value
 * ------------
 * int -  0 upon successful return
 *     - -1 Num_dsyev/zheev was unsuccessful
 ******************************************************************************/

static int solve_H_RR_Sprimme(SCALAR *H, int ldH, SCALAR *hVecs,
   int ldhVecs, PRIMME_REAL *hVals, int basisSize, int numConverged, size_t *lrwork,
   SCALAR *rwork, int liwork, int *iwork, primme_params *primme) {

   int i, j; /* Loop variables    */
   int info; /* dsyev error value */
   int index;
   int *permu, *permw;
   double targetShift;
#ifdef USE_COMPLEX
   PRIMME_REAL  *doubleWork;
#endif

   /* Some LAPACK implementations don't like zero-size matrices */
   if (basisSize == 0) return 0;

   /* Return memory requirements */
   if (H == NULL) {
      SCALAR rwork0;
      CHKERR((Num_heev_Sprimme("V", "U", basisSize, hVecs, basisSize, hVals,
               &rwork0, -1, &info), info), -1);
      *lrwork = max(*lrwork, (size_t)REAL_PART(rwork0));
      return 0;
   }

   /* ---------------------- */
   /* Divide the iwork space */
   /* ---------------------- */
   assert(liwork >= 2*basisSize);
   permu  = iwork;
   permw = permu + basisSize;


   /* ------------------------------------------------------------------- */
   /* Copy the upper triangular portion of H into hvecs.  We need to do   */
   /* this since DSYEV overwrites the input matrix with the eigenvectors. */  
   /* Note that H is maxBasisSize-by-maxBasisSize and the basisSize-by-   */
   /* basisSize submatrix of H is copied into hvecs.                      */
   /* ------------------------------------------------------------------- */

   if (primme->target != primme_largest) {
      for (j=0; j < basisSize; j++) {
	    #ifdef __NVCC__
	    if(primme!=NULL && primme->data_in_gpu == 1){
		  #ifdef USE_DOUBLE
      	          magma_dcopy(j+1,(magmaDouble_const_ptr)(H+ldH*j),1,(magmaDouble_ptr)(hVecs+ldhVecs*j),1,primme->queue);
   		  #elif USE_DOUBLECOMPLEX
		  magma_zcopy(j+1,(magmaDoubleComplex_const_ptr)(H+ldH*j),1,(magmaDoubleComplex_ptr)(hVecs+ldhVecs*j),1,primme->queue);
	 	  #elif USE_FLOAT
		  magma_scopy(j+1,(magmaFloat_const_ptr)(H+ldH*j),1,(magmaFloat_ptr)(hVecs+ldhVecs*j),1,primme->queue);
                  #elif USE_FLOATCOMPLEX
	 	  magma_ccopy(j+1,(magmaFloatComplex_const_ptr)(H+ldH*j),1,(magmaFloatComplex_ptr)(hVecs+ldhVecs*j),1,primme->queue);
	 	  #endif
	    }else
	    #endif

         for (i=0; i <= j; i++) { 
            hVecs[ldhVecs*j+i] = H[ldH*j+i];
         }
      }      
   }
   else { /* (primme->target == primme_largest) */
      for (j=0; j < basisSize; j++) {
		 #ifdef __NVCC__
	      if(primme!=NULL && primme->data_in_gpu == 1){
			  #ifdef USE_DOUBLE
		 	  magma_dcopy(j+1,(magmaDouble_const_ptr)H+ldH*j,1,(magmaDouble_ptr)hVecs+ldhVecs*j,1,primme->queue);
			  magma_dscal(j+1,-1,(magmaDouble_ptr)hVecs+ldhVecs*j,1,primme->queue);
	   		  #elif USE_DOUBLECOMPLEX
			  magma_zcopy(j+1,(magmaDoubleComplex_const_ptr)H+ldH*j,1,(magmaDoubleComplex_ptr)hVecs+ldhVecs*j,1,primme->queue);
			  magma_zscal(j+1,MAGMA_Z_MAKE(-1,0),(magmaDoubleComplex_ptr)hVecs+ldhVecs*j,1,primme->queue);
		 	  #elif USE_FLOAT
			  magma_scopy(j+1,(magmaFloat_const_ptr)H+ldH*j,1,(magmaFloat_ptr)hVecs+ldhVecs*j,1,primme->queue);
			  magma_sscal(j+1,-1,(magmaFloat_ptr)hVecs+ldhVecs*j,1,primme->queue);
		          #elif USE_FLOATCOMPLEX
		 	  magma_ccopy(j+1,(magmaFloatComplex_const_ptr)H+ldH*j,1,(magmaFloatComplex_ptr)hVecs+ldhVecs*j,1,primme->queue);
			  magma_cscal(j+1,MAGMA_C_MAKE(-1,0),(magmaFloatComplex_ptr)hVecs+ldhVecs*j,1,primme->queue);
		 	  #endif
			  	

	    }else
	    #endif

         for (i=0; i <= j; i++) { 
            hVecs[ldhVecs*j+i] = -H[ldH*j+i];
         }
      }
   }

  #ifdef __NVCC__
  if(primme!=NULL && primme->data_in_gpu==1){
	primme->data_in_gpu=0;

        SCALAR *cpu_hVecs = malloc(ldhVecs*primme->maxBasisSize*sizeof(SCALAR)); 
	SCALAR *cpu_rwork = malloc(*lrwork*sizeof(SCALAR));//primme->cpu_rwork;
     
	magma_getvector(ldhVecs*primme->maxBasisSize,sizeof(SCALAR),hVecs,1,cpu_hVecs,1,primme->queue);

	CHKERR((Num_heev_Sprimme("V", "U", basisSize, cpu_hVecs, ldhVecs, hVals, cpu_rwork, 
                TO_INT(*lrwork), &info), info), -1);
	
	
	primme->data_in_gpu=1;

	magma_setvector(ldhVecs*primme->maxBasisSize,sizeof(SCALAR),cpu_hVecs,1,hVecs,1,primme->queue);

	free(cpu_hVecs);
	free(cpu_rwork);

  }else
  #endif
  CHKERR((Num_heev_Sprimme("V", "U", basisSize, hVecs, ldhVecs, hVals, rwork, 
                TO_INT(*lrwork), &info), info), -1);
   /* ---------------------------------------------------------------------- */
   /* ORDER the eigenvalues and their eigenvectors according to the desired  */
   /* target:  smallest/Largest or interior closest abs/leq/geq to a shift   */
   /* ---------------------------------------------------------------------- */

   if (primme->target == primme_smallest) 
      return 0;

   if (primme->target == primme_largest) {

	 #if 0 
def __NVCC__
	 if(primme!=NULL && primme->data_in_gpu==1){
		#if defined(USE_DOUBLE) || defined(USE_DOUBLECOMPLEX)
		magma_dscal(basisSize,-1,hVals,1,primme->queue);
		#else
		magma_sscal(basisSize,-1,hVals,1,primme->queue);
		#endif	 
	 }else
	 #endif
      for (i = 0; i < basisSize; i++) {
         hVals[i] = -hVals[i];
      }
   }

   else { 
      /* ---------------------------------------------------------------- */
      /* Select the interior shift. Use the first unlocked shift, and not */
      /* higher ones, even if some eigenpairs in the basis are converged. */
      /* Then order the ritz values based on the closeness to the shift   */
      /* from the left, from right, or in absolute value terms            */
      /* ---------------------------------------------------------------- */

      /* TODO: order properly when numTargetShifts > 1 */

	 /* this part to be done in cpu */

      targetShift = 
        primme->targetShifts[min(primme->numTargetShifts-1, numConverged)];

      if (primme->target == primme_closest_geq) {
   
         /* ---------------------------------------------------------------- */
         /* find hVal closest to the right of targetShift, i.e., closest_geq */
         /* ---------------------------------------------------------------- */
         for (j=0;j<basisSize;j++) 
              if (hVals[j]>=targetShift) break;
           
         /* figure out this ordering */
         index = 0;
   
         for (i=j; i<basisSize; i++) {
            permu[index++]=i;
         }
         for (i=0; i<j; i++) {
            permu[index++]=i;
         }
      }
      else if (primme->target == primme_closest_leq) {
         /* ---------------------------------------------------------------- */
         /* find hVal closest_leq to targetShift                             */
         /* ---------------------------------------------------------------- */
         for (j=basisSize-1; j>=0 ;j--) 
             if (hVals[j]<=targetShift) break;
           
         /* figure out this ordering */
         index = 0;
   
         for (i=j; i>=0; i--) {
            permu[index++]=i;
         }
         for (i=basisSize-1; i>j; i--) {
            permu[index++]=i;
         }
      }
      else if (primme->target == primme_closest_abs) {

         /* ---------------------------------------------------------------- */
         /* find hVal closest but geq than targetShift                       */
         /* ---------------------------------------------------------------- */
         for (j=0;j<basisSize;j++) 
             if (hVals[j]>=targetShift) break;

         i = j-1;
         index = 0;
         while (i>=0 && j<basisSize) {
            if (fabs(hVals[i]-targetShift) < fabs(hVals[j]-targetShift)) 
               permu[index++] = i--;
            else 
               permu[index++] = j++;
         }
         if (i<0) {
            for (i=j;i<basisSize;i++) 
                    permu[index++] = i;
         }
         else if (j>=basisSize) {
            for (j=i;j>=0;j--)
                    permu[index++] = j;
         }
      }
      else if (primme->target == primme_largest_abs) {

         j = 0;
         i = basisSize-1;
         index = 0;
         while (i>=j) {
            if (fabs(hVals[i]-targetShift) > fabs(hVals[j]-targetShift)) 
               permu[index++] = i--;
            else 
               permu[index++] = j++;
         }

      }

      #ifdef __NVCC__
      if(primme!=NULL && primme->data_in_gpu==1){
		 //hVals = gpu_hVals;
		 SCALAR *cpu_rwork = malloc(*lrwork*sizeof(SCALAR));
		 permute_vecs_Rprimme(hVals, 1, basisSize, 1, permu, (PRIMME_REAL*)cpu_rwork, permw);
     	         permuteGPU_vecs_Sprimme(hVecs, basisSize, basisSize, ldhVecs, permu, rwork, permw, primme);
		 free(cpu_rwork);
      }else{
      #endif
      /* ---------------------------------------------------------------- */
      /* Reorder hVals and hVecs according to the permutation             */
      /* ---------------------------------------------------------------- */
      permute_vecs_Rprimme(hVals, 1, basisSize, 1, permu, (PRIMME_REAL*)rwork, permw);
      permute_vecs_Sprimme(hVecs, basisSize, basisSize, ldhVecs, permu, rwork, permw);

      #ifdef __NVCC__
      }
      #endif
   }


   return 0;   
}

/*******************************************************************************
 * Subroutine solve_H_Harm - This procedure implements the harmonic extraction
 *    in a novelty way. In standard harmonic the next eigenproblem is solved:
 *       V'*(A-s*I)'*(A-s*I)*V*X = V'*(A-s*I)'*V*X*L,
 *    where (L_{i,i},X_i) are the harmonic-Ritz pairs. In practice, it is
 *    computed (A-s*I)*V = Q*R and it is solved instead:
 *       R*X = Q'*V*X*L,
 *    which is a generalized non-Hermitian problem. Instead of dealing with
 *    complex solutions, which are unnatural in context of Hermitian problems,
 *    we propose the following. Note that,
 *       (A-s*I)*V = Q*R -> Q'*V*inv(R) = Q'*inv(A-s*I)*Q.
 *    And note that Q'*V*inv(R) is Hermitian if A is, and also that
 *       Q'*V*inv(R)*Y = Y*inv(L) ->  Q'*V*X*L = R*X,
 *    with Y = R*X. So this routine computes X by solving the Hermitian problem
 *    Q'*V*inv(R).
 *        
 * INPUT ARRAYS AND PARAMETERS
 * ---------------------------
 * H             The matrix V'*A*V
 * ldH           The leading dimension of H
 * R             The R factor for the QR decomposition of (A - target*I)*V
 * ldR           The leading dimension of R
 * basisSize     Current size of the orthonormal basis V
 * lrwork        Length of the work array rwork
 * primme        Structure containing various solver parameters
 * 
 * INPUT/OUTPUT ARRAYS
 * -------------------
 * hVecs         The orthogonal basis of inv(R) * eigenvectors of QtV/R
 * ldhVecs       The leading dimension of hVecs
 * hU            The eigenvectors of QtV/R
 * ldhU          The leading dimension of hU
 * hVals         The Ritz values of the vectors in hVecs
 * rwork         Workspace
 *
 * Return Value
 * ------------
 * int -  0 upon successful return
 *     - -1 Num_dsyev/zheev was unsuccessful
 ******************************************************************************/

static int solve_H_Harm_Sprimme(SCALAR *H, int ldH, SCALAR *QtV, int ldQtV,
   SCALAR *R, int ldR, SCALAR *hVecs, int ldhVecs, SCALAR *hU, int ldhU,
   PRIMME_REAL *hVals, int basisSize, int numConverged, double machEps,
   size_t *lrwork, SCALAR *rwork, int liwork, int *iwork,
   primme_params *primme) {

   int i, ret;
   double *oldTargetShifts, zero=0.0;
   primme_target oldTarget;

   (void)numConverged; /* unused parameter */

   /* Some LAPACK implementations don't like zero-size matrices */
   if (basisSize == 0) return 0;

   /* Return memory requirements */
   if (QtV == NULL) {
      CHKERR(solve_H_RR_Sprimme(QtV, ldQtV, hVecs, ldhVecs, hVals, basisSize,
         0, lrwork, rwork, liwork, iwork, primme), -1);
      return 0;
   }

   /* QAQ = QtV*inv(R) */
   #ifdef __NVCC__
   if(primme!=NULL && primme->data_in_gpu==1){
       NumGPU_copy_matrix_Sprimme(QtV, basisSize, basisSize, ldQtV, hVecs, ldhVecs, primme);
       NumGPU_trsm_Sprimme("R", "U", "N", "N", basisSize, basisSize, 1.0, R, ldR,
         hVecs, ldhVecs, primme);

   }else{
   #endif
   Num_copy_matrix_Sprimme(QtV, basisSize, basisSize, ldQtV, hVecs, ldhVecs);
   Num_trsm_Sprimme("R", "U", "N", "N", basisSize, basisSize, 1.0, R, ldR,
         hVecs, ldhVecs);

   #ifdef __NVCC__
   }
   #endif

   /* Compute eigenpairs of QAQ */

   oldTargetShifts = primme->targetShifts;
   oldTarget = primme->target;
   primme->targetShifts = &zero;
   switch(primme->target) {
      case primme_closest_geq:
         primme->target = primme_largest;
         break;
      case primme_closest_leq:
         primme->target = primme_smallest;
         break;
      case primme_closest_abs:
         primme->target = primme_largest_abs;
         break;
      default:
         assert(0);
   }
   ret = solve_H_RR_Sprimme(hVecs, ldhVecs, hVecs, ldhVecs, hVals,
         basisSize, 0, lrwork, rwork, liwork, iwork, primme);
   primme->targetShifts = oldTargetShifts;
   primme->target = oldTarget;
   CHKERRM(ret, -1, "Error calling solve_H_RR_Sprimme\n");


   #ifdef __NVCC__
   if(primme!=NULL && primme->data_in_gpu==1){
     NumGPU_copy_matrix_Sprimme(hVecs, basisSize, basisSize, ldhVecs, hU, ldhU,primme);
     NumGPU_trsm_Sprimme("L", "U", "N", "N", basisSize, basisSize, 1.0, R, ldR,
         hVecs, ldhVecs,primme);


     CHKERR(ortho_Sprimme(hVecs, ldhVecs, NULL, 0, 0, basisSize-1, NULL, 0, 0,
         basisSize, primme->iseed, machEps, rwork, lrwork, primme), -1);
 
     /* Compute Rayleigh quotient lambda_i = x_i'*H*x_i */

     NumGPU_hemm_Sprimme("L", "U", basisSize, basisSize, 1.0, H,
        ldH, hVecs, ldhVecs, 0.0, rwork, basisSize,primme);


	//PRIMME_REAL *cpu_hVals = primme->cpu_hVals;
     for (i=0; i<basisSize; i++) {
	   hVals[i] =
	      REAL_PART(NumGPU_dot_Sprimme(basisSize, &hVecs[ldhVecs*i], 1,
	               &rwork[basisSize*i], 1,primme));


	//magma_setvector(basisSize,sizeof(PRIMME_REAL),cpu_hVals,1,hVals,1,primme->queue);
   }


   }else{
   #endif
   Num_copy_matrix_Sprimme(hVecs, basisSize, basisSize, ldhVecs, hU, ldhU);

   /* Transfer back the eigenvectors to V, hVecs = R\hVecs */

   Num_trsm_Sprimme("L", "U", "N", "N", basisSize, basisSize, 1.0, R, ldR,
         hVecs, ldhVecs);

   CHKERR(ortho_Sprimme(hVecs, ldhVecs, NULL, 0, 0, basisSize-1, NULL, 0, 0,
         basisSize, primme->iseed, machEps, rwork, lrwork, primme), -1);
 
   /* Compute Rayleigh quotient lambda_i = x_i'*H*x_i */

   Num_hemm_Sprimme("L", "U", basisSize, basisSize, 1.0, H,
      ldH, hVecs, ldhVecs, 0.0, rwork, basisSize);

   for (i=0; i<basisSize; i++) {
      hVals[i] =
         REAL_PART(Num_dot_Sprimme(basisSize, &hVecs[ldhVecs*i], 1,
                  &rwork[basisSize*i], 1));
   }
   #ifdef __NVCC__
   }
   #endif
   return 0;
}

/*******************************************************************************
 * Subroutine solve_H_Ref - This procedure solves the singular value
 *            decomposition of matrix R
 *        
 * INPUT ARRAYS AND PARAMETERS
 * ---------------------------
 * H             The matrix V'*A*V
 * ldH           The leading dimension of H
 * R             The R factor for the QR decomposition of (A - target*I)*V
 * ldR           The leading dimension of R
 * basisSize     Current size of the orthonormal basis V
 * lrwork        Length of the work array rwork
 * primme        Structure containing various solver parameters
 * 
 * INPUT/OUTPUT ARRAYS
 * -------------------
 * hVecs         The right singular vectors of R
 * ldhVecs       The leading dimension of hVecs
 * hU            The left singular vectors of R
 * ldhU          The leading dimension of hU
 * hSVals        The singular values of R
 * hVals         The Ritz values of the vectors in hVecs
 * rwork         Workspace
 *
 * Return Value
 * ------------
 * int -  0 upon successful return
 *     - -1 was unsuccessful
 ******************************************************************************/

static int solve_H_Ref_Sprimme(SCALAR *H, int ldH, SCALAR *hVecs,
   int ldhVecs, SCALAR *hU, int ldhU, PRIMME_REAL *hSVals, SCALAR *R, int ldR,
   PRIMME_REAL *hVals, int basisSize, int targetShiftIndex, size_t *lrwork,
   SCALAR *rwork, int liwork, int *iwork, primme_params *primme) {
   int i, j; /* Loop variables    */
   int info; /* error value */

   (void)targetShiftIndex; /* unused parameter */

   /* Some LAPACK implementations don't like zero-size matrices */
   if (basisSize == 0) return 0;

   /* Return memory requirements */
   if (H == NULL) {
      SCALAR rwork0;
      size_t lrwork0 = 0;
#ifdef USE_COMPLEX
      lrwork0 = (size_t)(3*basisSize);
      CHKERR((Num_gesvd_Sprimme("S", "O", basisSize, basisSize, R, basisSize,
            NULL, NULL, basisSize, hVecs, basisSize, &rwork0,
            -1, hVals, &info), info), -1);
#else
      CHKERR((Num_gesvd_Sprimme("S", "O", basisSize, basisSize, R, basisSize, 
            NULL, NULL, basisSize, hVecs, basisSize, &rwork0, -1, &info), info),
            -1);
#endif
      lrwork0 += (size_t)REAL_PART(rwork0);
      lrwork0 += (size_t)basisSize*(size_t)basisSize; /* aux for transpose V and hemm */
      *lrwork = max(*lrwork, lrwork0);
      /* for perm and permute_vecs */
      *iwork = max(*iwork, 2*basisSize);
      return 0;
   }


 
   #ifdef __NVCC__
   SCALAR *cpu_H;
   SCALAR *cpu_hVecs;
   SCALAR *cpu_hU;
   //PRIMME_REAL *cpu_hSVals;
   SCALAR *cpu_R;
   //PRIMME_REAL *cpu_hVals;
   SCALAR *cpu_rwork;
  

   if(primme!=NULL && primme->data_in_gpu==1){
     cpu_H = malloc(ldH*basisSize*sizeof(SCALAR));//primme->cpu_H;
     cpu_hVecs = malloc(ldhVecs*basisSize*sizeof(SCALAR));//primme->cpu_hVecs;
     cpu_hU = malloc(ldhU*basisSize*sizeof(SCALAR));//primme->cpu_hU;
     cpu_R = malloc(ldR*basisSize*sizeof(SCALAR));//primme->cpu_R;
     cpu_rwork = malloc(*lrwork*sizeof(SCALAR));//primme->cpu_rwork;

   } 

   #endif


   /* Copy R into hVecs */
   #ifdef __NVCC__
   if(primme!=NULL && primme->data_in_gpu==1){
	  NumGPU_copy_matrix_Sprimme(R, basisSize, basisSize, ldR, hVecs, ldhVecs, primme);
   }else
   #endif
   Num_copy_matrix_Sprimme(R, basisSize, basisSize, ldR, hVecs, ldhVecs);

   /* Note gesvd returns transpose(V) rather than V and sorted in descending  */
   /* order of the singular values                                            */

#ifdef USE_COMPLEX
   /* zgesvd requires 5*basisSize double work space; booked 3*basisSize complex double */
   assert(*lrwork >= (size_t)(3*basisSize));

   #ifdef __NVCC__
   if(primme!=NULL && primme->data_in_gpu==1){
	 // magma_getvector(primme->realWorkSize,1,primme->gpu_realWork,1,primme->cpu_realWork,1,primme->queue);
	 magma_getvector_async(ldhVecs*basisSize,sizeof(SCALAR),hVecs,1,cpu_hVecs,1,primme->queue);
	 //magma_getvector_async(basisSize,sizeof(PRIMME_REAL),hSVals,1,cpu_hSVals,1,primme->queue);
	 magma_getvector(ldhU*basisSize,sizeof(SCALAR),hU,1,cpu_hU,1,primme->queue);
	 //magma_getvector(*lrwork,sizeof(SCALAR),rwork,1,cpu_rwork,1,primme->queue);

       CHKERR((NumGPU_gesvd_Sprimme("S", "O", basisSize, basisSize, cpu_hVecs, ldhVecs,
           hSVals, cpu_hU, ldhU, cpu_hVecs, ldhVecs, cpu_rwork+3*basisSize,
           TO_INT(*lrwork-(size_t)(3*basisSize)), (PRIMME_REAL*)cpu_rwork, &info,primme), info),
           -1);

	 magma_setvector_async(ldhVecs*basisSize,sizeof(SCALAR),cpu_hVecs,1,hVecs,1,primme->queue);
	 //magma_setvector_async(basisSize,sizeof(PRIMME_REAL),cpu_hSVals,1,hSVals,1,primme->queue);
	 magma_setvector(ldhU*basisSize,sizeof(SCALAR),cpu_hU,1,hU,1,primme->queue);
	 //magma_setvector(*lrwork,sizeof(SCALAR),cpu_rwork,1,rwork,1,primme->queue);
	 //magma_setvector(primme->realWorkSize,1,primme->cpu_realWork,1,primme->gpu_realWork,1,primme->queue);

   }else
   #endif
   CHKERR((Num_gesvd_Sprimme("S", "O", basisSize, basisSize, hVecs, ldhVecs,
         hSVals, hU, ldhU, hVecs, ldhVecs, rwork+3*basisSize,
         TO_INT(*lrwork-(size_t)(3*basisSize)), (PRIMME_REAL*)rwork, &info), info),
         -1);
#else

   
   #ifdef __NVCC__
   if(primme!=NULL && primme->data_in_gpu==1){
	  //magma_getvector(primme->realWorkSize,1,primme->gpu_realWork,1,primme->cpu_realWork,1,primme->queue);
	
      magma_getvector_async(ldhVecs*basisSize,sizeof(SCALAR),hVecs,1,cpu_hVecs,1,primme->queue);
	 //magma_getvector_async(basisSize,sizeof(PRIMME_REAL),hSVals,1,cpu_hSVals,1,primme->queue);
	 magma_getvector(ldhU*basisSize,sizeof(SCALAR),hU,1,cpu_hU,1,primme->queue);
	 //magma_getvector(*lrwork,sizeof(SCALAR),rwork,1,cpu_rwork,1,primme->queue);

	  CHKERR((NumGPU_gesvd_Sprimme("S", "O", basisSize, basisSize, cpu_hVecs, ldhVecs,
           hSVals, cpu_hU, ldhU, cpu_hVecs, ldhVecs, cpu_rwork, TO_INT(*lrwork), &info, primme),
           info), -1);	 

	 magma_setvector_async(ldhVecs*basisSize,sizeof(SCALAR),cpu_hVecs,1,hVecs,1,primme->queue);
	 //magma_setvector_async(basisSize,sizeof(PRIMME_REAL),cpu_hSVals,1,hSVals,1,primme->queue);
	 magma_setvector(ldhU*basisSize,sizeof(SCALAR),cpu_hU,1,hU,1,primme->queue);
	 //magma_setvector(*lrwork,sizeof(SCALAR),cpu_rwork,1,rwork,1,primme->queue);
	 //magma_setvector(primme->realWorkSize,1,primme->cpu_realWork,1,primme->gpu_realWork,1,primme->queue);
   }else
   #endif
   CHKERR((Num_gesvd_Sprimme("S", "O", basisSize, basisSize, hVecs, ldhVecs,
         hSVals, hU, ldhU, hVecs, ldhVecs, rwork, TO_INT(*lrwork), &info),
         info), -1);
#endif

   /* Transpose back V */

   assert(*lrwork >= (size_t)basisSize*(size_t)basisSize);
   #ifdef __NVCC__
   if(primme!=NULL && primme->data_in_gpu==1){
	 #ifdef USE_DOUBLE
	 magmablas_dtranspose_inplace(basisSize,hVecs,ldhVecs,primme->queue);		
	 #elif USE_DOUBLECOMPLEX
      magmablas_ztranspose_conj_inplace(basisSize,(magmaDoubleComplex_ptr)hVecs,ldhVecs,primme->queue);
      #elif USE_FLOAT	
	 magmablas_stranspose_inplace(basisSize,hVecs,ldhVecs,primme->queue);
      #elif USE_FLOATCOMPLEX
      magmablas_ctranspose_conj_inplace(basisSize,(magmaFloatComplex_ptr)hVecs,ldhVecs,primme->queue);
      #endif
   }else{
   #endif
   for (j=0; j < basisSize; j++) {
      for (i=0; i < basisSize; i++) { 
         rwork[basisSize*j+i] = CONJ(hVecs[ldhVecs*i+j]);
      }
   }
   Num_copy_matrix_Sprimme(rwork, basisSize, basisSize, basisSize, hVecs, ldhVecs);

   #ifdef __NVCC__
   }
   #endif
   /* Rearrange V, hSVals and hU in ascending order of singular value   */
   /* if target is not largest abs.                                     */

   if (primme->target == primme_closest_abs 
         || primme->target == primme_closest_leq
         || primme->target == primme_closest_geq) {
      int *perm = iwork;
      int *iwork0 = iwork + basisSize;
      assert(liwork >= 2*basisSize);

      for (i=0; i<basisSize; i++) perm[i] = basisSize-1-i;



	 #ifdef __NVCC__
	 if(primme!=NULL && primme->data_in_gpu==1){
		
		permute_vecs_Rprimme(hSVals, 1, basisSize, 1, perm, (PRIMME_REAL*)cpu_rwork, iwork0);
	     permuteGPU_vecs_Sprimme(hVecs, basisSize, basisSize, ldhVecs, perm, rwork, iwork0, primme);
	     permuteGPU_vecs_Sprimme(hU, basisSize, basisSize, ldhU, perm, rwork, iwork0, primme);


      }else{
      #endif
      permute_vecs_Rprimme(hSVals, 1, basisSize, 1, perm, (PRIMME_REAL*)rwork, iwork0);
      permute_vecs_Sprimme(hVecs, basisSize, basisSize, ldhVecs, perm, rwork, iwork0);
      permute_vecs_Sprimme(hU, basisSize, basisSize, ldhU, perm, rwork, iwork0);
	
      #ifdef __NVCC__
	 }
      #endif

   }

   /* compute Rayleigh quotient lambda_i = x_i'*H*x_i */
   #ifdef __NVCC__
   if(primme!=NULL && primme->data_in_gpu==1){
	 	
      //magma_getvector_async(ldH*basisSize,sizeof(SCALAR),H,1,cpu_H,1,primme->queue);
	 //magma_getvector(ldhVecs*basisSize,sizeof(SCALAR),hVecs,1,cpu_hVecs,1,primme->queue);
	 //magma_getvector(*lrwork,sizeof(SCALAR),rwork,1,cpu_rwork,1,primme->queue);

	 NumGPU_hemm_Sprimme("L", "U", basisSize, basisSize, 1.0, H,
            ldH, hVecs, ldhVecs, 0.0, rwork, basisSize, primme);

	 //magma_setvector_async(ldH*basisSize,sizeof(SCALAR),cpu_H,1,H,1,primme->queue);
	 //magma_setvector(ldhVecs*basisSize,sizeof(SCALAR),cpu_hVecs,1,hVecs,1,primme->queue);
	 //magma_setvector(*lrwork,sizeof(SCALAR),cpu_rwork,1,rwork,1,primme->queue);
	
      for (i=0; i<basisSize; i++) {
         hVals[i] = REAL_PART(NumGPU_dot_Sprimme(basisSize, &hVecs[ldhVecs*i], 1,
                  &rwork[basisSize*i], 1, primme));
       }

	  //magma_setvector(basisSize,sizeof(PRIMME_REAL),cpu_hVals,1,hVals,1,primme->queue);
      //magma_setvector(primme->realWorkSize,1,primme->cpu_realWork,1,primme->gpu_realWork,1,primme->queue);

   }else{
   #endif

   Num_hemm_Sprimme("L", "U", basisSize, basisSize, 1.0, H,
      ldH, hVecs, ldhVecs, 0.0, rwork, basisSize);

   for (i=0; i<basisSize; i++) {
      hVals[i] = REAL_PART(Num_dot_Sprimme(basisSize, &hVecs[ldhVecs*i], 1,
               &rwork[basisSize*i], 1));
   }

   #ifdef __NVCC__
   }
   #endif


 
   #ifdef __NVCC__
   if(primme!=NULL && primme->data_in_gpu==1){
     free(cpu_H);
     free(cpu_hVecs);
     free(cpu_hU);
     free(cpu_R);
     free(cpu_rwork);

   } 

   #endif


   return 0;
}
/*******************************************************************************
 * Subroutine solve_H_brcast - This procedure broadcast the solution of the
 *       projected problem (hVals, hSVals, hVecs, hU) from process 0 to the rest.
 *
 * NOTE: the optimal implementation will use an user-defined broadcast function.
 *       To ensure backward compatibility, we used globalSum instead.
 * 
 * INPUT ARRAYS AND PARAMETERS
 * ---------------------------
 * basisSize      The dimension of hVecs, hU, hVals and hSVals
 * lrwork         Length of the work array rwork
 * primme         Structure containing various solver parameters
 * 
 * INPUT/OUTPUT ARRAYS
 * -------------------
 * hU             The left singular vectors of R or the eigenvectors of QtV/R
 * ldhU           The leading dimension of hU
 * hVecs          The coefficient vectors such as V*hVecs will be the Ritz vectors
 * ldhVecs        The leading dimension of hVecs
 * hVals          The Ritz values
 * hSVals         The singular values of R
 * rwork          Workspace
 *
 * Return Value
 * ------------
 * error code                          
 ******************************************************************************/

static int solve_H_brcast_Sprimme(int basisSize, SCALAR *hU, int ldhU,
      SCALAR *hVecs, int ldhVecs, PRIMME_REAL *hVals, PRIMME_REAL *hSVals, size_t *lrwork,
      SCALAR *rwork, primme_params *primme) {

   int n=0;                            /* number of SCALAR packed */
   SCALAR *rwork0 = rwork;             /* next SCALAR free */
   const size_t c = sizeof(SCALAR)/sizeof(PRIMME_REAL);

   /* Return memory requirements */

   if (hVecs == NULL) {
      switch (primme->projectionParams.projection) {
         case primme_proj_RR:
            /* Broadcast hVecs, hVals */
            *lrwork = max(*lrwork, (size_t)2*basisSize*(basisSize+1));
            break;

         case primme_proj_harmonic:
            /* Broadcast hVecs, hVals, hU */
            *lrwork = max(*lrwork, (size_t)2*basisSize*(2*basisSize+1));
            break;

         case primme_proj_refined:
            /* Broadcast hVecs, hVals, hU, hSVals */
            *lrwork = max(*lrwork, (size_t)2*basisSize*(2*basisSize+2));
            break;

         default:
            assert(0);
      }
      return 0;
   }
   #ifdef __NVCC__
//   if(primme==NULL || primme->data_in_gpu!=1)
   #endif
   assert(*lrwork >= (size_t)2*basisSize*((hU?2:1)*basisSize + (hSVals?2:1)));

   /* Pack hVecs */

   if (primme->procID == 0) {
	 #ifdef __NVCC__
	 if(primme!=NULL && primme->data_in_gpu==1){
//	    printf("Before\n");
/*	    SCALAR *cpu_hVecs = malloc(basisSize*ldhVecs*sizeof(SCALAR));
	    SCALAR *cpu_rwork0 = malloc(basisSize*basisSize*sizeof(SCALAR));

	    magma_getvector(basisSize*ldhVecs,sizeof(SCALAR),hVecs,1,cpu_hVecs,1,primme->queue);
	    magma_getvector(basisSize*basisSize,sizeof(SCALAR),rwork0,1,cpu_rwork0,1,primme->queue);

	    Num_copy_matrix_Sprimme(cpu_hVecs, basisSize, basisSize, ldhVecs, cpu_rwork0,
            basisSize);

	    magma_setvector(basisSize*ldhVecs,sizeof(SCALAR),cpu_hVecs,1,hVecs,1,primme->queue);
	    magma_setvector(basisSize*basisSize,sizeof(SCALAR),cpu_rwork0,1,rwork0,1,primme->queue);
*/
	    NumGPU_copy_matrix_Sprimme(hVecs, basisSize, basisSize, ldhVecs, rwork0,
            basisSize, primme);
//printf("After\n");
	 }else
	 #endif
      Num_copy_matrix_Sprimme(hVecs, basisSize, basisSize, ldhVecs, rwork0,
            basisSize);
   }
   n += basisSize*basisSize;
   rwork0 += basisSize*basisSize;

   /* Pack hU */

   if (hU) {
      if (primme->procID == 0) {
	    #ifdef __NVCC__
	    if(primme!=NULL && primme->data_in_gpu==1){
		  NumGPU_copy_matrix_Sprimme(hU, basisSize, basisSize, ldhU, rwork0,
               basisSize,primme);
         }else
	    #endif
         Num_copy_matrix_Sprimme(hU, basisSize, basisSize, ldhU, rwork0,
               basisSize);
      }
      n += basisSize*basisSize;
      rwork0 += basisSize*basisSize;
   }

   /* Pack hVals */

   if (primme->procID == 0) {
	 
	 #ifdef __NVCC__
      if(primme!=NULL && primme->data_in_gpu==1){
          //SCALAR tmpzero=0.0;

		//magma_setvector(1,sizeof(SCALAR),&tmpzero,1,rwork0+(basisSize/c),1,primme->queue);

		#ifdef USE_DOUBLE
		magma_dscal(1,0,rwork0+basisSize/c,1,primme->queue);
		#elif USE_DOUBLECOMPLEX
		magma_zscal(1,MAGMA_Z_ZERO,rwork0+basisSize/c,1,primme->queue);
		#elif USE_FLOAT
		magma_sscal(1,0,rwork0+basisSize/c,1,primme->queue);
		#elif USE_FLOATCOMPLEX
		magma_cscal(1,MAGMA_C_ZERO,rwork0+basisSize/c,1,primme->queue);
		#endif
		
		magma_setvector(basisSize,sizeof(PRIMME_REAL),hVals,1,rwork0,1,primme->queue);
		//NumGPU_copy_matrix_Rprimme(hVals, basisSize, 1, basisSize, (PRIMME_REAL*)rwork0,
          //    basisSize,primme);
	
      }else{
      #endif
      rwork0[basisSize/c] = 0.0; /* When complex, avoid to reduce with an   */
                                   /* uninitialized value                     */
      Num_copy_matrix_Rprimme(hVals, basisSize, 1, basisSize, (PRIMME_REAL*)rwork0,
            basisSize);
	 #ifdef __NVCC__
      }
      #endif

   }
   n += (basisSize + c-1)/c;
   rwork0 += (basisSize + c-1)/c;

   /* Pack hSVals */

   if (hSVals) {
      if (primme->procID == 0) {
 		#ifdef __NVCC__
	   if(primme!=NULL && primme->data_in_gpu==1){
//          SCALAR tmpzero=0.0;

//		magma_setvector(1,sizeof(SCALAR),&tmpzero,1,rwork0+(basisSize/c),1,primme->queue);

		#ifdef USE_DOUBLE
		magma_dscal(1,0,rwork0+basisSize/c,1,primme->queue);
		#elif USE_DOUBLECOMPLEX
		magma_zscal(1,MAGMA_Z_ZERO,rwork0+basisSize/c,1,primme->queue);
		#elif USE_FLOAT
		magma_sscal(1,0,rwork0+basisSize/c,1,primme->queue);
		#elif USE_FLOATCOMPLEX
		magma_cscal(1,MAGMA_C_ZERO,rwork0+basisSize/c,1,primme->queue);
		#endif

		magma_setvector(basisSize,sizeof(PRIMME_REAL),hSVals,1,rwork0,1,primme->queue);
		//NumGPU_copy_matrix_Rprimme(hSVals, basisSize, 1, basisSize, (PRIMME_REAL*)rwork0,
          //    basisSize,primme);
	
         }else{
         #endif
      
         rwork0[basisSize/c] = 0.0; /* When complex, avoid to reduce with an*/
                                      /* uninitialized value                  */
         Num_copy_matrix_Rprimme(hSVals, basisSize, 1, basisSize, (PRIMME_REAL*)rwork0,
               basisSize);
	    #ifdef __NVCC__
 	    }
         #endif

      }
      n += (basisSize + c-1)/c;
      rwork0 += (basisSize + c-1)/c;
   }

   /* If this is not proc 0, zero the input rwork */

   if (primme->procID != 0) {
	 #ifdef __NVCC__
	 if(primme!=NULL && primme->data_in_gpu==1){
	    NumGPU_zero_matrix_Sprimme(rwork, n, 1, n,primme);
	 }else
      #endif
      Num_zero_matrix_Sprimme(rwork, n, 1, n);
   }
 
   /* Perform the broadcast by using a reduction */
   #if 0 
def __NVCC__
   if(primme!=NULL && primme->data_in_gpu==1){
	 SCALAR *cpu_rwork=malloc(*lrwork*sizeof(SCALAR));//primme->cpu_rwork;
	 SCALAR *cpu_rwork0 = (SCALAR*)malloc(n*sizeof(SCALAR));
	
	 magma_getvector_async(n,sizeof(SCALAR),rwork,1,cpu_rwork,1,primme->queue);
	 magma_getvector(n,sizeof(SCALAR),rwork0,1,cpu_rwork0,1,primme->queue);

	 primme->data_in_gpu=0;
	 CHKERR(globalSum_Sprimme(cpu_rwork, cpu_rwork0, n, primme), -1);
	 primme->data_in_gpu=1;
	 magma_setvector_async(n,sizeof(SCALAR),cpu_rwork,1,rwork,1,primme->queue);
	 magma_setvector(n,sizeof(SCALAR),cpu_rwork0,1,rwork0,1,primme->queue);

	

   }else
   #endif
   CHKERR(globalSum_Sprimme(rwork, rwork0, n, primme), -1);

   /* Unpack hVecs */

   #ifdef __NVCC__
   if(primme!=NULL && primme->data_in_gpu==1){
	  NumGPU_copy_matrix_Sprimme(rwork0, basisSize, basisSize, basisSize, hVecs,
         ldhVecs,primme);
   
   }else
   #endif
   Num_copy_matrix_Sprimme(rwork0, basisSize, basisSize, basisSize, hVecs,
         ldhVecs);
   rwork0 += basisSize*basisSize;

   /* Unpack hU */

   if (hU) {
	 #ifdef __NVCC__
	 if(primme!=NULL && primme->data_in_gpu==1){
	    NumGPU_copy_matrix_Sprimme(rwork0, basisSize, basisSize, basisSize, hU,
            ldhU,primme);
      }else
      #endif
      Num_copy_matrix_Sprimme(rwork0, basisSize, basisSize, basisSize, hU,
            ldhU);
      rwork0 += basisSize*basisSize;
   }

   /* Unpack hVals */

   #ifdef __NVCC__
   if(primme!=NULL && primme->data_in_gpu==1){

	  magma_getvector(basisSize,sizeof(PRIMME_REAL),rwork0,1,hVals,1,primme->queue);
//	  NumGPU_copy_matrix_Rprimme((PRIMME_REAL*)rwork0, basisSize, 1, basisSize, hVals,
//         basisSize,primme);
   
   }else
   #endif
   Num_copy_matrix_Rprimme((PRIMME_REAL*)rwork0, basisSize, 1, basisSize, hVals,
         basisSize);
   rwork0 += (basisSize + c-1)/c;

   /* Unpack hSVals */

   if (hSVals) {

      #ifdef __NVCC__
	 if(primme!=NULL && primme->data_in_gpu==1){

		magma_getvector(basisSize,sizeof(PRIMME_REAL),rwork0,1,hSVals,1,primme->queue);
//		NumGPU_copy_matrix_Rprimme((PRIMME_REAL*)rwork0, basisSize, 1, basisSize, hSVals,
//               basisSize,primme);
      
	 }else
      #endif
      Num_copy_matrix_Rprimme((PRIMME_REAL*)rwork0, basisSize, 1, basisSize, hSVals,
               basisSize);
      rwork0 += (basisSize + c-1)/c;
   }

   return 0;
}

/*******************************************************************************
 * Function prepare_vecs - This subroutine checks that the
 *    conditioning of the coefficient vectors are good enough to converge
 *    with the requested accuracy. For now refined extraction is the only one that
 *    may present problems: two similar singular values in the projected problem
 *    may correspond to distinct eigenvalues in the original problem. If that is
 *    the case, the singular vector may have components of both eigenvectors,
 *    which prevents the residual norm be lower than some degree. Don't dealing
 *    with this may lead into stagnation.
 *
 *    It is checked that the next upper bound about the angle of the right
 *    singular vector v of A and the right singular vector vtilde of A+E,
 *
 *      sin(v,vtilde) <= sqrt(2)*||E||/sval_gap <= sqrt(2)*||A||*machEps/sval_gap,
 *
 *    is less than the upper bound about the angle of exact eigenvector u and
 *    the approximate eigenvector utilde,
 *
 *      sin(u,utilde) <= ||r||/eval_gap <= ||A||*eps/eval_gap.
 *
 *    (see pp. 211 in Matrix Algorithms vol. 2 Eigensystems, G. W. Steward).
 *
 *    If the inequality doesn't hold, do Rayleigh-Ritz onto the subspace
 *    spanned by both vectors.
 *
 *    we have found cases where this is not enough or the performance improves
 *    if Rayleigh-Ritz is also done when the candidate vector has a small
 *    angle with the last vector in V and when the residual norm is larger than
 *    the singular value.
 *
 *    When only one side of the shift is targeted (primme_closest_leq/geq), we
 *    allow to take eigenvalue of the other side but close to the shift. In the
 *    current heuristic they shouldn't be farther than the smallest residual
 *    norm in the block. This heuristic obtained good results in solving the
 *    augmented problem with shifts from solving the normal equations.
 *
 * NOTE: this function assumes hSVals are arranged in increasing order.
 *
 * INPUT ARRAYS AND PARAMETERS
 * ---------------------------
 * basisSize    Projected problem size
 * i0           Index of the first pair to check
 * blockSize    Number of candidates wanted
 * H            The matrix V'*A*V
 * ldH          The leading dimension of H
 * hVals        The Ritz values
 * hSVals       The singular values of R
 * hVecs        The coefficient vectors
 * ldhVecs      The leading dimension of hVecs
 * targetShiftIndex The target shift used in (A - targetShift*B) = Q*R
 * arbitraryVecs The number of vectors modified (input/output)
 * smallestResNorm The smallest residual norm in the block
 * flags        Array indicating the convergence of the Ritz vectors
 * RRForAll     If false compute Rayleigh-Ritz only in clusters with
 *              candidates. If true, compute it in every cluster.
 * machEps      Machine precision
 * rworkSize    The length of rwork
 * rwork        Workspace
 * iwork        Integer workspace
 * primme       Structure containing various solver parameters
 *
 ******************************************************************************/

TEMPLATE_PLEASE
int prepare_vecs_Sprimme(int basisSize, int i0, int blockSize,
      SCALAR *H, int ldH, PRIMME_REAL *hVals, PRIMME_REAL *hSVals, SCALAR *hVecs,
      int ldhVecs, int targetShiftIndex, int *arbitraryVecs,
      double smallestResNorm, int *flags, int RRForAll, SCALAR *hVecsRot,
      int ldhVecsRot, double machEps, size_t *rworkSize, SCALAR *rwork,
      int iworkSize, int *iwork, primme_params *primme) {

   int i, j, k;         /* Loop indices */
   int candidates;      /* Number of eligible pairs */
   int someCandidate;   /* If there is an eligible pair in the cluster */
   double aNorm;

   /* Quick exit */

   if (primme->projectionParams.projection != primme_proj_refined
         || basisSize == 0) {
      return 0;
   }

   /* Return memory requirement */

   if (H == NULL) {
      size_t rworkSize0=0;
      CHKERR(compute_submatrix_Sprimme(NULL, basisSize, 0, NULL,
               basisSize, 0, NULL, 0, NULL, &rworkSize0), -1);
      CHKERR(solve_H_RR_Sprimme(NULL, 0, NULL, 0, NULL, basisSize, 0,
            &rworkSize0, NULL, 0, iwork, primme), -1);
      rworkSize0 += (size_t)basisSize*(size_t)basisSize; /* aH */
      *rworkSize = max(*rworkSize, rworkSize0);
      return 0;
   }

   /* Quick exit */

   if (blockSize == 0) {
      return 0;
   }

   /* Special case: If (basisSize+numLocked) is the entire space, */
   /* then everything should be converged. Just do RR with the    */
   /* entire space.                                               */
 
   if (basisSize + (primme->locking?primme->initSize:0) 
         + primme->numOrthoConst >= primme->n) {

      /* Compute and sort eigendecomposition aH*ahVecs = ahVecs*diag(hVals(j:i-1)) */
      CHKERR(solve_H_RR_Sprimme(H, ldH, hVecs, ldhVecs, hVals, basisSize,
            targetShiftIndex, rworkSize, rwork, iworkSize, iwork, primme), -1);

      *arbitraryVecs = 0;

      return 0;
   }

   aNorm = (primme->aNorm <= 0.0) ?
      primme->stats.estimateLargestSVal : primme->aNorm;

   for (candidates=0, i=min(*arbitraryVecs,basisSize), j=i0;
         j < basisSize && candidates < blockSize; ) {

      double ip;
      /* -------------------------------------------------------------------- */
      /* Count all eligible values (candidates) from j up to i.               */
      /* -------------------------------------------------------------------- */

      for ( ; j < i; j++)
         if (!flags || flags[j] == UNCONVERGED)
            candidates++;
     
      if (candidates >= blockSize) break;
 
      /* -------------------------------------------------------------------- */
      /* Find the first i-th vector i>j with enough good conditioning, ie.,   */
      /* the singular value is separated enough from the rest (see the header */
      /* comment in this function). Also check if there is an unconverged     */
      /* value in the block.                                                  */
      /* -------------------------------------------------------------------- */
     for (i=j+1, someCandidate=0, ip=0.0; i<basisSize; i++) {

         /* Check that this approximation:                                    */
         /* sin singular vector: max(hSVals)*machEps/(hSvals[i]-hSVals[i+1])  */
         /* is less than the next one:                                        */
         /* sin eigenvector    : aNorm*eps/(hVals[i]-hVals[i+1]).             */
         /* Also try to include enough coefficient vectors into the cluster   */
         /* so that the cluster is close the last included vectors into the   */
         /* basis.                                                            */
         /* TODO: check the angle with all vectors added to the basis in the  */
         /* previous iterations; for now only the last one is considered.     */
         /* NOTE: we don't want to check hVecs(end,i) just after restart, so  */
         /* we don't use the value when it is zero.                           */
        double minDiff = sqrt(2.0)*hSVals[basisSize-1]*machEps/
            (aNorm*primme->eps/fabs(hVals[i]-hVals[i-1]));
        double ip0;
	#ifdef __NVCC__
	if(primme!=NULL && primme->data_in_gpu==1){
	   SCALAR tmp;
	   magma_getvector(1,sizeof(SCALAR),hVecs+(i-1)*ldhVecs+basisSize-1,1,&tmp,1,primme->queue);

	   ip0 = ABS(tmp);
         
         }else
	 #endif
	 ip0 = ABS(hVecs[(i-1)*ldhVecs+basisSize-1]);
         double ip1 = ((ip += ip0*ip0) != 0.0) ? ip : HUGE_VAL;

         if (!flags || flags[i-1] == UNCONVERGED) someCandidate = 1;

         if (fabs(hSVals[i]-hSVals[i-1]) >= minDiff
               && (smallestResNorm >= HUGE_VAL
                  || sqrt(ip1) >= smallestResNorm/aNorm/3.16)) 
            break;


      }
      i = min(i, basisSize);

      /* ----------------------------------------------------------------- */
      /* If the cluster j:i-1 is larger than one vector and there is some  */
      /* unconverged pair in there, compute the approximate eigenvectors   */
      /* with Rayleigh-Ritz. If RRForAll do also this when there is no     */
      /* candidate in the cluster.                                         */
      /* ----------------------------------------------------------------- */

      if (i-j > 1 && (someCandidate || RRForAll)) {
         SCALAR *rwork0 = rwork, *aH, *ahVecs;
         size_t rworkSize0 = *rworkSize;
         int aBasisSize = i-j;
         aH = rwork0; rwork0 += aBasisSize*aBasisSize;
         assert(rworkSize0 >= (size_t)aBasisSize*(size_t)aBasisSize);
         rworkSize0 -= (size_t)aBasisSize*(size_t)aBasisSize;
         ahVecs = &hVecsRot[ldhVecsRot*j+j];

         /* Zero hVecsRot(:,arbitraryVecs:i-1) */
	    #ifdef __NVCC__
	    if(primme!=NULL && primme->data_in_gpu==1){
		    NumGPU_zero_matrix_Sprimme(&hVecsRot[ldhVecsRot*(*arbitraryVecs)],
		          primme->maxBasisSize, i-*arbitraryVecs, ldhVecsRot,primme);

		    #ifdef USE_DOUBLE
		    magmablas_dlaset(MagmaFull,i-*arbitraryVecs,1,1,1,
					 hVecsRot +ldhVecsRot*(*arbitraryVecs)+(*arbitraryVecs),ldhVecsRot+1,primme->queue);
		    #elif USE_DOUBLECOMPLEX
		    magmablas_zlaset(MagmaFull,i-*arbitraryVecs,1,MAGMA_Z_ONE,MAGMA_Z_ONE,
		 			 hVecsRot +ldhVecsRot*(*arbitraryVecs)+(*arbitraryVecs),ldhVecsRot+1,primme->queue);
	   	    #elif USE_FLOAT
		    magmablas_slaset(MagmaFull,i-*arbitraryVecs,1,1,1,
					 hVecsRot +ldhVecsRot*(*arbitraryVecs)+(*arbitraryVecs),ldhVecsRot+1,primme->queue);
		    #elif USE_FLOATCOMPLEX
		    magmablas_claset(MagmaFull,i-*arbitraryVecs,1,MAGMA_C_ONE,MAGMA_C_ONE,
		 			 hVecsRot +ldhVecsRot*(*arbitraryVecs)+(*arbitraryVecs),ldhVecsRot+1,primme->queue);
		    #endif

		    /* aH = hVecs(:,j:i-1)'*H*hVecs(:,j:i-1) */
		    computeGPU_submatrix_Sprimme(&hVecs[ldhVecs*j], aBasisSize,
		          ldhVecs, H, basisSize, ldH, aH, aBasisSize, rwork0,
		          &rworkSize0,primme);

		    /* Compute and sort eigendecomposition aH*ahVecs = ahVecs*diag(hVals(j:i-1)) */
		    CHKERR(solve_H_RR_Sprimme(aH, aBasisSize, ahVecs, ldhVecsRot,
		          &hVals[j], aBasisSize, targetShiftIndex, &rworkSize0, rwork0,
		          iworkSize, iwork, primme), -1);

		    /* hVecs(:,j:i-1) = hVecs(:,j:i-1)*ahVecs */
		    NumGPU_gemm_Sprimme("N", "N", basisSize, aBasisSize, aBasisSize,
		          1.0, &hVecs[ldhVecs*j], ldhVecs, ahVecs, ldhVecsRot, 0.0,
		          rwork0, basisSize, primme);
		    NumGPU_copy_matrix_Sprimme(rwork0, basisSize, aBasisSize, basisSize,
		          &hVecs[ldhVecs*j], ldhVecs, primme);


         }else{
	    #endif
         Num_zero_matrix_Sprimme(&hVecsRot[ldhVecsRot*(*arbitraryVecs)],
               primme->maxBasisSize, i-*arbitraryVecs, ldhVecsRot);

         /* hVecsRot(:,arbitraryVecs:i-1) = I */
         for (k=*arbitraryVecs; k<i; k++)
            hVecsRot[ldhVecsRot*k+k] = 1.0;
 
         /* aH = hVecs(:,j:i-1)'*H*hVecs(:,j:i-1) */
         compute_submatrix_Sprimme(&hVecs[ldhVecs*j], aBasisSize,
               ldhVecs, H, basisSize, ldH, aH, aBasisSize, rwork0,
               &rworkSize0);

         /* Compute and sort eigendecomposition aH*ahVecs = ahVecs*diag(hVals(j:i-1)) */
         CHKERR(solve_H_RR_Sprimme(aH, aBasisSize, ahVecs, ldhVecsRot,
               &hVals[j], aBasisSize, targetShiftIndex, &rworkSize0, rwork0,
               iworkSize, iwork, primme), -1);

         /* hVecs(:,j:i-1) = hVecs(:,j:i-1)*ahVecs */
         Num_gemm_Sprimme("N", "N", basisSize, aBasisSize, aBasisSize,
               1.0, &hVecs[ldhVecs*j], ldhVecs, ahVecs, ldhVecsRot, 0.0,
               rwork0, basisSize);
         Num_copy_matrix_Sprimme(rwork0, basisSize, aBasisSize, basisSize,
               &hVecs[ldhVecs*j], ldhVecs);

	    #ifdef __NVCC__
         }
         #endif
         /* Indicate that before i may not be singular vectors */
         *arbitraryVecs = i;

         /* Remove converged flags from j upto i */
         if (flags && !RRForAll) for (k=j; k<i; k++) flags[k] = UNCONVERGED;
      }
   }

   return 0;
}
