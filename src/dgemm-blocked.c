const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (const int lda, const int M, const int N, const int K, const double* const A, const double* const B, double* const __restrict__ C)
{
  /* For each row i of A */
  for (int i = 0; i < N; ++i) {
    /* For each column j of B */ 
    for (int j = 0; j < M; ++j) 
    {
      /* Compute C(i,j) */
      const double const *Bj = B + j*lda;
      double cij = C[i+j*lda];
      int l = 0;
      for (int k = 0; k < K; ++k, l+= lda)
	cij += Bj[k] * A[i + l];
      C[i+j*lda] = cij;
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (const int lda, const double* const A, const double* const B, double* const __restrict__ C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE) {
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE) {
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE, lda-i);
	int N = min (BLOCK_SIZE, lda-j);
	int K = min (BLOCK_SIZE, lda-k);

	/* Perform individual block dgemm */
	do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
    }
  }
}
