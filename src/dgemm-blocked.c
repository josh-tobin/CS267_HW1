#include <stdlib.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#define min(a,b) (((a)<(b))?(a):(b))

typedef const double* const __attribute__((aligned(16))) aligned_cpd;
typedef       double* restrict __attribute__((aligned(16))) aligned_rpd;

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (const int lda, const int M, const int N, const int K, aligned_cpd A, aligned_cpd B, aligned_rpd C)
{
  /* For each row i of A */
  for (int j = 0; j < N; ++j) {
    /* For each column j of B */
    aligned_cpd Bj = B + j*lda;
 
    for (int k = 0; k < K; ++k) {

      aligned_cpd Ak = A + k*lda;
      for (int i = 0; i < M; ++i) {
	C[i + j*lda] += Ak[i] * Bj[k];
      }
    }
  }
}

static aligned_cpd copy(const int lda, aligned_cpd A, aligned_cpd B) {
  aligned_cpd buf = NULL;

  int ret = posix_memalign((void*) &buf, 16, 2*lda*lda*sizeof(double));
  if (ret != 0) { return NULL; }

  aligned_cpd A_copy = buf;
  aligned_cpd B_copy = buf + (lda*lda*sizeof(double)); 

  for (int i = 0; i < lda; i += BLOCK_SIZE) {
    for (int j = 0; j < lda; j += BLOCK_SIZE) {
      for (int k = 0; k < lda; k += BLOCK_SIZE) {
	A_copy[k + i*lda]
      }
    }
  }
}

// src stored in column major; dest stored in row major
static void copy_block_rmaj(const int lda, aligned_cpd dest, aligned_cpd src) {
  // r indexes a row; s indexes a col
  for (int r = 0; r < BLOCK_SIZE; ++r) {
    for (int s = 0; s < BLOCK_SIZE; ++s) {
      dest[s + r*BLOCK_SIZE] = src[r + s*lda];
    }
  }
}

static void copy_block_cmaj(const int lda, aligned_cpd dest, aligned_cpd src) {

}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (const int lda, aligned_cpd A, aligned_cpd B, aligned_rpd C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE) {
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE) {
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = BLOCK_SIZE; //min (BLOCK_SIZE, lda-i);
	int N = BLOCK_SIZE; //min (BLOCK_SIZE, lda-j);
	int K = BLOCK_SIZE; //min (BLOCK_SIZE, lda-k);

	/* Perform individual block dgemm */
	do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
    }
  }
}
