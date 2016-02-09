#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif
#if !defined(OUTER_BLOCK_SIZE)
#define OUTER_BLOCK_SIZE = 128
#endifma

#define min(a,b) (((a)<(b))?(a):(b))

typedef const double* const __attribute__((aligned(32))) aligned_cpd;
typedef       double*       __attribute__((aligned(32))) aligned_pd;
typedef       double* restrict __attribute__((aligned(32))) aligned_rpd;

//static aligned_pd _memalign(size_t alignment, size_t size) {
//  void *buf = NULL;
//  posix_memalign((void*) &buf, alignment, size);
//
//  return (aligned_pd) buf;
//}

//extern aligned_pd __dgemm_blocked_buf; // = NULL;
//static { 
//  __dgemm_blocked_buf = _memalign(16, 2*(512*512)*sizeof(double));
//}

static void print_matrix_rmaj(aligned_cpd, const int);
static void print_matrix_cmaj(aligned_cpd, const int); 

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

static void do_block_cont(const int lda, const int M, const int N, const int K, aligned_cpd A, aligned_cpd B, aligned_rpd C)
{
  // A is in row major; B is in column major (this is for a single block)

  //print_matrix_rmaj(A, BLOCK_SIZE);
  //print_matrix_cmaj(B, BLOCK_SIZE);

  for (int i = 0; i < M; i += 2) {
    aligned_cpd Ai = A + i*K; // store row of A

    for (int j = 0; j < N; ++j) {
      aligned_cpd Bj = B + j*K; // store col of B
      
      double cij = C[i + j*lda];
      for (int k = 0; k < K; k += 2) {
        cij += Ai[k] * Bj[k];
      }

      C[i + j*lda] = cij;
    }
  }
}
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values.
void square_dgemm (const int lda, aligned_cpd A, aligned_cpd B, aligned_rpd C)
{
  aligned_cpd buf = copy(lda, A, B);
  if (buf == NULL) {
    perror("Failed to allocate memory for copying.");
    exit(EXIT_FAILURE);
  }

  aligned_cpd Ac  = buf;
  aligned_cpd Bc  = buf + (lda*lda);

  int num_A_blocks_done = 0;
  
  const int lda_by_sz = lda / BLOCK_SIZE;
  const int lda_by_outer_sz = lda / OUTER_BLOCK_SIZE;
  const int sz2 = BLOCK_SIZE * BLOCK_SIZE;
  const int outer_sz_by_sz = OUTER_BLOCK_SIZE / BLOCK_SIZE;
  const int outer_sz2 = OUTER_BLOCK_SIZE * OUTER_BLOCK_SIZE;

  //print_matrix_cmaj(A, lda);
  //print_matrix_rmaj_blocks(Ac, lda); 

  //print_matrix_cmaj(B, lda);
  //print_matrix_cmaj_blocks(Bc, lda);

  //For each outer-block-row of A
  for (int i = 0; i < lda; i += OUTER_BLOCK_SIZE) {
    //For each outer-block-column of B
    for (int k = 0; k < lda; k += OUTER_BLOCK_SIZE; ++num_A_blocks_done) {// 
      aligned_cpd Ac_block = Ac + (((k*lda_by_sz + i) / BLOCK_SIZE) * sz2);

      // Accumulate block dgemms into block of C
      for (int j = 0; j < lda; j += OUTER_BLOCK_SIZE)
      {
  // Correct block dimensions if block "goes off edge of" the matrix
  int M = OUTER_BLOCK_SIZE; //min (BLOCK_SIZE, lda-i);
  int N = OUTER_BLOCK_SIZE; //min (BLOCK_SIZE, lda-j);
  int K = OUTER_BLOCK_SIZE; //min (BLOCK_SIZE, lda-k);

  int B_block_index = (j * lda_by_outer_sz + k) / OUTER_BLOCK_SIZE;

  //printf("dgemm: A_block_index = %d, B_block_index = %d\n", ((k * lda_by_sz + i) / BLOCK_SIZE), B_block_index); 

  //Perform individual block dgemm
  do_block_cont_avx(lda, M, N, K, Ac_block, Bc + B_block_index * sz2, C + i + j*lda);
      }
    }
  }
} */

static void do_block_cont_simd(const int lda, const int M, const int N, const int K, aligned_cpd A, aligned_cpd B, aligned_rpd C)
{
  // A is in col major; B is in row major (this is for a single block)

  __m128d a, b0, b1, c0, c1;

  for (int i = 0; i < M; i += 2) {
    for (int j = 0; j < N; j += 2) {
      //aligned_cpd Bj = B + j*K; // store col of B

      c0 = _mm_loadu_pd(C + i + j*lda);
      c1 = _mm_loadu_pd(C + i + (j + 1)*lda); 

      //double cij = C[i + j*lda];
      for (int k = 0; k < K; ++k) {
        a  = _mm_load_pd(A + i + k*M); // load A[i:i+1,k]
	b0 = _mm_load1_pd(B + k*N + j); // load B[k,j]
	b1 = _mm_load1_pd(B + k*N + j + 1); // load B[k,j+1]

	c0 = _mm_add_pd(c0, _mm_mul_pd(a, b0)); // C[i:i+1,j] += A[i:i+1,k] * B[k,j]
	c1 = _mm_add_pd(c1, _mm_mul_pd(a, b1)); // C[i:i+1,j+1] += A[i:i+1,k] * B[k,j+1]
      }

      // write value of C back to memory
      _mm_storeu_pd(C + i + j*lda, c0);
      _mm_storeu_pd(C + i + (j + 1)*lda, c1); 

      //C[i + j*lda] = cij;
    }
  }
}

static void do_block_cont_avx(const int lda, const int M, const int N, const int K, aligned_cpd A, aligned_cpd B, aligned_rpd C)
{
  // A is in col major; B is in row major (this is for a single block)

  __m256d a, b0, b1, b2, b3, c0, c1, c2, c3;

  for (int i = 0; i < M; i += 4) {
    for (int j = 0; j < N; j += 4) {
      //aligned_cpd Bj = B + j*K; // store col of B

      c0 = _mm256_loadu_pd(C + i + j*lda);
      c1 = _mm256_loadu_pd(C + i + (j + 1)*lda);
      c2 = _mm256_loadu_pd(C + i + (j + 2)*lda);
      c3 = _mm256_loadu_pd(C + i + (j + 3)*lda); 

      for (int k = 0; k < K; ++k) {
        a  = _mm256_load_pd(A + i + k*M); // load A[i:i+3,k]
        b0 = _mm256_broadcast_sd(B + k*N + j); // load B[k,j]
        b1 = _mm256_broadcast_sd(B + k*N + j + 1); // load B[k,j+1]
	b2 = _mm256_broadcast_sd(B + k*N + j + 2); // load B[k,j+2]
	b3 = _mm256_broadcast_sd(B + k*N + j + 3); // load B[k,j+3]

        c0 = _mm256_add_pd(c0, _mm256_mul_pd(a, b0)); // C[i:i+3,j] += A[i:i+3,k] * B[k,j]
        c1 = _mm256_add_pd(c1, _mm256_mul_pd(a, b1)); // C[i:i+3,j+1] += A[i:i+3,k] * B[k,j+1]
	c2 = _mm256_add_pd(c2, _mm256_mul_pd(a, b2)); // etc.
	c3 = _mm256_add_pd(c3, _mm256_mul_pd(a, b3)); 
      }

      // write value of C back to memory
      _mm256_storeu_pd(C + i + j*lda, c0);
      _mm256_storeu_pd(C + i + (j + 1)*lda, c1);
      _mm256_storeu_pd(C + i + (j + 2)*lda, c2);
      _mm256_storeu_pd(C + i + (j + 3)*lda, c3);
    }
  }
}


// src stored in column major; dest stored in row major
static void copy_block_rmaj(const int lda, aligned_cpd src, aligned_pd dest) {
  
  const int sz2 = BLOCK_SIZE * BLOCK_SIZE;
  const int row_sz = BLOCK_SIZE * OUTER_BLOCK_SIZE;
  const int blk_per_row = OUTER_BLOCK_SIZE / BLOCK_SIZE;
  const int lda_row = BLOCK_SIZE * lda;
  // r, t indexes a row; s, v indexes a col
  for (int r = 0; r < blk_per_row; ++r) {
    for (int s = 0; s < blk_per_row ++s) {
      for (int t = 0; t < BLOCK_SIZE; ++t) {
        for (int v = 0; v < BLOCK_SIZE; ++v) {
          dest[v + t*BLOCK_SIZE + s*sz2 + r*row_sz] = src[t + r*BLOCK_SIZE + v*lda + s*lda_row];
      }
    }
  }
}

// src and dest both stored in col major
static void copy_block_cmaj(const int lda, aligned_cpd src, aligned_pd dest) {
  const int sz2 = BLOCK_SIZE * BLOCK_SIZE;
  const int row_sz = BLOCK_SIZE * OUTER_BLOCK_SIZE;
  const int blk_per_row = OUTER_BLOCK_SIZE / BLOCK_SIZE;
  // r, t indexes a row; s, v indexes a col
  for (int s = 0; s < blk_per_row; ++s) {
    for (int r = 0; r < blk_per_row; ++r) {
      for (int t = 0; t < BLOCK_SIZE; ++t) {
        for (int v = 0; v < BLOCK_SIZE; ++v) {
          dest[t + v * BLOCK_SIZE + r*sz2 + s*row_sz] = src[t + r*BLOCK_SIZE + v*lda + s*lda_row];
        }
      }
    }
  }
}


/* Copies A and B to make blocks contiguous. Each block of A ends up
   in row-major, but the results are still column major over blocks.
*/
static aligned_cpd copy(const int lda, aligned_cpd A, aligned_cpd B) {
  aligned_pd buf = NULL;

  posix_memalign((void*) &buf, 32, 2*lda*lda*sizeof(double));
  //if (ret != 0) { 
  //  perror("copy: failed to allocate memory."); 
  //  return NULL; 
  //}

  //printf("Copying matrices: lda = %d\n", lda); 

  aligned_pd A_copy = buf;
  aligned_pd B_copy = buf + (lda*lda);

  int num_A_blocks_done = 0;
  // copy A; i indexes rows, k indexes columns
  for (int k = 0; k < lda; k += OUTER_BLOCK_SIZE) {
    for (int i = 0; i < lda; i += OUTER_BLOCK_SIZE, ++num_A_blocks_done) {
      // at this point (k/BLOCK_SIZE) * (lda / BLOCK_SIZE) + (i/BLOCK_SIZE) blocks have been written

      //printf("Copying A: num_A_blocks_done = %d\n", num_A_blocks_done);

      copy_block_cmaj(lda, A + i + k*lda, A_copy + num_A_blocks_done * (OUTER_BLOCK_SIZE * OUTER_BLOCK_SIZE));
    }
  }

  int num_B_blocks_done = 0;
  // copy B; k indexes rows, j indexes cols
  for (int j = 0; j < lda; j += OUTER_BLOCK_SIZE) {
    for (int k = 0; k < lda; k+= OUTER_BLOCK_SIZE, ++num_B_blocks_done) {
      // at this point (j/BLOCK_SIZE) * (lda / BLOCK_SIZE) + (k/BLOCK_SIZE) blocks have been written

      //printf("Copying B: num_B_blocks_done = %d\n", num_B_blocks_done);

      copy_block_rmaj(lda, B + k + j*lda, B_copy + num_B_blocks_done * (OUTER_BLOCK_SIZE * OUTER_BLOCK_SIZE));
    }
  }

  //printf("Copy successful!\n"); 

  return (aligned_cpd) buf;
}

static void print_matrix_rmaj(aligned_cpd M, const int lda) {
  printf("Printing row major matrix:\n");
  for (int i = 0; i < lda; ++i) {
    for (int k = 0; k < lda; ++k) {
      printf("%g [(%d,%d,%d)] ", M[i*lda + k], i, k, i*lda + k);
    }

    printf("\n");
  }
  printf("\n");

  return;
}

static void print_matrix_cmaj(aligned_cpd M, const int lda) {

  printf("Printing column major matrix:\n"); 
  for (int i = 0; i < lda; ++i) {
    for (int k = 0; k < lda; ++k) {
      printf("%g [(%d,%d,%d)] ", M[i + k*lda], i, k, i + k*lda);
    }

    printf("\n"); 
  }
  printf("\n");

  return;
}

// M is still col major over blocks, but each block is row major
static void print_matrix_rmaj_blocks(aligned_cpd M, const int lda) {
  printf("Printing column-over-blocks row-within-block matrix:\n"); 

  for (int i = 0; i < lda; i += BLOCK_SIZE) {
    for (int r = 0; r < BLOCK_SIZE; ++r) {
      for (int k = 0; k < lda; k += BLOCK_SIZE) {
	int block_index = (i/BLOCK_SIZE) + ((k/BLOCK_SIZE) * (lda / BLOCK_SIZE));

	int i_full = i + r;

	aligned_cpd M_block = M + (block_index * (BLOCK_SIZE * BLOCK_SIZE));
	for (int s = 0; s < BLOCK_SIZE; ++s) {
	  int k_full = k + s;

	  printf("%g [(%d, %d, %d)] ", M_block[s + r*BLOCK_SIZE], i_full, k_full, i_full + k_full*lda);
	}
      }

      printf("\n"); 
    }
  }

  printf("\n");

  return;
}

static void print_matrix_cmaj_blocks(aligned_cpd M, const int lda) {
  printf("Printing column-over-blocks column-within-block matrix:\n");

  for (int i = 0; i < lda; i+= BLOCK_SIZE) {
    for (int r = 0; r < BLOCK_SIZE; ++r) {
      for (int k = 0; k < lda; k += BLOCK_SIZE) {
        int block_index = (i/BLOCK_SIZE) + ((k/BLOCK_SIZE) * (lda / BLOCK_SIZE));

        int i_full = i + r;

        aligned_cpd M_block = M + (block_index * (BLOCK_SIZE * BLOCK_SIZE));
        for (int s = 0; s < BLOCK_SIZE; ++s) {
          int k_full = k + s;

          printf("%g [(%d, %d, %d)] ", M_block[r + s*BLOCK_SIZE], i_full, k_full, i_full + k_full*lda);
        }
      }

      printf("\n");
    }
  }
}

static void print_array(aligned_cpd x, int n) {
  printf("Printing flat array:\n");

  for (int i = 0; i < n; ++i) {
    printf("%g ", x[i]);
  }

  printf("\n");
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (const int lda, aligned_cpd A, aligned_cpd B, aligned_rpd C)
{
  aligned_cpd buf = copy(lda, A, B);
  if (buf == NULL) {
    perror("Failed to allocate memory for copying.");
    exit(EXIT_FAILURE);
  }

  aligned_cpd Ac  = buf;
  aligned_cpd Bc  = buf + (lda*lda);

  int num_A_blocks_done = 0;
  
  const int lda_by_sz = lda / BLOCK_SIZE;
  const int lda_by_outer_sz = lda / OUTER_BLOCK_SIZE;
  const int sz2 = BLOCK_SIZE * BLOCK_SIZE;
  const int outer_sz_by_sz = OUTER_BLOCK_SIZE / BLOCK_SIZE;
  const int outer_sz2 = OUTER_BLOCK_SIZE * OUTER_BLOCK_SIZE;

  //print_matrix_cmaj(A, lda);
  //print_matrix_rmaj_blocks(Ac, lda); 

  //print_matrix_cmaj(B, lda);
  //print_matrix_cmaj_blocks(Bc, lda);

  /* For each outer-block-row of A */ 
  for (int i = 0; i < lda; i += OUTER_BLOCK_SIZE) {
    /* For each outer-block-column of B */
    for (int k = 0; k < lda; k += OUTER_BLOCK_SIZE; ++num_A_blocks_done) {// 
      aligned_cpd Ac_outer_block = Ac + (((k*lda_by_outer_sz + i) / OUTER_BLOCK_SIZE) * outer_sz2);

      /* Accumulate block dgemms into block of C */
      for (int j = 0; j < lda; j += OUTER_BLOCK_SIZE)
      {
        aligned_cpd Bc_outer_block = Bc + (((j*lda_by_outer_sz + k) / OUTER_BLOCK_SIZE) * outer_sz2);
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = OUTER_BLOCK_SIZE; //min (BLOCK_SIZE, lda-i);
        int N = OUTER_BLOCK_SIZE; //min (BLOCK_SIZE, lda-j);
        int K = OUTER_BLOCK_SIZE; //min (BLOCK_SIZE, lda-k);

        for (int i_in = 0; i_in < M; r += BLOCK_SIZE) {
          for (int k_in = 0; k_in < K; s += BLOCK_SIZE) {
            aligned_cpd Ac_inner_block = Ac_outer_block + (((k_in*outer_sz_by_sz + i_in)/BLOCK_SIZE) * sz2);
            for (int j_in = 0; j_in < N; t += BLOCK_SIZE) {
              int B_inner_block_index = (j_in * outer_sz_by_sz + k_in) / BLOCK_SIZE;

              int M_in = BLOCK_SIZE; //min (BLOCK_SIZE, M-i);
              int N_in = BLOCK_SIZE; //min (BLOCK_SIZE, N-j);
              int K_in = BLOCK_SIZE; //min (BLOCK_SIZE, K-k)
              /* Perform individual block dgemm */
              do_block_cont_avx(lda, M_in, N_in, K_in, Ac_inner_block, Bc_outer_block + B_inner_block_index * sz2, C + i + i_in + j*lda + j_in*lda);

            }
          }
        }
	

	int B_block_index = (j * lda_by_outer_sz + k) / OUTER_BLOCK_SIZE;

	//printf("dgemm: A_block_index = %d, B_block_index = %d\n", ((k * lda_by_sz + i) / BLOCK_SIZE), B_block_index); 

	/* Perform individual block dgemm */
	do_block_cont_avx(lda, M, N, K, Ac_block, Bc + B_block_index * sz2, C + i + j*lda);
      }
    }
  }
}
