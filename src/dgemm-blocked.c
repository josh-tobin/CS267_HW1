#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

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

static void do_block_cont_simd(const int lda, const int M, const int N, const int K, aligned_cpd A, aligned_cpd B, aligned_rpd C)
{
  // A is in col major; B is in row major (this is for a single block)

  __m128d a, b0, b1, c0, c1;

  int i, j;

  for (i = 0; i < M - 1; i += 2) {
    for (j = 0; j < N - 1; j += 2) {
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

    for (; j < N; ++j) {
      double cij = C[i + j*lda];

      for (int m = 0, n = 0; m < K*M; m += M, n += N) {
	cij += A[i + m] * B[j + n];
      }

      C[i + j*lda] = cij;
    }
  }

  for (; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      double cij = C[i + j*lda];

      for (int m = 0, n = 0; m < K*M; m += M, n += N) {
	cij += A[i + m] * B[j + n];
      }

      C[i + j*lda] = cij;
    }
  }

  return;
}

static void do_block_cont_avx(const int lda, const int M, const int N, const int K, aligned_cpd A, aligned_cpd B, aligned_rpd C)
{
  // A is in col major; B is in row major (this is for a single block)

  __m256d a, b0, b1, b2, b3, c0, c1, c2, c3;

  int i, j;

  for (i = 0; i < M - 3; i += 4) {
    for (j = 0; j < N - 3; j += 4) {
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

    for (; j < N; ++j) {
      double cij = C[i + j*lda];

      for (int m = 0, n = 0; m < M*K; 
	   m += M, n += N) {
	cij += A[i + m] * B[j + n];
      }

      C[i + j*lda] = cij;
    }
  }

  for (; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      double cij = C[i + j*lda];

      for (int k = 0; k < K; ++k) {
	cij += A[i + k*M] * B[j + k*N];
      }

      C[i + j*lda] = cij;
    }
  }

  //do_block_cont_simd(lda, M - i, N, K, A + i, B, C + i); 
}

// src stored in column major; dest stored in row major
static void copy_block_rmaj(const int lda, aligned_cpd src, aligned_pd dest) {
  // r indexes a row; s indexes a col
  for (int r = 0; r < BLOCK_SIZE; ++r) {
    for (int s = 0; s < BLOCK_SIZE; ++s) {
      dest[s + r*BLOCK_SIZE] = src[r + s*lda];
    }
  }
}

// src and dest both stored in col major
static void copy_block_cmaj(const int lda, aligned_cpd src, aligned_pd dest) {
  // r indexes a row; s indexes a col
  for (int s = 0; s < BLOCK_SIZE; ++s) {
    for (int r = 0; r < BLOCK_SIZE; ++r) {
      dest[r + s * BLOCK_SIZE] = src[r + s*lda];
    }
  }
}

// src and dest both stored in col major
static void copy_block_cmaj_pad(const int lda, const int r_to_go, const int s_to_go, aligned_cpd src, aligned_pd dest) {
  // r indexes a row; s indexes a col
  int r, s;
  for (s = 0; s < s_to_go; ++s) {
    for (r = 0; r < r_to_go; ++r) {
      dest[r + s * BLOCK_SIZE] = src[r + s*lda];
    }

    // write rows off edge of matrix
    for (; r < BLOCK_SIZE; ++r) {
      dest[r + s * BLOCK_SIZE] = 0.0;
    }
  }

  // write columns off edge of matrix
  for (; s < BLOCK_SIZE; ++s) {
    for (r = 0; r < BLOCK_SIZE; ++r) {
      dest[r + s * BLOCK_SIZE] = 0.0;
    }
  }
}

// src and dest both stored in col major
static void copy_block_rmaj_pad(const int lda, const int r_to_go, const int s_to_go, aligned_cpd src, aligned_pd dest) {
  // r indexes a row; s indexes a col
  int r, s;
  for (s = 0; s < s_to_go; ++s) {
    for (r = 0; r < r_to_go; ++r) {
      dest[s + r * BLOCK_SIZE] = src[r + s*lda];
    }

    // write rows off edge of matrix
    for (; r < BLOCK_SIZE; ++r) {
      dest[s + r * BLOCK_SIZE] = 0.0;
    }
  }

  // write columns off edge of matrix
  for (; s < BLOCK_SIZE; ++s) {
    for (r = 0; r < BLOCK_SIZE; ++r) {
      dest[s + r * BLOCK_SIZE] = 0.0;
    }
  }
}

/* Copies A and B to make blocks contiguous. Each block of A ends up
   in row-major, but the results are still column major over blocks.
*/
static aligned_cpd copy(const int lda, aligned_cpd A, aligned_cpd B) {
  aligned_pd buf = NULL;

  posix_memalign((void*) &buf, 32, 2*lda*lda*sizeof(double));

  aligned_pd A_copy = buf;
  aligned_pd B_copy = buf + (lda*lda);

  int num_A_blocks_done = 0;
  // copy A; i indexes rows, k indexes columns
  for (int k = 0; k < lda; k += BLOCK_SIZE) {
    for (int i = 0; i < lda; i += BLOCK_SIZE, ++num_A_blocks_done) {
      // at this point (k/BLOCK_SIZE) * (lda / BLOCK_SIZE) + (i/BLOCK_SIZE) blocks have been written

      copy_block_cmaj(lda, A + i + k*lda, A_copy + num_A_blocks_done * (BLOCK_SIZE * BLOCK_SIZE));
    }
  }

  int num_B_blocks_done = 0;
  // copy B; k indexes rows, j indexes cols
  for (int j = 0; j < lda; j += BLOCK_SIZE) {
    for (int k = 0; k < lda; k+= BLOCK_SIZE, ++num_B_blocks_done) {
      // at this point (j/BLOCK_SIZE) * (lda / BLOCK_SIZE) + (k/BLOCK_SIZE) blocks have been written

      copy_block_rmaj(lda, B + k + j*lda, B_copy + num_B_blocks_done * (BLOCK_SIZE * BLOCK_SIZE));
    }
  }

  return (aligned_cpd) buf;
}

// copy A and B into arrays padded so that their col and row #s (respectively)
// are multiples of BLOCK_SIZE
static aligned_cpd copy_padded(const int lda, const int lda_pad, aligned_cpd A, aligned_cpd B) {
  aligned_pd buf = NULL;

  posix_memalign((void*) &buf, 32, 2*lda_pad*lda_pad*sizeof(double));

  aligned_pd A_copy = buf;
  aligned_pd B_copy = buf + (lda_pad*lda_pad);

  aligned_pd A_copy_head = &(*A_copy);
  aligned_pd B_copy_head = &(*B_copy);

  int num_A_blocks_done = 0;
  // copy A; i indexes rows, k indexes columns
  for (int k = 0; k < lda; k += BLOCK_SIZE) {
    for (int i = 0; i < lda; i += BLOCK_SIZE, ++num_A_blocks_done) {
      // at this point (k/BLOCK_SIZE) * (lda / BLOCK_SIZE) + (i/BLOCK_SIZE) blocks have been written

      copy_block_cmaj_pad(lda, min(BLOCK_SIZE,lda-i), min(BLOCK_SIZE,lda-k), A + i + k*lda, A_copy + num_A_blocks_done * (BLOCK_SIZE * BLOCK_SIZE));
    }
  }

  int num_B_blocks_done = 0;
  // copy B; k indexes rows, j indexes cols
  for (int j = 0; j < lda; j += BLOCK_SIZE) {
    for (int k = 0; k < lda; k+= BLOCK_SIZE, ++num_B_blocks_done) {
      // at this point (j/BLOCK_SIZE) * (lda / BLOCK_SIZE) + (k/BLOCK_SIZE) blocks have been written

      copy_block_rmaj_pad(lda, min(BLOCK_SIZE,lda-k), min(BLOCK_SIZE,lda-j), B + k + j*lda, B_copy + num_B_blocks_done * (BLOCK_SIZE * BLOCK_SIZE));
    }
  }

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

static int round_up_to_mult(int lda, int sz) {
  if (lda % sz == 0) { return lda; }

  return ((lda/sz) + 1) * sz;
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (const int lda, aligned_cpd A, aligned_cpd B, aligned_rpd C)
{
  int lda_pad = round_up_to_mult(lda, BLOCK_SIZE);
  aligned_cpd buf = copy_padded(lda, lda_pad, A, B);
  if (buf == NULL) {
    perror("Failed to allocate memory for copying.");
    exit(EXIT_FAILURE);
  }

  aligned_cpd Ac  = buf;
  aligned_cpd Bc  = buf + (lda_pad*lda_pad);

  int num_A_blocks_done = 0;
  
  const int lda_by_sz = lda_pad / BLOCK_SIZE;
  const int sz2 = BLOCK_SIZE * BLOCK_SIZE;

  //print_matrix_cmaj(A, lda);
  //print_matrix_cmaj_blocks(Ac, lda_pad); 

  //print_matrix_cmaj(B, lda);
  //print_matrix_rmaj_blocks(Bc, lda_pad);

  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE) {
    /* For each block-column of B */
    for (int k = 0; k < lda; k += BLOCK_SIZE, ++num_A_blocks_done) {
      aligned_cpd Ac_block = Ac + (((k*lda_by_sz + i) / BLOCK_SIZE) * sz2);

      /* Accumulate block dgemms into block of C */
      for (int j = 0; j < lda; j += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = BLOCK_SIZE; //min (BLOCK_SIZE, lda-i);
	int N = BLOCK_SIZE; //min (BLOCK_SIZE, lda-j);
	int K = BLOCK_SIZE; //min (BLOCK_SIZE, lda-k);

	int B_block_index = (j * lda_by_sz + k) / BLOCK_SIZE;

	//printf("dgemm: A_block_index = %d, B_block_index = %d\n", ((k * lda_by_sz + i) / BLOCK_SIZE), B_block_index); 

	/* Perform individual block dgemm */
	do_block_cont_avx(lda, M, N, K, Ac_block, Bc + B_block_index * sz2, C + i + j*lda);
      }
    }
  }
}
