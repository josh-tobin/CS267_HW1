# on Edision we will benchmark you against the default vendor-tuned BLAS. The compiler wrappers handle all the linking. If you wish to compare with other BLAS implementations, check the NERSC documentation.
# This makefile is intended for the GNU C compiler. To change compilers, you need to type something like: "module swap PrgEnv-pgi PrgEnv-gnu" See the NERSC documentation for available compilers.

CC = cc 
OPT = -msse -msse2 -O3 -funsafe-math-optimizations -fopt-info-all-vec
CFLAGS = -Wall -std=gnu99 $(OPT)
LDFLAGS = -Wall
# librt is needed for clock_gettime
LDLIBS = -lrt

targets = benchmark-naive benchmark-blocked benchmark-blas benchmark-blocked-unrolled benchmark-blocked-unrolled-inner benchmark-blocked-ref test_loop
objects = benchmark-test.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o dgemm-blocked-unrolled.o dgemm-blocked-unrolled-inner.o dgemm-blocked-ref.o test_loop.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark-test.o dgemm-naive.o 
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark-test.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked-ref : benchmark-test.o dgemm-blocked-ref.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark-test.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked-unrolled : benchmark-test.o dgemm-blocked-unrolled.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked-unrolled-inner : benchmark-test.o dgemm-blocked-unrolled-inner.o
	$(CC) -o $@ $^ $(LDLIBS)
test-loop : test_loop.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
clean-out:
	rm -f *.out *.err
