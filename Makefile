EXTRA_CFLAGS_CUDA = -D_FORCE_INLINES --compiler-options -Wall -lm -Xptxas -v --use_fast_math -prec-div=false -O3 
CUDA_CC = /usr/local/cuda-7.5/bin/nvcc

hashcash.gpu: hashcash.gpu.cu
	$(CUDA_CC) $(EXTRA_CFLAGS_CUDA) -arch=sm_20 -lsodium -D VERBOSE -o $@ $^
	

