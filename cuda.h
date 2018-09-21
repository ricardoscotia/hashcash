__host__ __device__ uint32_t rotate1(uint32_t x);
__host__ __device__ uint32_t rotate5(uint32_t x);
__host__ __device__ uint32_t rotate30(uint32_t x);
__host__ __device__ uint32_t andnot(uint32_t a,uint32_t b);
__host__ __device__ void sha_transform(uint32_t *S,uint32_t * B);

#define CUDA_E(func)\
do {\
	cudaError_t err = (func);\
	if (err != cudaSuccess) {\
		printf("%s(%d): ERROR: %s returned %s (err#%d)\n", __FILE__, __LINE__, #func,cudaGetErrorString(err), err);\
		exit(-1);\
		}\
}	 while (0)


__device__ const uint8_t g_hexchar[64] =	{'0','1','2','3','4','5','6','7','8','9',
	'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
	'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
	'=','+'};
__device__ volatile uint8_t answered = 1;
__device__ uint32_t answer = 1;

__host__ __device__ uint32_t rotate1(uint32_t x) { return  (x << 1) | (x >> (31)); }
__host__ __device__ uint32_t rotate5(uint32_t x) { return  (x << 5) | (x >> (27)); }
__host__ __device__ uint32_t rotate30(uint32_t x) { return  (x << 30) | (x >> (2)); }
__host__ __device__ uint32_t andnot(uint32_t a,uint32_t b) { return (a & ~b); }
__host__ __device__ void sha_transform(uint32_t *S,uint32_t * B) {

  uint32_t a = S[0];
  uint32_t b = S[1];
  uint32_t c = S[2];
  uint32_t d = S[3];
  uint32_t e = S[4];
  uint32_t f;

   f = ((c & b) | andnot(d,b));
   e = rotate5(a) + f + e + 0x5a827999 + B[0];
   b = rotate30(b);
      f = (b & a) | andnot(c,a);
      d = rotate5(e) + f + d + 0x5a827999 + B[1];
      a = rotate30(a);
      f = (a & e) | andnot(b,e);
      c = rotate5(d) + f + c + 0x5a827999 + B[2];
      e = rotate30(e);
      f = (e & d) | andnot(a,d);
      b = rotate5(c) + f + b + 0x5a827999 + B[3];
      d = rotate30(d);
      f = (d & c) | andnot(e,c);
      a = rotate5(b) + f + a + 0x5a827999 + B[4];
      c = rotate30(c);
      f = (c & b) | andnot(d,b);
      e = rotate5(a) + f + e + 0x5a827999 + B[5];
      b = rotate30(b);
      f = (b & a) | andnot(c,a);
      d = rotate5(e) + f + d + 0x5a827999 + B[6];
      a = rotate30(a);
      f = (a & e) | andnot(b,e);
      c = rotate5(d) + f + c + 0x5a827999 + B[7];
      e = rotate30(e);
      f = (e & d) | andnot(a,d);
      b = rotate5(c) + f + b + 0x5a827999 + B[8];
      d = rotate30(d);
      f = (d & c) | andnot(e,c);
      a = rotate5(b) + f + a + 0x5a827999 + B[9];
      c = rotate30(c);
      f = (c & b) | andnot(d,b);
      e = rotate5(a) + f + e + 0x5a827999 + B[10];
      b = rotate30(b);
      f = (b & a) | andnot(c,a);
      d = rotate5(e) + f + d + 0x5a827999 + B[11];
      a = rotate30(a);
      f = (a & e) | andnot(b,e);
      c = rotate5(d) + f + c + 0x5a827999 + B[12];
      e = rotate30(e);
      f = (e & d) | andnot(a,d);
      b = rotate5(c) + f + b + 0x5a827999 + B[13];
      d = rotate30(d);
      f = (d & c) | andnot(e,c);
      a = rotate5(b) + f + a + 0x5a827999 + B[14];
      c = rotate30(c);
      f = (c & b) | andnot(d,b);
      e = rotate5(a) + f + e + 0x5a827999 + B[15];
      b = rotate30(b);
      B[0] = rotate1(B[13] ^ B[8] ^ B[2] ^ B[0]);
      f = (b & a) | andnot(c,a);
      d = rotate5(e) + f + d + 0x5a827999 + B[0];
      a = rotate30(a);
      B[1] = rotate1(B[14] ^ B[9] ^ B[3] ^ B[1]);
      f = (a & e) | andnot(b,e);
      c = rotate5(d) + f + c + 0x5a827999 + B[1];
      e = rotate30(e);
      B[2] = rotate1(B[15] ^ B[10] ^ B[4] ^ B[2]);
      f = (e & d) | andnot(a,d);
      b = rotate5(c) + f + b + 0x5a827999 + B[2];
      d = rotate30(d);
      B[3] = rotate1(B[0] ^ B[11] ^ B[5] ^ B[3]);
      f = (d & c) | andnot(e,c);
      a = rotate5(b) + f + a + 0x5a827999 + B[3];
      c = rotate30(c);
      B[4] = rotate1(B[1] ^ B[12] ^ B[6] ^ B[4]);
      f = b ^ c ^ d;
      e = rotate5(a) + f + e + 0x6ed9eba1 + B[4];
      b = rotate30(b);
      B[5] = rotate1(B[2] ^ B[13] ^ B[7] ^ B[5]);
      f = a ^ b ^ c;
      d = rotate5(e) + f + d + 0x6ed9eba1 + B[5];
      a = rotate30(a);
      B[6] = rotate1(B[3] ^ B[14] ^ B[8] ^ B[6]);
      f = e ^ a ^ b;
      c = rotate5(d) + f + c + 0x6ed9eba1 + B[6];
      e = rotate30(e);
      B[7] = rotate1(B[4] ^ B[15] ^ B[9] ^ B[7]);
      f = d ^ e ^ a;
      b = rotate5(c) + f + b + 0x6ed9eba1 + B[7];
      d = rotate30(d);
      B[8] = rotate1(B[5] ^ B[0] ^ B[10] ^ B[8]);
      f = c ^ d ^ e;
      a = rotate5(b) + f + a + 0x6ed9eba1 + B[8];
      c = rotate30(c);
      B[9] = rotate1(B[6] ^ B[1] ^ B[11] ^ B[9]);
      f = b ^ c ^ d;
      e = rotate5(a) + f + e + 0x6ed9eba1 + B[9];
      b = rotate30(b);
      B[10] = rotate1(B[7] ^ B[2] ^ B[12] ^ B[10]);
      f = a ^ b ^ c;
      d = rotate5(e) + f + d + 0x6ed9eba1 + B[10];
      a = rotate30(a);
      B[11] = rotate1(B[8] ^ B[3] ^ B[13] ^ B[11]);
      f = e ^ a ^ b;
      c = rotate5(d) + f + c + 0x6ed9eba1 + B[11];
      e = rotate30(e);
      B[12] = rotate1(B[9] ^ B[4] ^ B[14] ^ B[12]);
      f = d ^ e ^ a;
      b = rotate5(c) + f + b + 0x6ed9eba1 + B[12];
      d = rotate30(d);
      B[13] = rotate1(B[10] ^ B[5] ^ B[15] ^ B[13]);
      f = c ^ d ^ e;
      a = rotate5(b) + f + a + 0x6ed9eba1 + B[13];
      c = rotate30(c);
      B[14] = rotate1(B[11] ^ B[6] ^ B[0] ^ B[14]);
      f = b ^ c ^ d;
      e = rotate5(a) + f + e + 0x6ed9eba1 + B[14];
      b = rotate30(b);
      B[15] = rotate1(B[12] ^ B[7] ^ B[1] ^ B[15]);
      f = a ^ b ^ c;
      d = rotate5(e) + f + d + 0x6ed9eba1 + B[15];
      a = rotate30(a);
      B[0] = rotate1(B[13] ^ B[8] ^ B[2] ^ B[0]);
      f = e ^ a ^ b;
      c = rotate5(d) + f + c + 0x6ed9eba1 + B[0];
      e = rotate30(e);
      B[1] = rotate1(B[14] ^ B[9] ^ B[3] ^ B[1]);
      f = d ^ e ^ a;
      b = rotate5(c) + f + b + 0x6ed9eba1 + B[1];
      d = rotate30(d);
      B[2] = rotate1(B[15] ^ B[10] ^ B[4] ^ B[2]);
      f = c ^ d ^ e;
      a = rotate5(b) + f + a + 0x6ed9eba1 + B[2];
      c = rotate30(c);
      B[3] = rotate1(B[0] ^ B[11] ^ B[5] ^ B[3]);
      f = b ^ c ^ d;
      e = rotate5(a) + f + e + 0x6ed9eba1 + B[3];
      b = rotate30(b);
      B[4] = rotate1(B[1] ^ B[12] ^ B[6] ^ B[4]);
      f = a ^ b ^ c;
      d = rotate5(e) + f + d + 0x6ed9eba1 + B[4];
      a = rotate30(a);
      B[5] = rotate1(B[2] ^ B[13] ^ B[7] ^ B[5]);
      f = e ^ a ^ b;
      c = rotate5(d) + f + c + 0x6ed9eba1 + B[5];
      e = rotate30(e);
      B[6] = rotate1(B[3] ^ B[14] ^ B[8] ^ B[6]);
      f = d ^ e ^ a;
      b = rotate5(c) + f + b + 0x6ed9eba1 + B[6];
      d = rotate30(d);
      B[7] = rotate1(B[4] ^ B[15] ^ B[9] ^ B[7]);
      f = c ^ d ^ e;
      a = rotate5(b) + f + a + 0x6ed9eba1 + B[7];
      c = rotate30(c);
      B[8] = rotate1(B[5] ^ B[0] ^ B[10] ^ B[8]);
      f = (b & c) | (b & d) | (c & d);
      e = rotate5(a) + f + e + 0x8f1bbcdc + B[8];
      b = rotate30(b);
      B[9] = rotate1(B[6] ^ B[1] ^ B[11] ^ B[9]);
      f = (a & b) | (a & c) | (b & c);
      d = rotate5(e) + f + d + 0x8f1bbcdc + B[9];
      a = rotate30(a);
      B[10] = rotate1(B[7] ^ B[2] ^ B[12] ^ B[10]);
      f = (e & a) | (e & b) | (a & b);
      c = rotate5(d) + f + c + 0x8f1bbcdc + B[10];
      e = rotate30(e);
      B[11] = rotate1(B[8] ^ B[3] ^ B[13] ^ B[11]);
      f = (d & e) | (d & a) | (e & a);
      b = rotate5(c) + f + b + 0x8f1bbcdc + B[11];
      d = rotate30(d);
      B[12] = rotate1(B[9] ^ B[4] ^ B[14] ^ B[12]);
      f = (c & d) | (c & e) | (d & e);
      a = rotate5(b) + f + a + 0x8f1bbcdc + B[12];
      c = rotate30(c);
      B[13] = rotate1(B[10] ^ B[5] ^ B[15] ^ B[13]);
      f = (b & c) | (b & d) | (c & d);
      e = rotate5(a) + f + e + 0x8f1bbcdc + B[13];
      b = rotate30(b);
      B[14] = rotate1(B[11] ^ B[6] ^ B[0] ^ B[14]);
      f = (a & b) | (a & c) | (b & c);
      d = rotate5(e) + f + d + 0x8f1bbcdc + B[14];
      a = rotate30(a);
      B[15] = rotate1(B[12] ^ B[7] ^ B[1] ^ B[15]);
      f = (e & a) | (e & b) | (a & b);
      c = rotate5(d) + f + c + 0x8f1bbcdc + B[15];
      e = rotate30(e);
      B[0] = rotate1(B[13] ^ B[8] ^ B[2] ^ B[0]);
      f = (d & e) | (d & a) | (e & a);
      b = rotate5(c) + f + b + 0x8f1bbcdc + B[0];
      d = rotate30(d);
      B[1] = rotate1(B[14] ^ B[9] ^ B[3] ^ B[1]);
      f = (c & d) | (c & e) | (d & e);
      a = rotate5(b) + f + a + 0x8f1bbcdc + B[1];
      c = rotate30(c);
      B[2] = rotate1(B[15] ^ B[10] ^ B[4] ^ B[2]);
      f = (b & c) | (b & d) | (c & d);
      e = rotate5(a) + f + e + 0x8f1bbcdc + B[2];
      b = rotate30(b);
      B[3] = rotate1(B[0] ^ B[11] ^ B[5] ^ B[3]);
      f = (a & b) | (a & c) | (b & c);
      d = rotate5(e) + f + d + 0x8f1bbcdc + B[3];
      a = rotate30(a);
      B[4] = rotate1(B[1] ^ B[12] ^ B[6] ^ B[4]);
      f = (e & a) | (e & b) | (a & b);
      c = rotate5(d) + f + c + 0x8f1bbcdc + B[4];
      e = rotate30(e);
      B[5] = rotate1(B[2] ^ B[13] ^ B[7] ^ B[5]);
      f = (d & e) | (d & a) | (e & a);
      b = rotate5(c) + f + b + 0x8f1bbcdc + B[5];
      d = rotate30(d);
      B[6] = rotate1(B[3] ^ B[14] ^ B[8] ^ B[6]);
      f = (c & d) | (c & e) | (d & e);
      a = rotate5(b) + f + a + 0x8f1bbcdc + B[6];
      c = rotate30(c);
      B[7] = rotate1(B[4] ^ B[15] ^ B[9] ^ B[7]);
      f = (b & c) | (b & d) | (c & d);
      e = rotate5(a) + f + e + 0x8f1bbcdc + B[7];
      b = rotate30(b);
      B[8] = rotate1(B[5] ^ B[0] ^ B[10] ^ B[8]);
      f = (a & b) | (a & c) | (b & c);
      d = rotate5(e) + f + d + 0x8f1bbcdc + B[8];
      a = rotate30(a);
      B[9] = rotate1(B[6] ^ B[1] ^ B[11] ^ B[9]);
      f = (e & a) | (e & b) | (a & b);
      c = rotate5(d) + f + c + 0x8f1bbcdc + B[9];
      e = rotate30(e);
      B[10] = rotate1(B[7] ^ B[2] ^ B[12] ^ B[10]);
      f = (d & e) | (d & a) | (e & a);
      b = rotate5(c) + f + b + 0x8f1bbcdc + B[10];
      d = rotate30(d);
      B[11] = rotate1(B[8] ^ B[3] ^ B[13] ^ B[11]);
      f = (c & d) | (c & e) | (d & e);
      a = rotate5(b) + f + a + 0x8f1bbcdc + B[11];
      c = rotate30(c);
      B[12] = rotate1(B[9] ^ B[4] ^ B[14] ^ B[12]);
      f = b ^ c ^ d;
      e = rotate5(a) + f + e + 0xca62c1d6 + B[12];
      b = rotate30(b);
      B[13] = rotate1(B[10] ^ B[5] ^ B[15] ^ B[13]);
      f = a ^ b ^ c;
      d = rotate5(e) + f + d + 0xca62c1d6 + B[13];
      a = rotate30(a);
      B[14] = rotate1(B[11] ^ B[6] ^ B[0] ^ B[14]);
      f = e ^ a ^ b;
      c = rotate5(d) + f + c + 0xca62c1d6 + B[14];
      e = rotate30(e);
      B[15] = rotate1(B[12] ^ B[7] ^ B[1] ^ B[15]);
      f = d ^ e ^ a;
      b = rotate5(c) + f + b + 0xca62c1d6 + B[15];
      d = rotate30(d);
      B[0] = rotate1(B[13] ^ B[8] ^ B[2] ^ B[0]);
      f = c ^ d ^ e;
      a = rotate5(b) + f + a + 0xca62c1d6 + B[0];
      c = rotate30(c);
      B[1] = rotate1(B[14] ^ B[9] ^ B[3] ^ B[1]);
      f = b ^ c ^ d;
      e = rotate5(a) + f + e + 0xca62c1d6 + B[1];
      b = rotate30(b);
      B[2] = rotate1(B[15] ^ B[10] ^ B[4] ^ B[2]);
      f = a ^ b ^ c;
      d = rotate5(e) + f + d + 0xca62c1d6 + B[2];
      a = rotate30(a);
      B[3] = rotate1(B[0] ^ B[11] ^ B[5] ^ B[3]);
      f = e ^ a ^ b;
      c = rotate5(d) + f + c + 0xca62c1d6 + B[3];
      e = rotate30(e);
      B[4] = rotate1(B[1] ^ B[12] ^ B[6] ^ B[4]);
      f = d ^ e ^ a;
      b = rotate5(c) + f + b + 0xca62c1d6 + B[4];
      d = rotate30(d);
      B[5] = rotate1(B[2] ^ B[13] ^ B[7] ^ B[5]);
      f = c ^ d ^ e;
      a = rotate5(b) + f + a + 0xca62c1d6 + B[5];
      c = rotate30(c);
      B[6] = rotate1(B[3] ^ B[14] ^ B[8] ^ B[6]);
      f = b ^ c ^ d;
      e = rotate5(a) + f + e + 0xca62c1d6 + B[6];
      b = rotate30(b);
      B[7] = rotate1(B[4] ^ B[15] ^ B[9] ^ B[7]);
      f = a ^ b ^ c;
      d = rotate5(e) + f + d + 0xca62c1d6 + B[7];
      a = rotate30(a);
      B[8] = rotate1(B[5] ^ B[0] ^ B[10] ^ B[8]);
      f = e ^ a ^ b;
      c = rotate5(d) + f + c + 0xca62c1d6 + B[8];
      e = rotate30(e);
      B[9] = rotate1(B[6] ^ B[1] ^ B[11] ^ B[9]);
      f = d ^ e ^ a;
      b = rotate5(c) + f + b + 0xca62c1d6 + B[9];
      d = rotate30(d);
      B[10] = rotate1(B[7] ^ B[2] ^ B[12] ^ B[10]);
      f = c ^ d ^ e;
      a = rotate5(b) + f + a + 0xca62c1d6 + B[10];
      c = rotate30(c);
      B[11] = rotate1(B[8] ^ B[3] ^ B[13] ^ B[11]);
      f = b ^ c ^ d;
      e = rotate5(a) + f + e + 0xca62c1d6 + B[11];
      b = rotate30(b);
      B[12] = rotate1(B[9] ^ B[4] ^ B[14] ^ B[12]);
      f = a ^ b ^ c;
      d = rotate5(e) + f + d + 0xca62c1d6 + B[12];
      a = rotate30(a);
      B[13] = rotate1(B[10] ^ B[5] ^ B[15] ^ B[13]);
      f = e ^ a ^ b;
      c = rotate5(d) + f + c + 0xca62c1d6 + B[13];
      e = rotate30(e);
      B[14] = rotate1(B[11] ^ B[6] ^ B[0] ^ B[14]);
      f = d ^ e ^ a;
      b = rotate5(c) + f + b + 0xca62c1d6 + B[14];
      d = rotate30(d);
      B[15] = rotate1(B[12] ^ B[7] ^ B[1] ^ B[15]);
      f = c ^ d ^ e;
      a = rotate5(b) + f + a + 0xca62c1d6 + B[15];
      c = rotate30(c);
    S[0] += a;
    S[1] += b;
    S[2] += c;
    S[3] += d;
    S[4] += e;
}


__global__ void hashIt(const uint8_t * bytes,uint32_t *answer,unsigned long long int* increments,const uint32_t* pre_state,const uint32_t* pre_buffer,const uint32_t increment,const uint32_t m0) {

	uint32_t S[5] = {0};
	uint32_t B[16] = {0};
	uint32_t counter;
	uint8_t i;
	if(threadIdx.x == 0 && blockIdx.x == 0) {
		answer = 0;
		answered = 0;
	}
	
	__syncthreads();
		// Iterate until we find an answer or try ~0.5 billion hashes
		for(counter = ((threadIdx.x +blockIdx.x*blockDim.x) + 1);(answered == 0) && (counter < 0x20000000);counter += increment) {
			// State 
				#pragma unroll 5
				for(i = 0;i < 5;i++) 
					S[i] = pre_state[i];
			// Buffer 
				#pragma unroll 16
				for(i = 0;i < 16;i++) 
					B[i] = pre_buffer[i];
			// Counter component
				#pragma unroll 5
				for(i = 0;i < 5;i++) 
					B[bytes[i]] += (g_hexchar[(counter >> (24 - (i * 6))) & 0x3F] << (24 - ((B[15] - (40 - (i * 8))) & 24)));
			
			// Perform SHA1 transformation  
			sha_transform(S,B);
			// Compare the first 32 bits of the hash to our chosen complexity
			if((S[0] <= m0) && !answered) {
				answered = 1;
				atomicAdd(answer,(unsigned long long int) counter);
				break;
			}
		}
	atomicAdd(increments,(unsigned long long int) counter / increment);
	__threadfence();
}

