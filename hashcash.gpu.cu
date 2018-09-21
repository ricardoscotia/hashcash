/*
 * Copyright (c) 2013
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
 
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <inttypes.h>
#include <signal.h>
#include <sys/inotify.h> // Listens for incoming files to process
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sodium.h> // For random byte generation
#include <unistd.h>
#include "hashcash.gpu.h" // Struct and macro definitions
#include "cuda.h" // GPU orientated functionality
#include "hashcash.gpu.init.c" // Initialisation functions kept separately


/*
 * For long inputs, we can compute the first blocks of the hash on the CPU
*/
inline uint16_t 
hashcash_sha_block(unsigned char * model,uint32_t * buffer,uint16_t offset) {
	uint16_t i;
	memset(buffer,0,64);
	for(i = offset;i < offset + 64;i++) {
		if(model[i] == 0) 
			return i - offset;
		buffer[(i >> 2) & 15] += (model[i] << (24 - ((i & 3) * 8)));
	}
	return 64;
}

/*
 * Process incoming files with work to do
*/
int 
hashcash_process_file(char * filename_in) {

	cudaSetDevice(0);
	cudaDeviceReset();
	
	// Defaults, per file
	// If you wanted to customise this per file, perhaps put the nbits variable in the filename and parse that.
	uint8_t nbits = settings->nbits;
	

	// An intermediate processing file is used. This way, an additional inotify could be placed on the --output folder to deal with completed jobs
	char * base = basename(filename_in);
	char * filename_proc = NULL;
	if(-1 == asprintf(&filename_proc,"%s%s",settings->process_folder,base))
		return -1;
	
	// Open the files to read from and write to
	FILE *fpin = fopen(filename_in,"r");
	FILE *fpout = fopen(filename_proc,"w");
	if(fpin == NULL) {
		fprintf(stderr,"Invalid input file %s\n",filename_in);
		exit(0);	
	}
	if(fpout == NULL) {
		fprintf(stderr,"Invalid output file %s\n",filename_proc);
		exit(0);	
	}
	
	// If verbose, some stats will be written to STDOUT
	#ifdef VERBOSE
	uint64_t total_computed = 0;
	uint64_t total_rows = 0;
	float total_time = 0.0000;
	printf("\nSHA computed\tLoop time (ms)\t\tRate (Mhash/s)\n");
	#endif
		
	while(!feof(fpin)) {
	
		unsigned char model[320] = {0};
		char date[7];
		char address[257];
		memset(address,0,257);
		memset(model,0,320);
		memset(date,0,7);
  	
  	// If there is a line that has an email address longer than 255 chars, or a date that is not 6 chars in length, we're going to ignore it
  	// Proper formatting of the date is up to the user (should by YYMMDD)
		if(fscanf(fpin,"%255[^\t]\t%6[^\n]\n",address,date) != 2)
			continue;
		
		// A unique string, used so that emails sent to the user on the same day have a unique hashcash token
		char randstr[10];
		randombytes_buf(randstr,10);
		for(int i = 0;i < 10;i++)
			randstr[i] = hexchar[(randstr[i] & 63)];
			
		uint32_t model_len = sprintf((char *) model,"1:%d:%s:%s::%.*s:",nbits,date,address,10,randstr);
	
		// Need to pad awkwardly lengthed inputs so the CPU can preprocess all but the last block
		if((model_len & 63) > 50) {
			while(model_len & 63)
				model[model_len++] = '0';
		}

		// Add the model string to an 16*uint32_t buffer
		uint32_t state[5] = {0x67452301,0xEFCDAB89,0x98BADCFE,0x10325476,0xC3D2E1F0};
		uint32_t buffer[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	
		// If the model is >56 bytes, the first blocks need processed, this can be done trivially on the CPU
		uint16_t curpos;
		uint16_t cur_block = 0;
		while((curpos = hashcash_sha_block(model,buffer,cur_block++ * 64)) == 64)
			sha_transform(state,buffer);

		// Space for counter
		model_len += 5;
	
		// Add length to last 8 bytes
		if((buffer[15] += model_len << 3) < (model_len << 3)) 
			buffer[15]++;
		buffer[14] += (model_len >> 29);
	
		// Space for counter
		uint8_t bytes[5] = {0,0,0,0,0};
		for(int i = 0;i < 5;i++) 
			bytes[i] = curpos++ >> 2;
	
		// Pad with 128, string already padded with 0's.
		buffer[curpos>>2] |= (128 << (24 - ((curpos & 3)* 8)));	
	
		// Memory to copy to and from the GPU
		uint8_t *d_bytes;
		uint32_t *d_state,*d_buffer,*d_answer;
		unsigned long long int *d_increments;
		uint32_t answer = 0;
		unsigned long long int increments = 0; 
	
		// Create and assign some memory on the CUDA device
		CUDA_E(cudaMalloc((void**)&d_state,20));
		CUDA_E(cudaMalloc((void**)&d_buffer,64));
	 	CUDA_E(cudaMalloc((void**)&d_increments,8));
	 	CUDA_E(cudaMalloc((void**)&d_answer,4));
	 	CUDA_E(cudaMalloc((void**)&d_bytes,5));
	
		CUDA_E(cudaMemcpy((void *)d_state, state,20, cudaMemcpyHostToDevice));
		CUDA_E(cudaMemcpy((void *)d_buffer,buffer,64, cudaMemcpyHostToDevice));
		CUDA_E(cudaMemcpy((void *)d_increments, &increments,8,cudaMemcpyHostToDevice));
		CUDA_E(cudaMemcpy((void *)d_answer, &answer,4, cudaMemcpyHostToDevice));
		CUDA_E(cudaMemcpy((void *)d_bytes, &bytes,5, cudaMemcpyHostToDevice));
	 	
	 	// hashit() with timing
		cudaEvent_t start, stop;
		float time;
		CUDA_E(cudaEventCreate(&start));
		CUDA_E(cudaEventCreate(&stop));
		CUDA_E(cudaEventRecord(start, 0));
		hashIt<<<settings->ngroups,settings->nthreads>>>(d_bytes,d_answer,d_increments,d_state,d_buffer,settings->nthreads * settings->ngroups,(1 << (32 - nbits)) - 1); 
		CUDA_E(cudaEventRecord(stop, 0));
	 	CUDA_E(cudaGetLastError());
	 	CUDA_E(cudaThreadSynchronize());
		CUDA_E(cudaMemcpy(&increments,d_increments,sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
		CUDA_E(cudaMemcpy(&answer,d_answer,sizeof(uint32_t), cudaMemcpyDeviceToHost));
		CUDA_E(cudaEventElapsedTime(&time, start, stop));
		CUDA_E(cudaEventDestroy(start));
		CUDA_E(cudaEventDestroy(stop));
		
		// Free up assigned memory
		CUDA_E(cudaFree((void*)d_state));
		CUDA_E(cudaFree((void*)d_buffer));
	 	CUDA_E(cudaFree((void*)d_increments));
	 	CUDA_E(cudaFree((void*)d_answer));
	 	CUDA_E(cudaFree((void*)d_bytes));
	
			for(int i = 0;i < 5;i++) {
				model[--model_len] = hexchar[answer & 0x3F];
				answer >>= 6;
			}
	
		fprintf(fpout,"%s\n",model);
		
		#ifdef VERBOSE
		printf("%12llu\t%14.5f\t%20.10f\n",increments,time,increments/(time*1000));
		total_computed += increments;
		total_time += time;
		++total_rows;
		#endif
	}
	
	#ifdef VERBOSE
	printf("JOB %s DONE\n%lu hashes computed in %f seconds from %lu rows, averaging %f seconds per row.\n",filename_in,total_computed,total_time / 1000,total_rows,(total_time / 1000) / total_rows);
	#endif
	
	fclose(fpin);
	fclose(fpout);
	
	// Move the output to its final destination
	char * filename_out = NULL;
	if(-1 == asprintf(&filename_out,"%s%s",settings->output_folder,base))
		return -1;
	rename(filename_proc,filename_out);
	
	// Cleanup
	unlink(filename_in);
	free(filename_in);
	free(filename_proc);
	free(filename_out);
	
	return 0;
}

int main
(int argc, char** argv) {
  
  // Declare default settings
	settings = (struct settings *) calloc(1,sizeof(struct settings));
	settings->nbits = 26;
	settings->nthreads = 256;
	settings->ngroups = 128;
	args_init(argc,argv);
	
	// CUDA config
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaGetDeviceCount(&settings->devices);
	
	// Inotify preparation
	inotify = (struct inotify *) calloc(1,sizeof(struct inotify));
	inotify->fd = inotify_init();
	strcpy(inotify->watchfolder,settings->input_folder);
	inotify->type = IN_CLOSE_WRITE | IN_MOVED_TO;
	inotify->wd = inotify_add_watch(inotify->fd,inotify->watchfolder,inotify->type);
	
	// An endless loop of inotify notifications
	while(1) {
	
		// Wait for a new notification
		memset(inotify->buffer,0,EVENT_BUF_LEN);
		int length = read(inotify->fd,inotify->buffer,EVENT_BUF_LEN); 
		if(length < 0) 
			break;
			
		int i = 0;
		
		// Process 1 or more notifications that have been sent here
		while (i < length) {
			inotify->event = ( struct inotify_event * ) &inotify->buffer[i];
					
			if(inotify->event->len && !(inotify->event->mask & IN_ISDIR)) {
				char * eventname = NULL;
				if(-1 != asprintf(&eventname,"%s%s",settings->input_folder,inotify->event->name)) {
					// Process the incoming file
					(*hashcash_process_file)(eventname);
					// You could install a signal handler to break out of this endless loop by setting settings->exit to a non-zero value
					if(settings->exit)
						break;
				}
				else
					free(eventname);
			}

			i += EVENT_SIZE + inotify->event->len;
		}	
		
		if(settings->exit)
			break;
	}
	
	// If we ever do arrive here, here's the cleanup
	inotify_rm_watch(inotify->fd,inotify->wd);
  close(inotify->fd);

	free(inotify);
	free(settings->input_folder);
	free(settings->output_folder);
	free(settings->process_folder);
	free(settings);
	

}


