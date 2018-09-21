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
 

/*
 * Show usage on error or --help as a command line argument
 */
void 
usage(const char * error) {
	
	if(error)
		printf("\nERROR: %s\n\n",error);
		
	printf("USAGE:\n\n"
		" --input\t(required) A folder that accepts input files\n"
		" --process\t(required) A folder that holds files currently being processed\n"
		" --output\t(required) A folder that holds processed files\n"
		" --nbits\t(default: 26) 2^nbits complexity to calculate hashes for. \n"
		" --nthreads\t(default: 256) The number of CUDA threads to use. \n" 
		" --ngroups\t(default: 128) The number of CUDA groups/blocks to use.\n" 
		"\n\n"
		"Example: ./hashcash.gpu --input /tmp/hashcash/input --process /tmp/hashcash/process --output /tmp/hashcash/output --nbits 21"
		"\n\n"
		"This program will wait for files in the --input folder via inotify and compute hashcash strings from lines in the input file.\n\n"
	);
	
	exit(0);
}

/*
 * Validate a folder name command line argument
 */
char *
folder_validate(char * input,int vlen) {

	struct stat s;
	if((-1 == stat(input, &s)) || (S_ISDIR(s.st_mode) == 0))
		return NULL;

	char * output = strdup(input);
	
	if(output[vlen-1] != '/') {
		output = (char *) realloc((char *) output,vlen+2);
		output[vlen] = '/';
		output[vlen+1] = 0;
	}
	
	return output;
}

/*
 * Validate an integer command line argument
 */
uint32_t 
int_validate(char * input,uint32_t vlen,uint32_t lowerbound,uint32_t upperbound) {
	char * xptr = NULL;
	uint32_t value = strtoul((char *) input,&xptr,10);
	if(((xptr - input) != vlen) || (value < lowerbound) || (value > upperbound))
		return 0;
	return value;
}

/*
 * Checks that command line arguments are present and valid
 */
void 
args_init(int argc, char** argv) {
	for(int i = 1;i < argc;i += 2) {
		if((i + 1) == argc)
			break;
		int len  = strlen(argv[i]);
		int vlen = strlen(argv[i+1]);
		
		if((len == 6) && (strncmp(argv[i],"--help",len) == 0)) 
			usage(NULL);
		else if((len == 7) && (strncmp(argv[i],"--input",len) == 0)) {
			if(NULL == (settings->input_folder = folder_validate(argv[i+1],vlen)))
				usage("--input folder does not exist or is not a folder");
		}
		else if((len == 8) && (strncmp(argv[i],"--output",len) == 0)) {
			if(NULL == (settings->output_folder = folder_validate(argv[i+1],vlen)))
				usage("--output folder does not exist or is not a folder");
		}
		else if((len == 9) && (strncmp(argv[i],"--process",len) == 0)) {
			if(NULL == (settings->process_folder = folder_validate(argv[i+1],vlen)))
				usage("--process folder does not exist or is not a folder");
		}
		else if((len == 7) && (strncmp(argv[i],"--nbits",len) == 0)) {
			if(0 == (settings->nbits = int_validate(argv[i+1],vlen,1,31)))
				usage("--nbit should be a value between 1 and 32");
		}
		else if((len == 10) && (strncmp(argv[i],"--nthreads",len) == 0)) {
			if(0 == (settings->nthreads = int_validate(argv[i+1],vlen,1,31)))
				usage("--nthreads should be a value between 1 and 9999 (preferably a power of 2)");
		}
		else if((len == 9) && (strncmp(argv[i],"--ngroups",len) == 0)) {
			if(0 == (settings->ngroups = int_validate(argv[i+1],vlen,1,9999)))
				usage("--ngroups should be a value between 1 and 9999 (preferably a power of 2)");
		}
		else {
			printf("Unknown input parameter %s. Typo?\n",argv[i]);
			usage(NULL);
		}
		
	}
	
	if(
		settings->input_folder == NULL ||
		settings->process_folder == NULL ||
		settings->output_folder == NULL
	)
		usage("Provide folder information");
}
