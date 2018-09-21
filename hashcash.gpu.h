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
 
#define EVENT_SIZE	(sizeof (struct inotify_event))
#define EVENT_BUF_LEN		(1024 * (EVENT_SIZE + 16))
#define FNAMESIZE 512
#define handle_error(msg) do { perror(msg); exit(EXIT_FAILURE); } while (0)
static char hexchar[64] = {'0','1','2','3','4','5','6','7','8','9',
	'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
	'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
	'=','+'};
		
struct settings {
	char * input_folder;
	char * process_folder;
	char * output_folder;
	
	int devices;
	uint32_t device_counter;
	
	uint32_t exit;
	
	uint32_t nbits;
	uint32_t nthreads;
	uint32_t ngroups;
};

struct inotify {
	uint32_t type;
	char watchfolder[FNAMESIZE];
	int wd;
	int fd;
	struct inotify_event *event;
	char buffer[EVENT_BUF_LEN];
};

struct settings * settings;
struct inotify * inotify;
