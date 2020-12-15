#include <omp.h>
#include <stdio.h>
#include <unistd.h>
#define N 40

main () {
	int tid;
	int A[N];
	int i;

	for(i=0; i<N; i++) 
		A[i]=-1;
	
	#pragma omp parallel for schedule(dynamic,4) private(tid)
	
	for (i=0; i<N; i++){
		tid = omp_get_thread_num();
		A[i] = tid;
		usleep(1);
	}
	
	for (i=0; i<N/2; i++) 
		printf (" %2d", i); 
	printf ("\n");
	
	for (i=0; i<N/2; i++) 
		printf (" %2d", A[i]);
	printf ("\n\n\n");

	for (i=N/2; i<N; i++) 
		printf (" %2d",i); 
	printf ("\n");

	for (i=N/2; i<N; i++) 
		printf (" %2d", A[i]);
	printf ("\n\n");
}
