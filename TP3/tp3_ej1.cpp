#include <stdio.h>
#include <omp.h>


int main(int argc, char **argv){
	int n,tid;
	omp_set_num_threads(5);
	#pragma omp parallel private (tid, n)
	{
		tid = omp_get_thread_num();
		printf("Hola mundo desde el thread N° %d \n",tid);
	}
	return(0);
}