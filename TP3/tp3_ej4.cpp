#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <chrono> 
using namespace std::chrono;


int main(int arc, char** argv)
{ 

  int x[] = { 10, 20, 30, 40};
  int y[] = { 10, 20, 30, 40};
  int z[] = {};
  int n;
  omp_set_num_threads(10);

  #pragma omp for
  for(n=0; n<4; n++){
    z[n] = x[n] + y[n];
  }

  for(n=0; n<4; n++) 
    printf("POS: %d, VALUE: %d \n", n, z[n]);


  return 0;
} 
