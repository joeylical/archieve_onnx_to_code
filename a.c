#include <stdio.h>
#include "unistd.h"
#include "mnist.c"

int main(int argc, char* argv[])
{
  FILE* fp;
  float inf[28][28];
  
  fp = fopen(argv[1], "rb");
  fread(inf, 28*28*sizeof(float), 1, fp);
  fclose(fp);
  
  float out[10];
  
  Model(inf, out);
  
  for(int i=0;i < 10;i++) {
    printf("%.2f ", out[i]);
  }
  
  printf("\n");
  fflush(stdout);
  
  //sleep(1);
  
  return 0;
}