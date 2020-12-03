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
  
  int max_i;
  float max=-99999;
  for(int i=0;i < 10;i++) {
    if(max < out[i]) {
      max = out[i];
      max_i = i;
    }
  }
  
  printf("num: %d\n", max_i);
  
  return 0;
}