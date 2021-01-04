#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

void export_layer(float*p, uint32_t size)
{
  FILE* fp = fopen("export.dat", "wb");
  fwrite(p, size, sizeof(float), fp);
  fclose(fp);
}

#include "mobilenet_v1_1.0_224.c"
#include "labels.c"

float inf1[224][224][3];
float inf[3][224][224];
float out[1001] = {0.0};

int main(int argc, char* argv[])
{
  FILE* fp;
  
  fp = fopen("img.dat", "rb");
  if(!fp) {
    printf("cannot open file\n");
    return 1;
  }
  fread(inf, 224*224*3, sizeof(float), fp);
//   for(int c=0;c<3;c++) {
//     for(int i=0;i<224;i++) {
//         for(int j=0;j<224;j++) {
// //             inf[c][i][j] = (inf1[i][j][c]/2+0.5)*255;
//           inf[c][i][j] = inf1[i][j][c];
//         }
//     }
//   }

  fclose(fp);
  
  Model(inf, out);
  
  int top5_i[5] = {0};
  float top5[5]={FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN};
  
  for(int i=0;i < 1001;i++) {
//     printf("%.3f ", out[i]);
    for(int j=0;j<5;j++) {
      if(top5[j] < out[i]) {
        if(j==4) {
          top5[j] = out[i];
          top5_i[j] = i;
        }else{
          memcpy(top5+j, top5+j+1, (4-j)*sizeof(float));
          memcpy(top5_i+j, top5_i+j+1, (4-j)*sizeof(int));
          top5[j] = out[i];
          top5_i[j] = i;
          break;
        }
      }
    }
  }
  
  printf("\nTOP 5:\n");
  for(int i=0;i<5;i++) {
    printf("   %d %.3f\t%s\n", top5_i[i], top5[i], labels[top5_i[i]]);
  }

  return 0;
}