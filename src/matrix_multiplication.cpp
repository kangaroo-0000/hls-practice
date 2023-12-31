#include "matrix_multiplication.h"

#include "hls_vector.h"

void blockmatmul(hls::stream<blockvec> &Arows, hls::stream<blockvec> &Bcols,
                 blockmat &ABpartial, int it) {
#pragma HLS DATAFLOW
  int counter = it % (SIZE / BLOCK_SIZE);
  static DTYPE A[BLOCK_SIZE][SIZE];
  if (counter == 0) {  // only load the A rows when necessary
  loadA:
    for (int i = 0; i < SIZE; i++) {
      blockvec tempA = Arows.read();
      for (int j = 0; j < BLOCK_SIZE; j++) {
#pragma HLS PIPELINE
        A[j][i] = tempA.a[j];
      }
    }
  }
  DTYPE AB[BLOCK_SIZE][BLOCK_SIZE] = {0};
#pragma HLS ARRAY_PARTITION variable=AB type=block factor=2 dim=2

partialsum:
  for (int k = 0; k < SIZE; k++) {
    blockvec tempB = Bcols.read();
    for (int i = 0; i < BLOCK_SIZE; i++) {
      for (int j = 0; j < BLOCK_SIZE; j++) {
#pragma HLS PIPELINE
        AB[i][j] = AB[i][j] + A[i][k] * tempB.a[j];
      }
    }
  }

writeoutput:
  for (int i = 0; i < BLOCK_SIZE; i++) {
    for (int j = 0; j < BLOCK_SIZE; j++) {
#pragma HLS UNROLL
      ABpartial.out[i][j] = AB[i][j];
    }
  }
}
