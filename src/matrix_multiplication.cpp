#include "matrix_multiplication.h"

#include <iostream>

#include "hls_stream.h"
#include "hls_vector.h"

void mm(DTYPE *A, hls::vector<DTYPE, DSIZE> *B, hls::vector<DTYPE, DSIZE> *AB,
        int N) {
#pragma HLS DATAFLOW
  hls::stream<hls::vector<DTYPE, DSIZE> > A_fifo;
  hls::stream<hls::vector<DTYPE, DSIZE> > B_fifo;
  hls::stream<hls::vector<DTYPE, DSIZE> > AB_fifo;
  ReadA(A, A_fifo, N);
  ReadB(B, B_fifo, N);
  Compute(A_fifo, B_fifo, AB_fifo, N);
  WriteAB(AB, AB_fifo, N);
}

void ReadA(DTYPE *A, hls::stream<hls::vector<DTYPE, DSIZE> > &A_fifo, int N) {
  for (int i = 0; i < N; i += DSIZE) {
    hls::vector<DTYPE, DSIZE> tmp;
    for (int j = 0; j < DSIZE; j++) {
#pragma HLS PIPELINE
      tmp[j] = A[i + j];
    }
    A_fifo << tmp;
  }
}

void ReadB(hls::vector<DTYPE, DSIZE> *B,
           hls::stream<hls::vector<DTYPE, DSIZE> > &B_fifo, int N) {
  for (int i = 0; i < N; i += DSIZE) {
    hls::vector<DTYPE, DSIZE> tmp;
    for (int j = 0; j < DSIZE; j++) {
#pragma HLS PIPELINE
      tmp[j] = B[i + j];
    }
    B_fifo << tmp;
  }
}

void Compute(hls::stream<hls::vector<DTYPE, DSIZE> > &A_fifo,
             hls::stream<hls::vector<DTYPE, DSIZE> > &B_fifo,
             hls::stream<hls::vector<DTYPE, DSIZE> > &AB_fifo, int N) {
  for (int i = 0; i < N; i += DSIZE) {
    hls::vector<DTYPE, DSIZE> tmpA = A_fifo.read();
    hls::vector<DTYPE, DSIZE> tmpB = B_fifo.read();
    hls::vector<DTYPE, DSIZE> tmpAB;
    for (int j = 0; j < DSIZE; j++) {
#pragma HLS PIPELINE
      tmpAB[j] = tmpA[j] * tmpB[j];
    }
    AB_fifo << tmpAB;
  }
}

void WriteAB(hls::vector<DTYPE, DSIZE> *AB,
             hls::stream<hls::vector<DTYPE, DSIZE> > &AB_fifo, int N) {
  for (int i = 0; i < N; i += DSIZE) {
    hls::vector<DTYPE, DSIZE> tmp = AB_fifo.read();
    for (int j = 0; j < DSIZE; j++) {
#pragma HLS PIPELINE
      AB[i + j] = tmp[j];
    }
  }
}
