#pragma once
#include <iostream>
#include <vector>
#include "hls_stream.h"
#include "matrix_multiplication.h"

typedef int DTYPE;
const int SIZE = 8;
const int BLOCK_SIZE = 4;

typedef struct {
  DTYPE a[BLOCK_SIZE];
} blockvec;

typedef struct {
  DTYPE out[BLOCK_SIZE][BLOCK_SIZE];
} blockmat;

void blockmatmul(hls::stream<blockvec> &Arows, hls::stream<blockvec> &Bcols,
blockmat &ABpartial, DTYPE iteration);
