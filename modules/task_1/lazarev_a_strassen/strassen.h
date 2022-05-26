// Copyright 2022 Lazarev Aleksey
#ifndef MODULES_TASK_1_LAZAREV_A_STRASSEN_STRASSEN_H_
#define MODULES_TASK_1_LAZAREV_A_STRASSEN_STRASSEN_H_

#include <string>
#include <vector>

const int threshold = 128;

using vector = std::vector<int>;
using matrix = std::vector<std::vector<int>>;

void setToRandom(matrix* A, int n);

matrix strassenMultiply(matrix* A, matrix* B, int n);
matrix multiply(matrix* A, matrix* B, int n);

#endif  // MODULES_TASK_1_LAZAREV_A_STRASSEN_STRASSEN_H_
