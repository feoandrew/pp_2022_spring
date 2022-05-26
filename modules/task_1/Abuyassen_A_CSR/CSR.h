// Copyright 2022 Abuyassen Albara

#ifndef MODULES_TASK_1_ABUYASSEN_A_CSR_CSR_H_
#define MODULES_TASK_1_ABUYASSEN_A_CSR_CSR_H_

#include <vector>
#include <complex>

typedef std::vector<std::vector<std::complex<int>>> matrix;

struct CSR {
    std::vector<std::complex<int>> values;
    std::vector<int> row_ptr;
    std::vector<int> cols_index;
    int rows;
    int cols;
    int NNZ;

    friend const matrix operator*(const CSR &A, const CSR &B);
    friend const bool operator==(const CSR &A, const CSR &B);
};

CSR sparesify(const matrix &M);
CSR sparse_transpose(const CSR &input);
const matrix operator*(const CSR &A, const CSR &B);
matrix randomSparseMatrix(int N, int M);

#endif   // MODULES_TASK_1_ABUYASSEN_A_CSR_CSR_H_
