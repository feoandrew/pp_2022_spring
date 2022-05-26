// Copyright 2022 Abuyassen Albara

#include "../../../modules/task_1/Abuyassen_A_CSR/CSR.h"

#include <vector>

#include <random>

#include <complex>

typedef std::vector < std::vector < std::complex < int >>> matrix;

CSR sparesify(const matrix & M) {
  CSR new_csr;
  new_csr.rows = M.size();
  new_csr.cols = M[0].size();
  new_csr.NNZ = 0;
  new_csr.row_ptr.push_back(0);
  for (int i = 0; i < new_csr.rows; i++) {
    for (int j = 0; j < new_csr.cols; j++) {
      if (M[i][j] != 0) {
        new_csr.values.push_back(M[i][j]);
        new_csr.cols_index.push_back(j);

        new_csr.NNZ++;
      }
    }
    new_csr.row_ptr.push_back(new_csr.NNZ);
  }

  return new_csr;
}

CSR sparse_transpose(const CSR & input) {
  CSR res {
    std::vector < std::complex < int >> (input.NNZ, 0),
      std::vector < int > (input.cols + 2, 0),  // one extra
      std::vector < int > (input.NNZ, 0),
      input.cols,
      input.rows,
      input.NNZ
  };

  // count per column
  for (int i = 0; i < input.NNZ; ++i) {
    ++res.row_ptr[input.cols_index[i] + 2];
  }

  // from count per column generate new rowPtr (but shifted)
  for (size_t i = 2; i < res.row_ptr.size(); ++i) {
    // create incremental sum
    res.row_ptr[i] += res.row_ptr[i - 1];
  }

  // perform the main part
  for (int i = 0; i < input.rows; ++i) {
    for (int j = input.row_ptr[i]; j < input.row_ptr[i + 1]; ++j) {
      // calculate index to transposed matrix at which we should place current element,
      // and at the same time build final rowPtr
      const int new_index = res.row_ptr[input.cols_index[j] + 1]++;
      res.values[new_index] = input.values[j];
      res.cols_index[new_index] = i;
    }
  }
  res.row_ptr.pop_back();  // pop that one extra

  return res;
}

const matrix operator * (const CSR & A,
  const CSR & B) {
  if (A.cols != B.rows)
    throw "Eroor";
  CSR B_t = sparse_transpose(B);

  matrix result(A.rows, std::vector < std::complex < int >> (B.cols, 0));
  for (int i = 0; i < A.rows; i++) {
    for (int j = 0; j < B_t.rows; j++) {
      std::complex < int > sum(0, 0);
      for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++)
        for (int l = B_t.row_ptr[j]; l < B_t.row_ptr[j + 1]; l++)
          if (A.cols_index[k] == B_t.cols_index[l]) {
            sum += A.values[k] * B_t.values[l];
            break;
          }
      if (sum != std::complex < int > (0, 0)) {
        result[i][j] += sum;
      }
    }
  }
  return result;
}

const bool operator == (const CSR & A,
  const CSR & B) {
  return A.values == B.values && A.cols_index == B.cols_index
  && A.row_ptr == B.row_ptr && A.rows == B.rows && A.cols == B.cols && A.NNZ == B.NNZ;
}

matrix randomSparseMatrix(int N, int M) {
  std::random_device dev;
  std::mt19937 gen(dev());
  matrix result(N, std::vector < std::complex < int >> (M, 0));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      if (gen() % 100 == 0) {
        result[i][j] = std::complex < int > (gen() % 10, gen() % 10);
      } else {
        result[i][j] = std::complex < int > (0, 0);
      }
    }
  }
  return result;
}
