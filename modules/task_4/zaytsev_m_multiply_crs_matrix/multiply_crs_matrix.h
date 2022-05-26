// Copyright 2022 Zaytsev Mikhail
#ifndef MODULES_TASK_4_ZAYTSEV_M_MULTIPLY_CRS_MATRIX_MULTIPLY_CRS_MATRIX_H_
#define MODULES_TASK_4_ZAYTSEV_M_MULTIPLY_CRS_MATRIX_MULTIPLY_CRS_MATRIX_H_

#include <complex>
#include <utility>
#include <vector>

class MatrixCRS {
  size_t m_numberOfRows;
  size_t m_numberOfColumns;

  std::vector<size_t> m_accumulateNonZeros;
  std::vector<size_t> m_columnsOfValues;
  std::vector<std::complex<double>> m_values;

 public:
  MatrixCRS(const size_t numberOfRows, const size_t numberOfColumns);
  MatrixCRS(
      const size_t numberOfRows,
      const size_t numberOfColumns,
      const std::vector<std::vector<std::pair<size_t, std::complex<double>>>>&
          vectorOfPair);
  MatrixCRS(const MatrixCRS& otherMatrix);

  std::vector<std::pair<size_t, std::complex<double>>> getRow(
      const size_t row) const;
  void transponse();

  MatrixCRS operator*(const MatrixCRS& otherMatrix);
  bool operator==(const MatrixCRS& otherMatrix);

  friend MatrixCRS getParallelMult(const MatrixCRS& first,
                                   const MatrixCRS& second);
};

#endif  // MODULES_TASK_4_ZAYTSEV_M_MULTIPLY_CRS_MATRIX_MULTIPLY_CRS_MATRIX_H_
