// Copyright 2022 Zaytsev Mikhail
#include "../../../modules/task_3/zaytsev_m_multiply_crs_matrix/multiply_crs_matrix.h"
#include <tbb/tbb.h>
#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

using std::complex;
using std::pair;
using std::vector;

MatrixCRS::MatrixCRS(const size_t numberOfRows, const size_t numberOfColumns)
    : m_numberOfRows(numberOfRows),
      m_numberOfColumns(numberOfColumns),
      m_accumulateNonZeros(),
      m_columnsOfValues(),
      m_values() {
  m_accumulateNonZeros.reserve(m_numberOfRows + 1);
  m_accumulateNonZeros.push_back(0);
}
MatrixCRS::MatrixCRS(
    const size_t numberOfRows,
    const size_t numberOfColumns,
    const vector<vector<pair<size_t, complex<double>>>>& vectorOfPair)
    : m_numberOfRows(numberOfRows),
      m_numberOfColumns(numberOfColumns),
      m_accumulateNonZeros(),
      m_columnsOfValues(),
      m_values() {
  m_accumulateNonZeros.reserve(m_numberOfRows + 1);
  m_accumulateNonZeros.push_back(0);

  for (auto& vec : vectorOfPair) {
    m_accumulateNonZeros.push_back(
        m_accumulateNonZeros[m_accumulateNonZeros.size() - 1] + vec.size());
    for (auto& elem : vec) {
      m_columnsOfValues.push_back(elem.first);
      m_values.push_back(elem.second);
    }
  }
}
MatrixCRS::MatrixCRS(const MatrixCRS& otherMatrix)
    : m_numberOfRows(otherMatrix.m_numberOfRows),
      m_numberOfColumns(otherMatrix.m_numberOfColumns),
      m_accumulateNonZeros(otherMatrix.m_accumulateNonZeros),
      m_columnsOfValues(otherMatrix.m_columnsOfValues),
      m_values(otherMatrix.m_values) {}

vector<pair<size_t, complex<double>>> MatrixCRS::getRow(
    const size_t row) const {
  assert(row < m_numberOfRows);

  vector<pair<size_t, complex<double>>> result;
  size_t numberOfElements =
      m_accumulateNonZeros[row + 1] - m_accumulateNonZeros[row];

  if (numberOfElements) {
    result.reserve(numberOfElements);

    for (size_t i = m_accumulateNonZeros[row];
         i < m_accumulateNonZeros[row + 1]; ++i) {
      result.push_back({m_columnsOfValues[i], m_values[i]});
    }
  }

  return result;
}
void MatrixCRS::transponse() {
  vector<vector<pair<size_t, complex<double>>>> intermediateResult(
      m_numberOfColumns);

  for (size_t i = 0; i < m_accumulateNonZeros.size() - 1; ++i) {
    for (size_t el = m_accumulateNonZeros[i]; el < m_accumulateNonZeros[i + 1];
         ++el) {
      intermediateResult[m_columnsOfValues[el]].push_back({i, m_values[el]});
    }
  }

  std::swap(m_numberOfRows, m_numberOfColumns);
  m_accumulateNonZeros.clear();
  m_columnsOfValues.clear();
  m_values.clear();

  m_accumulateNonZeros.push_back(0);

  for (auto& vec : intermediateResult) {
    m_accumulateNonZeros.push_back(
        m_accumulateNonZeros[m_accumulateNonZeros.size() - 1] + vec.size());
    for (auto& elem : vec) {
      m_columnsOfValues.push_back(elem.first);
      m_values.push_back(elem.second);
    }
  }
}

complex<double> multiplicationUnit(
    const vector<pair<size_t, complex<double>>>& first,
    const vector<pair<size_t, complex<double>>>& second) {
  complex<double> result(0.0, 0.0);

  for (auto el : first) {
    auto finded = std::find_if(second.begin(), second.end(),
                               [&el](const pair<size_t, complex<double>>& p) {
                                 return p.first == el.first;
                               });

    if (finded != second.end())
      result += el.second * (*finded).second;
  }

  return result;
}

MatrixCRS MatrixCRS::operator*(const MatrixCRS& otherMatrix) {
  assert(this->m_numberOfColumns == otherMatrix.m_numberOfRows);

  MatrixCRS result(this->m_numberOfRows, otherMatrix.m_numberOfColumns);
  MatrixCRS transponseOther = otherMatrix;
  transponseOther.transponse();

  for (size_t i = 0; i < this->m_numberOfRows; ++i) {
    result.m_accumulateNonZeros.push_back(
        result.m_accumulateNonZeros[result.m_accumulateNonZeros.size() - 1]);
    auto currentRow = this->getRow(i);
    for (size_t j = 0; j < transponseOther.m_numberOfRows; ++j) {
      auto currentCol = transponseOther.getRow(j);
      auto res = multiplicationUnit(currentRow, currentCol);
      if (res != complex<double>(0.0, 0.0)) {
        result.m_columnsOfValues.push_back(j);
        result.m_values.push_back(res);
        result.m_accumulateNonZeros[i + 1]++;
      }
    }
  }

  return result;
}

bool MatrixCRS::operator==(const MatrixCRS& otherMatrix) {
  if (this->m_numberOfRows != otherMatrix.m_numberOfRows ||
      this->m_numberOfColumns != otherMatrix.m_numberOfColumns) {
    return false;
  }

  for (size_t i = 0; i < this->m_accumulateNonZeros.size(); ++i) {
    if (this->m_accumulateNonZeros[i] != otherMatrix.m_accumulateNonZeros[i])
      return false;
  }

  for (size_t i = 0; i < this->m_columnsOfValues.size(); ++i) {
    if (this->m_columnsOfValues[i] != otherMatrix.m_columnsOfValues[i])
      return false;
  }

  for (size_t i = 0; i < this->m_values.size(); ++i) {
    if (this->m_values[i] != otherMatrix.m_values[i])
      return false;
  }

  return true;
}

MatrixCRS getParallelMult(const MatrixCRS& first, const MatrixCRS& second) {
  assert(first.m_numberOfColumns == second.m_numberOfRows);

  MatrixCRS result(first.m_numberOfRows, second.m_numberOfColumns);
  MatrixCRS transponseSecond = second;
  transponseSecond.transponse();

  vector<vector<pair<size_t, complex<double>>>> intermediateResult(
      first.m_numberOfRows);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, first.m_numberOfRows),
                    [&](tbb::blocked_range<size_t> r) {
                      for (size_t i = r.begin(); i < r.end(); ++i) {
                        auto currentRow = first.getRow(i);
                        for (size_t j = 0; j < transponseSecond.m_numberOfRows;
                             ++j) {
                          auto currentCol = transponseSecond.getRow(j);
                          auto res = multiplicationUnit(currentRow, currentCol);
                          if (res != complex<double>(0.0, 0.0)) {
                            intermediateResult[i].push_back({j, res});
                          }
                        }
                      }
                    });

  for (auto& vec : intermediateResult) {
    result.m_accumulateNonZeros.push_back(
        result.m_accumulateNonZeros[result.m_accumulateNonZeros.size() - 1] +
        vec.size());
    for (auto& elem : vec) {
      result.m_columnsOfValues.push_back(elem.first);
      result.m_values.push_back(elem.second);
    }
  }

  return result;
}
