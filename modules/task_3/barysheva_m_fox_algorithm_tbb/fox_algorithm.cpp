// Copyright 2022 Barysheva Maria
#include "../../../modules/task_3/barysheva_m_fox_algorithm_tbb/fox_algorithm.h"

#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

std::vector<std::vector<double>> GetRandomMatrix(const int& size) {
  if (size <= 0) {
    throw "Wrong size matrix";
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1000.0);

  std::vector<std::vector<double>> C(size, std::vector<double>(size));
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      C[i][j] = dis(gen);
    }
  }
  return C;
}

std::vector<std::vector<double>> SimpleMultiplication(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {
  if (A.size() != B.size()) {
    throw "Different size";
  }
  if (A.size() <= 0 || B.size() <= 0) {
    throw "Size of matrix must be > 0";
  }

  size_t n = A.size();
  size_t m = B[0].size();
  std::vector<std::vector<double>> C(n, std::vector<double>(n, 0));

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      for (size_t k = 0; k < B.size(); ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C;
}

std::vector<std::vector<double>> BlockMultiplication(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {
  if (A.size() != B.size()) {
    throw "Different size";
  }
  if (A.size() <= 0 || B.size() <= 0) {
    throw "Size of matrix must be > 0";
  }

  size_t n = A.size();
  size_t q = n / std::sqrt(n);
  size_t BlockSize = n / q;
  std::vector<std::vector<double>> C(n, std::vector<double>(n, 0));
  size_t EndA, EndB;
  for (size_t a = 0; a < n; a += BlockSize) {
    EndA = std::min(a + BlockSize, n);
    for (size_t b = 0; b < n; b += BlockSize) {
      EndB = std::min(b + BlockSize, n);
      for (size_t i = 0; i < n; i++) {
        for (size_t j = a; j < EndA; j++) {
          for (size_t k = b; k < EndB; k++) {
            C[i][j] += A[i][k] * B[k][j];
          }
        }
      }
    }
  }
  return C;
}

bool CompareMatrix(const std::vector<std::vector<double>>& A,
                   const std::vector<std::vector<double>>& B) {
  if (A.size() != B.size()) {
    throw "Different size";
  }
  if (A.size() <= 0 || B.size() <= 0) {
    throw "Size of matrix must be > 0";
  }

  bool eq = true;
  for (size_t i = 0; i < A.size(); i++) {
    for (size_t j = 0; j < A[0].size(); j++)
      if (std::fabs(A[i][j] - B[i][j]) >
          std::numeric_limits<double>::epsilon() * std::max(A[i][j], B[i][j]) *
              100)
        eq = false;
  }
  return eq;
}

std::vector<std::vector<double>> FoxParallel(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {
  if (A.size() != B.size()) {
    throw "Different size";
  }
  if (A.size() <= 0 || B.size() <= 0) {
    throw "Size of matrix must be > 0";
  }

  tbb::task_scheduler_init init;
  size_t n = A.size();
  size_t q = std::sqrt(init.default_num_threads());
  size_t BlockSize = n / q;
  size_t nold = n;

  std::vector<std::vector<double>> EndA = A;
  std::vector<std::vector<double>> EndB = B;

  while (n % q != 0) {
    EndA.push_back(std::vector<double>(n, 0));
    EndB.push_back(std::vector<double>(n, 0));
    std::for_each(EndA.begin(), EndA.end(),
                  [](std::vector<double>& a) { a.push_back(0); });
    std::for_each(EndB.begin(), EndB.end(),
                  [](std::vector<double>& a) { a.push_back(0); });
    n++;
    BlockSize = n / q;
  }

  std::vector<std::vector<double>> C(n, std::vector<double>(n));
  BlockSize = n / q;

  tbb::parallel_for(
      tbb::blocked_range2d<size_t>(0, n, BlockSize, 0, n, BlockSize),
      [&](const tbb::blocked_range2d<size_t>& r) {
        std::vector<std::vector<double>> A1(BlockSize), B1(BlockSize);
        std::vector<std::vector<double>> tmp(BlockSize,
                                             std::vector<double>(BlockSize, 0));

        for (size_t m = 0; m < q; m++) {
          for (size_t n = 0; n < BlockSize; n++) {
            A1[n] = std::vector<double>(
                EndA[r.rows().begin() + n].begin() +
                    (((r.rows().begin() / BlockSize) + m) % q) * BlockSize,
                EndA[r.rows().begin() + n].begin() +
                    (((r.rows().begin() / BlockSize) + m) % q) * BlockSize +
                    BlockSize);
            B1[n] = std::vector<double>(
                EndB[(((r.rows().begin() / BlockSize) + m) % q) * BlockSize + n]
                        .begin() +
                    r.cols().begin(),
                EndB[(((r.rows().begin() / BlockSize) + m) % q) * BlockSize + n]
                        .begin() +
                    (r.cols().begin() + BlockSize));
          }
          for (size_t i = 0; i < BlockSize; i++) {
            for (size_t j = 0; j < BlockSize; j++) {
              for (size_t k = 0; k < BlockSize; k++)
                tmp[i][j] += A1[i][k] * B1[k][j];
            }
          }
        }

        for (size_t i = 0; i < BlockSize; i++) {
          for (size_t j = 0; j < BlockSize; j++) {
            C[i + r.rows().begin()][j + r.cols().begin()] = tmp[i][j];
          }
        }
      });

  if (nold != n) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                      [&](const tbb::blocked_range<size_t>& r) {
                        for (size_t i = r.begin(); i < r.end(); i++) {
                          C[i].resize(nold);
                        }
                      });
  }
  C.resize(nold);
  return C;
}
