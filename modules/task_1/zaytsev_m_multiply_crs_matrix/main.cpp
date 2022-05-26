// Copyright 2022 Zaytsev Mikhail
#include <gtest/gtest.h>
#include <complex>
#include <random>
#include <vector>
#include "./multiply_crs_matrix.h"

std::vector<std::vector<std::pair<size_t, std::complex<double>>>>
getRandomVector(const size_t rows, const size_t cols) {
  std::vector<std::vector<std::pair<size_t, std::complex<double>>>> result(
      rows);

  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> urd(0, 1);

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      if (urd(gen) > 0.70) {
        result[i].push_back(
            {j, std::complex<double>(gen() % 100, gen() % 100)});
      }
    }
  }

  return result;
}

TEST(SequentialMultiply, Matrix3x3) {
  size_t numberOfRows = 3;
  size_t numberOfColumns = 3;

  MatrixCRS firstMatrix(numberOfRows, numberOfColumns,
                        getRandomVector(numberOfRows, numberOfColumns));
  MatrixCRS secondMatrix(numberOfRows, numberOfColumns,
                         getRandomVector(numberOfRows, numberOfColumns));

  ASSERT_NO_THROW(firstMatrix * secondMatrix);
}

TEST(SequentialMultiply, Matrix4x4) {
  size_t numberOfRows = 4;
  size_t numberOfColumns = 4;

  MatrixCRS firstMatrix(numberOfRows, numberOfColumns,
                        getRandomVector(numberOfRows, numberOfColumns));
  MatrixCRS secondMatrix(numberOfRows, numberOfColumns,
                         getRandomVector(numberOfRows, numberOfColumns));

  ASSERT_NO_THROW(firstMatrix * secondMatrix);
}

TEST(SequentialMultiply, Matrix5x5) {
  size_t numberOfRows = 5;
  size_t numberOfColumns = 5;

  MatrixCRS firstMatrix(numberOfRows, numberOfColumns,
                        getRandomVector(numberOfRows, numberOfColumns));
  MatrixCRS secondMatrix(numberOfRows, numberOfColumns,
                         getRandomVector(numberOfRows, numberOfColumns));

  ASSERT_NO_THROW(firstMatrix * secondMatrix);
}

TEST(SequentialMultiply, Matrix6x6) {
  size_t numberOfRows = 6;
  size_t numberOfColumns = 6;

  MatrixCRS firstMatrix(numberOfRows, numberOfColumns,
                        getRandomVector(numberOfRows, numberOfColumns));
  MatrixCRS secondMatrix(numberOfRows, numberOfColumns,
                         getRandomVector(numberOfRows, numberOfColumns));

  ASSERT_NO_THROW(firstMatrix * secondMatrix);
}

TEST(SequentialMultiply, Matrix7x7) {
  size_t numberOfRows = 7;
  size_t numberOfColumns = 7;

  MatrixCRS firstMatrix(numberOfRows, numberOfColumns,
                        getRandomVector(numberOfRows, numberOfColumns));
  MatrixCRS secondMatrix(numberOfRows, numberOfColumns,
                         getRandomVector(numberOfRows, numberOfColumns));

  ASSERT_NO_THROW(firstMatrix * secondMatrix);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
