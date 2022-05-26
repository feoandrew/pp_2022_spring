// Copyright 2022 Boyarskov Anton
#include <gtest/gtest.h>

#include "./conjugate_gradient.h"

TEST(CONJUGATE_GRAD_SEQ, TEST_1) {
  int n = 2;

  matrix A;
  vec B;

  B = generateRandomVector(n);
  A = generateMatrix(generateRandomVector(n));

  print("A", A);
  print("B", B);

  std::vector<double> X = conjugateGradientSolver(A, B);

  print("X", X);
  vec check = matrixTimesVector(A, X);
  print("Check AX:", check);

  printError(B, check, n);
}

TEST(CONJUGATE_GRAD_SEQ, TEST_2) {
  int n = 3;

  matrix A;
  vec B;

  B = generateRandomVector(n);
  A = generateMatrix(generateRandomVector(n));

  print("A", A);
  print("B", B);

  std::vector<double> X = conjugateGradientSolver(A, B);

  print("X", X);
  vec check = matrixTimesVector(A, X);
  print("Check AX:", check);

  printError(B, check, n);
}

TEST(CONJUGATE_GRAD_SEQ, TEST_3) {
  int n = 10;

  matrix A;
  vec B;

  B = generateRandomVector(n);
  A = generateMatrix(generateRandomVector(n));

  std::vector<double> X = conjugateGradientSolver(A, B);
  vec check = matrixTimesVector(A, X);

  printError(B, check, n);
}

TEST(CONJUGATE_GRAD_SEQ, TEST_4) {
  int n = 20;

  matrix A;
  vec B;

  B = generateRandomVector(n);
  A = generateMatrix(generateRandomVector(n));

  std::vector<double> X = conjugateGradientSolver(A, B);
  vec check = matrixTimesVector(A, X);

  printError(B, check, n);
}

TEST(CONJUGATE_GRAD_SEQ, TEST_5) {
  int n = 30;

  matrix A;
  vec B;

  B = generateRandomVector(n);
  A = generateMatrix(generateRandomVector(n));

  std::vector<double> X = conjugateGradientSolver(A, B);
  vec check = matrixTimesVector(A, X);

  printError(B, check, n);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
