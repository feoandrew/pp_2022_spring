// Copyright 2022 Pichugin Ilya
#include "../../../modules/task_1/pichugin_i_sobel/operator_sobel.h"
#include "gtest/gtest.h"

TEST(TEST_SOBOL, SIZE_MATR_100) {
  Sobel s1;
  size_t size = 100;
  ASSERT_NO_FATAL_FAILURE(s1.operator_Sobel(s1.RandomMatrix(size)));
}
TEST(TEST_SOBOL, SIZE_MATR_500) {
  Sobel s1;
  size_t size = 500;
  ASSERT_NO_FATAL_FAILURE(s1.operator_Sobel(s1.RandomMatrix(size)));
}
TEST(TEST_SOBOL, SIZE_MATR_1000) {
  Sobel s1;
  size_t size = 1000;
  ASSERT_NO_FATAL_FAILURE(s1.operator_Sobel(s1.RandomMatrix(size)));
}
TEST(TEST_SOBOL, SIZE_MATR_2000) {
  Sobel s1;
  size_t size = 2000;
  ASSERT_NO_FATAL_FAILURE(s1.operator_Sobel(s1.RandomMatrix(size)));
}
TEST(TEST_SOBOL, SIZE_MATR_3000) {
  Sobel s1;
  size_t size = 3000;
  ASSERT_NO_FATAL_FAILURE(s1.operator_Sobel(s1.RandomMatrix(size)));
}
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
