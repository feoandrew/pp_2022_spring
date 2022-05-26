// Copyright 2022 Utyugov Denis
#include <gtest/gtest.h>

#include <vector>

#include "./cannons_alg.h"

TEST(getRndMatrix, Create_random_matrix) {
  int size = 3;
  ASSERT_NO_THROW(getRndMatrix(size));
}

TEST(Multiplicate, Mult_matrix_3x3) {
  int size = 3;
  std::vector<std::vector<double>> A = getRndMatrix(size);
  std::vector<std::vector<double>> B = getRndMatrix(size);
  ASSERT_NO_THROW(Multiplicate(A, B));
}

TEST(Multiplicate, Check_Result_On_3x3) {
  int size = 3;
  double sum = 0;
  std::vector<std::vector<double>> A = {{4, -1, 3}, {4, -2, -6}, {2, 0, 3}};
  std::vector<std::vector<double>> B = {{5, 3, -7}, {-1, 6, -3}, {2, -4, 1}};
  std::vector<std::vector<double>> C = Multiplicate(A, B);
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++) sum += C[i][j];
  ASSERT_EQ(sum, 4);
}

TEST(Multiplicate, Check_Result_On_5x5) {
  std::vector<std::vector<double>> A = {{1, 2, 3, 4, 5},
                                        {6, 7, 8, 9, 10},
                                        {11, 12, 13, 14, 15},
                                        {16, 17, 18, 19, 20},
                                        {21, 22, 23, 24, 25}};
  std::vector<std::vector<double>> B = {{1, 2, 3, 4, 5},
                                        {6, 7, 8, 9, 10},
                                        {11, 12, 13, 14, 15},
                                        {16, 17, 18, 19, 20},
                                        {21, 22, 23, 24, 25}};
  std::vector<std::vector<double>> C = Multiplicate(A, B);
  ASSERT_EQ(C[4][4], 1775);
}

TEST(BlockMultiplicate, BlockMult_and_Mult_BlockSize_is_two) {
  std::vector<std::vector<double>> A = {{1, 2, 3, 4, 5},
                                        {6, 7, 8, 9, 10},
                                        {11, 12, 13, 14, 15},
                                        {16, 17, 18, 19, 20},
                                        {21, 22, 23, 24, 25}};
  std::vector<std::vector<double>> B = {{1, 2, 3, 4, 5},
                                        {6, 7, 8, 9, 10},
                                        {11, 12, 13, 14, 15},
                                        {16, 17, 18, 19, 20},
                                        {21, 22, 23, 24, 25}};
  ASSERT_EQ(Multiplicate(A, B), BlockMultiplicate(A, B, 2));
}

TEST(BlockMultiplicate, BlockMult_and_Mult_BlockSize_is_three) {
  std::vector<std::vector<double>> A = {{1, 2, 3, 4, 5},
                                        {6, 7, 8, 9, 10},
                                        {11, 12, 13, 14, 15},
                                        {16, 17, 18, 19, 20},
                                        {21, 22, 23, 24, 25}};
  std::vector<std::vector<double>> B = {{1, 2, 3, 4, 5},
                                        {6, 7, 8, 9, 10},
                                        {11, 12, 13, 14, 15},
                                        {16, 17, 18, 19, 20},
                                        {21, 22, 23, 24, 25}};
  ASSERT_EQ(Multiplicate(A, B), BlockMultiplicate(A, B, 3));
}
