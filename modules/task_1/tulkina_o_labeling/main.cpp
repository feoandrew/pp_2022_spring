// Copyright 2022 Tulkina Olga

#include <gtest/gtest.h>

#include "../../modules/task_1/tulkina_o_labeling/labeling.h"

TEST(Sequential, Test_1x1) {
  std::vector<std::vector<int>> binary_image = {{0}};

  binary_image = labeling(binary_image);

  std::vector<std::vector<int>> expected = {{0}};
  int a = binary_image.size();
  for (int i = 0; i < a; i++) EXPECT_EQ(expected[i], binary_image[i]);
}

TEST(Sequential, Test_1x5) {
  std::vector<std::vector<int>> binary_image = {{0, 1, 1, 0, 1}};

  binary_image = labeling(binary_image);

  std::vector<std::vector<int>> expected = {{0, 2, 2, 0, 5}};

  int a = binary_image.size();
  for (int i = 0; i < a; i++) EXPECT_EQ(expected[i], binary_image[i]);
}

TEST(Sequential, Test_5x1) {
  std::vector<std::vector<int>> binary_image = {{0}, {1}, {1}, {1}, {1}};

  binary_image = labeling(binary_image);

  std::vector<std::vector<int>> expected = {{0}, {2}, {2}, {2}, {2}};
  int a = binary_image.size();
  for (int i = 0; i < a; i++) EXPECT_EQ(expected[i], binary_image[i]);
}

TEST(Sequential, Test_2x10) {
  std::vector<std::vector<int>> binary_image = {{0, 1, 1, 0, 1, 0, 0, 0, 0, 0},
                                                {1, 0, 1, 1, 1, 1, 1, 1, 0, 0}};

  binary_image = labeling(binary_image);

  std::vector<std::vector<int>> expected = {{0, 2, 2, 0, 2, 0, 0, 0, 0, 0},
                                            {2, 0, 2, 2, 2, 2, 2, 2, 0, 0}};
  int a = binary_image.size();
  for (int i = 0; i < a; i++) EXPECT_EQ(expected[i], binary_image[i]);
}

TEST(Sequential, Test_10x10) {
  std::vector<std::vector<int>> binary_image = {
      {0, 1, 1, 0, 1, 0, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0, 0, 0, 1, 0},
      {1, 0, 1, 1, 1, 1, 1, 1, 0, 0}, {1, 0, 0, 0, 0, 1, 1, 0, 1, 0},
      {1, 0, 0, 0, 0, 0, 0, 1, 0, 1}, {1, 0, 0, 1, 0, 1, 0, 1, 0, 0},
      {1, 0, 1, 0, 1, 1, 1, 0, 0, 1}, {1, 0, 0, 1, 1, 1, 0, 0, 1, 1},
      {1, 0, 0, 0, 1, 0, 0, 0, 1, 1}, {1, 0, 0, 0, 0, 0, 1, 1, 1, 1}};

  binary_image = labeling(binary_image);

  std::vector<std::vector<int>> expected = {
      {0, 2, 2, 0, 2, 0, 0, 0, 0, 0},    {0, 0, 2, 2, 0, 0, 0, 0, 2, 0},
      {21, 0, 2, 2, 2, 2, 2, 2, 0, 0},   {21, 0, 0, 0, 0, 2, 2, 0, 2, 0},
      {21, 0, 0, 0, 0, 0, 0, 2, 0, 2},   {21, 0, 0, 2, 0, 2, 0, 2, 0, 0},
      {21, 0, 2, 0, 2, 2, 2, 0, 0, 70},  {21, 0, 0, 2, 2, 2, 0, 0, 70, 70},
      {21, 0, 0, 0, 2, 0, 0, 0, 70, 70}, {21, 0, 0, 0, 0, 0, 70, 70, 70, 70}};
  int a = binary_image.size();
  for (int i = 0; i < a; i++) EXPECT_EQ(expected[i], binary_image[i]);
}
