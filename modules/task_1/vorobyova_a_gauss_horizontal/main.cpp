// Copyright 2022 Vorobyova Anya
#include <gtest/gtest.h>

#include "./gauss_horizontal.h"

TEST(GAUSS_HORIZONTAL_SEQ, TEST_1) {
  const int width = 100, height = 100;
  img image = getRandomImage(width, height);

  img new_image = getGauss(image, width, height);

  ASSERT_EQ(static_cast<int>(new_image.size()), 98);
}

TEST(GAUSS_HORIZONTAL_SEQ, TEST_2) {
  const int width = 200, height = 100;
  img image = getRandomImage(width, height);

  img new_image = getGauss(image, width, height);

  ASSERT_EQ(static_cast<int>(new_image.size()), 98);
}

TEST(GAUSS_HORIZONTAL_SEQ, TEST_3) {
  const int width = 100, height = 200;
  img image = getRandomImage(width, height);

  img new_image = getGauss(image, width, height);

  ASSERT_EQ(static_cast<int>(new_image.size()), 198);
}

TEST(GAUSS_HORIZONTAL_SEQ, TEST_4) {
  const int width = 123, height = 234;
  img image = getRandomImage(width, height);

  img new_image = getGauss(image, width, height);

  ASSERT_EQ(static_cast<int>(new_image.size()), 232);
}

TEST(GAUSS_HORIZONTAL_SEQ, TEST_5) {
  const int width = 500, height = 500;
  img image = getRandomImage(width, height);

  img new_image = getGauss(image, width, height);

  ASSERT_EQ(static_cast<int>(new_image.size()), 498);
}
