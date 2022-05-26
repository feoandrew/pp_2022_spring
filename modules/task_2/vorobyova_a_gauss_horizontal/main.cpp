// Copyright 2022 Vorobyova Anya
#include <gtest/gtest.h>

#include "./gauss_horizontal.h"

TEST(GAUSS_HORIZONTAL_OMP, TEST_1) {
  const int width = 100, height = 100;
  img image = getRandomImage(width, height);

  img new_image = getGauss(image, width, height);
  img new_image_omp = getGaussOMP(image, width, height);

  for (int i = 0; i < static_cast<int>(new_image.size()); i++) {
    ASSERT_EQ(new_image[i], new_image_omp[i]);
  }
}

TEST(GAUSS_HORIZONTAL_OMP, TEST_2) {
  const int width = 200, height = 100;
  img image = getRandomImage(width, height);

  img new_image = getGauss(image, width, height);
  img new_image_omp = getGaussOMP(image, width, height);

  for (int i = 0; i < static_cast<int>(new_image.size()); i++) {
    ASSERT_EQ(new_image[i], new_image_omp[i]);
  }
}

TEST(GAUSS_HORIZONTAL_OMP, TEST_3) {
  const int width = 100, height = 200;
  img image = getRandomImage(width, height);

  img new_image = getGauss(image, width, height);
  img new_image_omp = getGaussOMP(image, width, height);

  for (int i = 0; i < static_cast<int>(new_image.size()); i++) {
    ASSERT_EQ(new_image[i], new_image_omp[i]);
  }
}

TEST(GAUSS_HORIZONTAL_OMP, TEST_4) {
  const int width = 123, height = 234;
  img image = getRandomImage(width, height);

  img new_image = getGauss(image, width, height);
  img new_image_omp = getGaussOMP(image, width, height);

  for (int i = 0; i < static_cast<int>(new_image.size()); i++) {
    ASSERT_EQ(new_image[i], new_image_omp[i]);
  }
}

TEST(GAUSS_HORIZONTAL_OMP, TEST_SEQ_TIME) {
  const int width = 500, height = 200;
  img image = getRandomImage(width, height);

  img new_image = getGauss(image, width, height);
}

TEST(GAUSS_HORIZONTAL_OMP, TEST_OMP_TIME) {
  const int width = 500, height = 200;
  img image = getRandomImage(width, height);

  img new_image_omp = getGaussOMP(image, width, height);
}
