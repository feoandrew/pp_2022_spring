// Copyright 2022 Pichugin Ilya
#include <omp.h>

#include "../../../modules/task_3/pichugin_i_sobel_tbb/operator_sobel.h"
#include "gtest/gtest.h"

TEST(TEST_SOBOL, SIZE_MATR_500_OMP) {
  size_t size = 500;
  double res_time;
  Sobel s1, s2, rand;
  tbb::tick_count time1, time2;
  rand = rand.RandomMatrix(size);

  time1 = tbb::tick_count::now();
  s1.operator_Sobel(rand);
  time2 = tbb::tick_count::now();
  res_time = (time2 - time1).seconds();
  std::cout.precision(4);
  std::cout << "Execution time of the SEQ algorithm: " << res_time << std::endl;

  time1 = tbb::tick_count::now();
  s2.operator_Sobel_tbb(rand);
  time2 = tbb::tick_count::now();
  res_time = (time2 - time1).seconds();
  std::cout << "Execution time of the TBB algorithm: " << res_time << std::endl;

  ASSERT_EQ(s1.operator_Sobel(rand) == s2.operator_Sobel_tbb(rand), true);
}

TEST(TEST_SOBOL, SIZE_MATR_1000_OMP) {
  size_t size = 1000;
  double res_time;
  Sobel s1, s2, rand;
  tbb::tick_count time1, time2;
  rand = rand.RandomMatrix(size);

  time1 = tbb::tick_count::now();
  s1.operator_Sobel(rand);
  time2 = tbb::tick_count::now();
  res_time = (time2 - time1).seconds();
  std::cout.precision(4);
  std::cout << "Execution time of the SEQ algorithm: " << res_time << std::endl;

  time1 = tbb::tick_count::now();
  s2.operator_Sobel_tbb(rand);
  time2 = tbb::tick_count::now();
  res_time = (time2 - time1).seconds();
  std::cout << "Execution time of the TBB algorithm: " << res_time << std::endl;

  ASSERT_EQ(s1.operator_Sobel(rand) == s2.operator_Sobel_tbb(rand), true);
}

TEST(TEST_SOBOL, SIZE_MATR_1500_OMP) {
  size_t size = 1500;
  double res_time;
  Sobel s1, s2, rand;
  tbb::tick_count time1, time2;
  rand = rand.RandomMatrix(size);

  time1 = tbb::tick_count::now();
  s1.operator_Sobel(rand);
  time2 = tbb::tick_count::now();
  res_time = (time2 - time1).seconds();
  std::cout.precision(4);
  std::cout << "Execution time of the SEQ algorithm: " << res_time << std::endl;

  time1 = tbb::tick_count::now();
  s2.operator_Sobel_tbb(rand);
  time2 = tbb::tick_count::now();
  res_time = (time2 - time1).seconds();
  std::cout << "Execution time of the TBB algorithm: " << res_time << std::endl;

  ASSERT_EQ(s1.operator_Sobel(rand) == s2.operator_Sobel_tbb(rand), true);
}

TEST(TEST_SOBOL, SIZE_MATR_2000_OMP) {
  size_t size = 2000;
  double res_time;
  Sobel s1, s2, rand;
  tbb::tick_count time1, time2;
  rand = rand.RandomMatrix(size);

  time1 = tbb::tick_count::now();
  s1.operator_Sobel(rand);
  time2 = tbb::tick_count::now();
  res_time = (time2 - time1).seconds();
  std::cout.precision(4);
  std::cout << "Execution time of the SEQ algorithm: " << res_time << std::endl;

  time1 = tbb::tick_count::now();
  s2.operator_Sobel_tbb(rand);
  time2 = tbb::tick_count::now();
  res_time = (time2 - time1).seconds();
  std::cout << "Execution time of the TBB algorithm: " << res_time << std::endl;

  ASSERT_EQ(s1.operator_Sobel(rand) == s2.operator_Sobel_tbb(rand), true);
}

TEST(TEST_SOBOL, SIZE_MATR_3000_OMP) {
  size_t size = 3000;
  double res_time;
  Sobel s1, s2, rand;
  tbb::tick_count time1, time2;
  rand = rand.RandomMatrix(size);

  time1 = tbb::tick_count::now();
  s1.operator_Sobel(rand);
  time2 = tbb::tick_count::now();
  res_time = (time2 - time1).seconds();
  std::cout.precision(4);
  std::cout << "Execution time of the SEQ algorithm: " << res_time << std::endl;

  time1 = tbb::tick_count::now();
  s2.operator_Sobel_tbb(rand);
  time2 = tbb::tick_count::now();
  res_time = (time2 - time1).seconds();
  std::cout << "Execution time of the TBB algorithm: " << res_time << std::endl;

  ASSERT_EQ(s1.operator_Sobel(rand) == s2.operator_Sobel_tbb(rand), true);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  tbb::task_scheduler_init init(2);
  return RUN_ALL_TESTS();
}
