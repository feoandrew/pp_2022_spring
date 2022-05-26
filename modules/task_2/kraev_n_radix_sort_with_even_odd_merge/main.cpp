// Copyright 2022 Kraev Nikita
#include <gtest/gtest.h>
#include "./radix_sort_with_even_odd_merge.h"

TEST(Sequential_Sort, Test_Radix_Sort_1) {
    std::vector<double> vec = { 1.0, -6.2, 7.45, -22.3457, 0.0, 45.69876, -0.03 };
    std::vector<double> sort_vec = { -22.3457, -6.2, -0.03, 0.0, 1.0, 7.45, 45.69876 };
    radixSort(&vec, 0, vec.size());
    ASSERT_EQ(sort_vec, vec);
}

TEST(Parallel_Sort, Test_Radix_Sort_2) {
    std::vector<double> vec = { 1.0, -6.2, 7.45, -22.3457, 0.0, 45.69876, -0.03 };
    std::vector<double> sort_vec = { -22.3457, -6.2, -0.03, 0.0, 1.0, 7.45, 45.69876 };
    int number_threads = 2;
    parallelRadixSort(&vec, number_threads, vec.size());
    ASSERT_EQ(sort_vec, vec);
}

TEST(Parallel_Sort, Test_Radix_Sort_3) {
    int size = 100;
    std::vector<double> vec = getRandomVector(size);
    std::vector<double> std_sort_vec = vec;
    int number_threads = 2;
    std::sort(std_sort_vec.begin(), std_sort_vec.end());
    parallelRadixSort(&vec, number_threads, vec.size());
    ASSERT_EQ(std_sort_vec, vec);
}

TEST(Parallel_Sort, Test_Radix_Sort_4) {
    int size = 200;
    std::vector<double> vec = getRandomVector(size);
    std::vector<double> seq_vec = vec;
    int number_threads = 2;
    radixSort(&seq_vec, 0, seq_vec.size());
    parallelRadixSort(&vec, number_threads, vec.size());
    ASSERT_EQ(seq_vec, vec);
}

TEST(Parallel_Sort, Test_Radix_Sort_5) {
    int size = 300;
    std::vector<double> vec = getRandomVector(size);
    std::vector<double> seq_vec = vec;
    int number_threads = 4;
    radixSort(&seq_vec, 0, seq_vec.size());
    parallelRadixSort(&vec, number_threads, vec.size());
    ASSERT_EQ(seq_vec, vec);
}

TEST(Parallel_Sort, Test_Radix_Sort_6) {
    int size = 400;
    std::vector<double> vec = getRandomVector(size);
    std::vector<double> seq_vec = vec;
    int number_threads = 8;
    radixSort(&seq_vec, 0, seq_vec.size());
    parallelRadixSort(&vec, number_threads, vec.size());
    ASSERT_EQ(seq_vec, vec);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
