// Copyright 2022 Korobeinikov Alexandr
#include <gtest/gtest.h>
#include<omp.h>
#include<iostream>
#include "./ParallelRadixSort.h"

TEST(RadixSort, SimpleCase_1) {
    std::vector<double> vec = { 1.123, 1.111, -1.1, -2.313, 34.23 };
    std::vector<double> vec1(vec);
    RadixSort(0, vec.size() - 1, &vec);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(RadixSort, SimpleCase_2) {
    std::vector<double> vec = getRandomVector(10, -100, 100);
    std::vector<double> vec1(vec);

    RadixSort(0, vec.size() - 1, &vec);

    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(RadixSort, NegativeAndPositiveCase_1) {
    std::vector<double> vec = getRandomVector(1000, -100, 100);
    std::vector<double> vec1(vec);

    RadixSort(0, vec.size() - 1, &vec);

    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(RadixSort, NegativeAndPositiveCase_2) {
    std::vector<double> vec = getRandomVector(100000, -100, 100);
    std::vector<double> vec1(vec);

    RadixSort(0, vec.size() - 1, &vec);

    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(ParallelRadixSort, SimpleCase) {
    std::vector<double> vec = { 1.123, 1.111, -1.1, -2.313, 34.23 };
    std::vector<double> vec1(vec);

    RadixSortParallel(&vec);

    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(ParallelRadixSort, NegativeAndPositiveCase_1) {
    std::vector<double> vec = getRandomVector(10000, -100, 100);
    std::vector<double> vec1(vec);
    std::vector<double> vec2(vec);

    double t1, t2;
    t1 = omp_get_wtime();
    RadixSort(0, vec1.size() - 1, &vec1);
    t2 = omp_get_wtime();
    std::cout << "Time_Seq: " << t2 - t1 << "\n";

    t1 = omp_get_wtime();
    RadixSortParallel(&vec2);
    t2 = omp_get_wtime();
    std::cout << "Time_Parallel: " << t2 - t1 << "\n";

    std::sort(vec.begin(), vec.end());
    ASSERT_EQ(vec, vec2);
}

TEST(ParallelRadixSort, NegativeAndPositiveCase_2) {
    std::vector<double> vec = getRandomVector(100000, -100, 100);
    std::vector<double> vec1(vec);
    std::vector<double> vec2(vec);

    double t1, t2;
    t1 = omp_get_wtime();
    RadixSort(0, vec1.size() - 1, &vec1);
    t2 = omp_get_wtime();
    std::cout << "Time_Seq: " << t2 - t1 << "\n";

    t1 = omp_get_wtime();
    RadixSortParallel(&vec2);
    t2 = omp_get_wtime();
    std::cout << "Time_Parallel: " << t2 - t1 << "\n";

    std::sort(vec.begin(), vec.end());
    ASSERT_EQ(vec, vec2);
}

TEST(ParallelRadixSort, NegativeAndPositiveCase_3) {
    std::vector<double> vec = getRandomVector(1000000, -100, 100);
    std::vector<double> vec1(vec);
    std::vector<double> vec2(vec);

    double t1, t2;
    t1 = omp_get_wtime();
    RadixSort(0, vec1.size() - 1, &vec1);
    t2 = omp_get_wtime();
    std::cout << "Time_Seq: " << t2 - t1 << "\n";

    t1 = omp_get_wtime();
    RadixSortParallel(&vec2);
    t2 = omp_get_wtime();
    std::cout << "Time_Parallel: " << t2 - t1 << "\n";

    std::sort(vec.begin(), vec.end());
    ASSERT_EQ(vec, vec2);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
