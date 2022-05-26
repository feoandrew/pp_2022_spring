// Copyright 2022 Korobeinikov Alexandr
#include <gtest/gtest.h>
#include "./RadixSort.h"

TEST(radixSort, SimpleCase_1) {
    std::vector<double> vec = { 1.123, 1.111, -1.1, -2.313, 34.23 };
    std::vector<double> vec1(vec);
    vec = RadixSort(vec);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(radixSort, SimpleCase_2) {
    std::vector<double> vec = { 1.12345, 1.12344, -2.123, -2.124 };
    std::vector<double> vec1(vec);
    vec = RadixSort(vec);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(radixSort, SimpleCasePositiveNumbers) {
    std::vector<double> vec = getRandomVector(10, 0, 10);
    std::vector<double> vec1(vec);
    vec = RadixSort(vec);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(radixSort, SimpleCaseNegativeNumbers) {
    std::vector<double> vec = getRandomVector(10, -10, 0);
    std::vector<double> vec1(vec);
    vec = RadixSort(vec);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(radixSort, CasePositiveNumbers_1) {
    std::vector<double> vec = getRandomVector(1000, 0, 500);
    std::vector<double> vec1(vec);
    vec = RadixSort(vec);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(radixSort, CasePositiveNumbers_2) {
    std::vector<double> vec = getRandomVector(10000, 0, 500);
    std::vector<double> vec1(vec);
    vec = RadixSort(vec);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(radixSort, CasePositiveNumbers_3) {
    std::vector<double> vec = getRandomVector(100000, 0, 500);
    std::vector<double> vec1(vec);
    vec = RadixSort(vec);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(radixSort, CasePositiveNumbers_4) {
    std::vector<double> vec = getRandomVector(100000, 0, 10);
    std::vector<double> vec1(vec);
    vec = RadixSort(vec);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(radixSort, CaseNegativeNumbers_1) {
    std::vector<double> vec = getRandomVector(1000, -500, 0);
    std::vector<double> vec1(vec);
    vec = RadixSort(vec);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(radixSort, CaseNegativeNumbers_2) {
    std::vector<double> vec = getRandomVector(10000, -500, 0);
    std::vector<double> vec1(vec);
    vec = RadixSort(vec);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(radixSort, CaseNegativeNumbers_3) {
    std::vector<double> vec = getRandomVector(100000, -500, 0);
    std::vector<double> vec1(vec);
    vec = RadixSort(vec);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(radixSort, CaseNegativeNumbers_4) {
    std::vector<double> vec = getRandomVector(100000, -10, 0);
    std::vector<double> vec1(vec);
    vec = RadixSort(vec);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(radixSort, CaseNegativeAndPositiveNumbers_1) {
    std::vector<double> vec = getRandomVector(1000, -10, 10);
    std::vector<double> vec1(vec);
    vec = RadixSort(vec);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(radixSort, CaseNegativeAndPositiveNumbers_2) {
    std::vector<double> vec = getRandomVector(10000, -10, 10);
    std::vector<double> vec1(vec);
    vec = RadixSort(vec);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}

TEST(radixSort, CaseNegativeAndPositiveNumbers_3) {
    std::vector<double> vec = getRandomVector(100000, -10, 10);
    std::vector<double> vec1(vec);
    vec = RadixSort(vec);
    std::sort(vec1.begin(), vec1.end());
    ASSERT_EQ(vec, vec1);
}
