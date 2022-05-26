// Copyright 2022 Denisova Julia
#include <gtest/gtest.h>
#include <vector>
#include "./radix_sort_omp.h"


TEST(OpenMP_RadixSort, 321) {
    std::vector<int> p = { 3, 2, 1 };
    parallelSort(&p[0], p.size());
    ASSERT_EQ(p, std::vector<int>({ 1, 2, 3 }));
}

TEST(OpenMP_RadixSort, 1) {
    std::vector<int> a = { 1 };
    parallelSort(&a[0], a.size());
    ASSERT_EQ(a, std::vector<int>({ 1 }));
}
TEST(OpenMP_RadixSort, negative) {
    std::vector<int> a = { -1, -2, -3 };
    parallelSort(&a[0], a.size());
    ASSERT_EQ(a, std::vector<int>({ -3, -2, -1 }));
}
TEST(OpenMP_RadixSort, mixed) {
    std::vector<int> a = { -1, -2, -3, 3, 1, 2, 0 };
    parallelSort(&a[0], a.size());
    ASSERT_EQ(a, std::vector<int>({ -3, -2, -1, 0, 1, 2, 3 }));
}
TEST(OpenMP_RadixSort, random) {
    std::vector<int> a = getRandomVector(1000, 1000000);
    // printVec(a);
    std::vector<int> b = a;
    std::vector<int> p = a;
    // double start_seq = omp_get_wtime();
    radixSort(&a[0], a.size());
    // double stop_seq = omp_get_wtime();
    parallelSort(&p[0], p.size());
    // double stop_par = omp_get_wtime();
    std::sort(std::begin(b), std::end(b));
    // printVec(a);
    ASSERT_EQ(a, b);
    ASSERT_EQ(p, b);
    // double seq = stop_seq-start_seq;
    // double par = stop_par- stop_seq;
    // cout<<seq<<"  "<<par<<"  "<<seq/par<<endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
