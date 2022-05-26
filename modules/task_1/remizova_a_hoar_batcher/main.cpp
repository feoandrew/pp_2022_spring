// Copyright 2022 Remizova Antonina
#include <gtest/gtest.h>
#include "../../../modules/task_1/remizova_a_hoar_batcher/hoar_batcher.h"

TEST(SEQ_hoar_batcher, can_create_rand_vec_10) {
    int size = 10;

    EXPECT_NO_THROW(getRandVector(size));
}

TEST(SEQ_hoar_batcher, can_create_rand_vec_10000) {
    int size = 10000;

    EXPECT_NO_THROW(getRandVector(size));
}

TEST(SEQ_hoar_batcher, can_not_create_rand_vec_with_negative_size) {
    int size = -10;

    EXPECT_ANY_THROW(getRandVector(size));
}

TEST(SEQ_hoar_batcher, can_hoar_sort_rand_vec) {
    int size = 231;
    std::vector<double> v = getRandVector(size);
    hoarSort(&v, 0, size - 1);
    EXPECT_EQ(1, isSortCorrectly(v));
}

TEST(SEQ_hoar_batcher, can_even_batcher_sort_true_vec) {
    std::vector<double> v1{ 9, 7, -6, 4, 0, 2, 13 };
    std::vector<double> v2{ -5, 9, -2, 1, 17, 4, -8 };
    std::vector<double> res{-5, -2, 9, -6, 0, 13, 17, -8};
    EXPECT_EQ(res, evenBatcherSort(v1, v2));
}

TEST(SEQ_hoar_batcher, can_odd_batcher_sort_true_vec) {
    std::vector<double> v1{ 9, 7, -6, 4, 0, 2, 13 };
    std::vector<double> v2{ -5, 9, -2, 1, 17, 4, -8 };
    std::vector<double> res{ 7, 4, 2, 9, 1, 4 };
    EXPECT_EQ(res, oddBatcherSort(v1, v2));
}

TEST(SEQ_hoar_batcher, can_odd_eveb_batcher_sort_true_vec) {
    std::vector<double> v1{ 9, 7, -6 };
    std::vector<double> v2{ -5, 9, -2};
    std::vector<double> res{ 9, 7, -6, -5, 9, -2 };
    EXPECT_EQ(res, oddEvenBatcherSort(v1, v2));
}

TEST(SEQ_hoar_batcher, can_merge_batcher_sort_true_vec) {
    std::vector<double> v1{ 9, 7, -6, 4, 0 };
    std::vector<double> v2{ -5, 9, -2, 1, 17 };
    std::vector<double> even = evenBatcherSort(v1, v2);
    std::vector<double> odd = oddBatcherSort(v1, v2);
    std::vector<double> odd_even_res = oddEvenBatcherSort(even, odd);
    std::vector<double> res{ -5, -2, 4, 7, 9, -6, 1, 0, 9, 17 };
    EXPECT_EQ(res, mergeBatcher(odd_even_res, even.size()));
}

TEST(SEQ_hoar_batcher, can_hoar_batcher_sort_zero_vec) {
    std::vector<double> v{0, 0, 1, 0, 0, 0, 0, 0};
    int size = v.size();
    std::vector<double> res = hoarBatcher(v, size);
    EXPECT_EQ(1, isSortCorrectly(res));
}

TEST(SEQ_hoar_batcher, can_hoar_batcher_sort_vec_with_even_size) {
    int size = 400;
    std::vector<double> v = getRandVector(size);
    std::vector<double> res = hoarBatcher(v, v.size());
    EXPECT_EQ(1, isSortCorrectly(res));
}

TEST(SEQ_hoar_batcher, can_hoar_batcher_sort_vec_with_odd_size) {
    int size = 1331;
    std::vector<double> v = getRandVector(size);
    std::vector<double> res = hoarBatcher(v, v.size());
    EXPECT_EQ(1, isSortCorrectly(res));
}

TEST(SEQ_hoar_batcher, can_hoar_batcher_sort_vec_with_double_elem) {
    int size = 9081;
    std::vector<double> v = getRandVector(size);
    v[740] = 4.5691;
    v[2943] = -17.89;
    std::vector<double> res = hoarBatcher(v, v.size());
    EXPECT_EQ(1, isSortCorrectly(res));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
