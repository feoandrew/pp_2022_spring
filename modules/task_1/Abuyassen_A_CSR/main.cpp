// Copyright 2022 Abuyassen Albara

#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include "./CSR.h"

typedef std::vector<std::vector<std::complex<int>>> matrix;
typedef std::complex<int> element;

TEST(CSR_seq, can_create_a_matrix_and_sparesify_it) {
    matrix A = {
        {0, element(1, 3)},
        {0, 0},
        {element(5, 5), 0},
    };
    CSR A_ = sparesify(A);
    std::vector<std::complex<int>> check = {element(1, 3), element(5, 5)};
    ASSERT_TRUE(A_.values == check);
}

TEST(CSR_seq, can_get_transpose) {
    matrix A = {
        {0, element(1, 3)},
        {0, 0},
        {element(5, 5), 0},
    };

    CSR A_ = sparesify(A);
    CSR A_t = sparse_transpose(A_);

    CSR check {
        {element(5, 5), element(1, 3)},
        {0, 1, 2},
        {2, 0},
        2,
        3,
        2
    };

    ASSERT_TRUE(A_t == check);
}

TEST(CSR_seq, can_multiply) {
    matrix A = {
        {0, element(1, 3)},
        {0, 0},
        {element(5, 5), 0},
    };
    CSR A_ = sparesify(A);

    matrix B = {
        {0, element(2, 4)},
        {element(1, 3), 0},
    };
    CSR B_ = sparesify(B);

    matrix result = A_ * B_;

    matrix check = {
        {element(-8, 6), 0},
        {0, 0},
        {0, element(-10, 30)},
    };
    ASSERT_TRUE(result == check);
}

TEST(CSR_seq, multiplication_throws_exception) {
    matrix A = {
        {0, element(1, 3)},
        {0, 0},
        {element(5, 5), 0},
    };
    CSR A_ = sparesify(A);

    matrix B = {
        {0, element(2, 4)},
        {element(1, 3), 0},
        {0, 0},
    };
    CSR B_ = sparesify(B);
    EXPECT_ANY_THROW(A_ * B_);
}

TEST(CSR_seq, can_multiply_random_matrix) {
    matrix A = randomSparseMatrix(4, 3);
    CSR A_ = sparesify(A);

    matrix B = randomSparseMatrix(3, 4);
    CSR B_ = sparesify(B);
    EXPECT_NO_THROW(A_ * B_);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
