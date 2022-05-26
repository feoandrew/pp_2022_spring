// Copyright 2022 Bakina Kseniia
#include <gtest/gtest.h>

#include "./ccs_matrix_mult.h"

TEST(Bakina_K_ccs_matrix_mult, check_matrix_transposition_small_size) {
    int n = 10, m = 15;
    std::vector<std::vector<double>> A = get_random_matrix(n, m);
    CCS_matrix A_ccs(convet_to_ccs(A));

    CCS_matrix AT_ccs(ccs_matrix_transposition(A_ccs));
    CCS_matrix AT_check(ccs_matrix_transposition_omp(A_ccs));

    EXPECT_TRUE(AT_check == AT_ccs);
}

TEST(Bakina_K_ccs_matrix_mult, check_matrix_transposition_medium_size) {
    int n = 250, m = 300;
    std::vector<std::vector<double>> A = get_random_matrix(n, m);
    CCS_matrix A_ccs(convet_to_ccs(A));

    CCS_matrix AT_ccs(ccs_matrix_transposition(A_ccs));
    CCS_matrix AT_check(ccs_matrix_transposition_omp(A_ccs));

    EXPECT_TRUE(AT_check == AT_ccs);
}

TEST(Bakina_K_ccs_matrix_mult, check_matrix_transposition_large_size) {
    int n = 200, m = 500;
    std::vector<std::vector<double>> A = get_random_matrix(n, m);
    CCS_matrix A_ccs(convet_to_ccs(A));

    CCS_matrix AT_ccs(ccs_matrix_transposition(A_ccs));
    CCS_matrix AT_check(ccs_matrix_transposition_omp(A_ccs));

    EXPECT_TRUE(AT_check == AT_ccs);
}

TEST(Bakina_K_ccs_matrix_mult, check_matrix_multiplication_small_size) {
    int n = 10, m = 15, k = 20;
    std::vector<std::vector<double>> A = get_random_matrix(n, m);
    std::vector<std::vector<double>> B = get_random_matrix(k, n);
    CCS_matrix A_ccs(convet_to_ccs(A));
    CCS_matrix B_ccs(convet_to_ccs(B));

    // double t1_s = omp_get_wtime();
    CCS_matrix C_ccs(ccs_matrix_multplication(A_ccs, B_ccs));
    // double t2_s = omp_get_wtime();

    // double t1 = omp_get_wtime();
    CCS_matrix C_check(ccs_matrix_multplication_omp(A_ccs, B_ccs));
    // double t2 = omp_get_wtime();

    // t2 = t2 - t1;
    // t2_s = t2_s - t1_s;
    // std::cout << "Sequential: " << t2_s << "\n";
    // std::cout << "OpenMP: " << t2 << "\n";
    // std::cout << "Sequential/OpenMP " << t2_s / t2 << "\n";

    EXPECT_TRUE(C_ccs == C_check);
}

TEST(Bakina_K_ccs_matrix_mult, check_matrix_multiplication_medium_size) {
    int n = 250, m = 100, k = 350;
    std::vector<std::vector<double>> A = get_random_matrix(n, m);
    std::vector<std::vector<double>> B = get_random_matrix(k, n);
    CCS_matrix A_ccs(convet_to_ccs(A));
    CCS_matrix B_ccs(convet_to_ccs(B));

    // double t1_s = omp_get_wtime();
    CCS_matrix C_ccs(ccs_matrix_multplication(A_ccs, B_ccs));
    // double t2_s = omp_get_wtime();

    // double t1 = omp_get_wtime();
    CCS_matrix C_check(ccs_matrix_multplication_omp(A_ccs, B_ccs));
    // double t2 = omp_get_wtime();

    // t2 = t2 - t1;
    // t2_s = t2_s - t1_s;
    // std::cout << "Sequential: " << t2_s << "\n";
    // std::cout << "OpenMP: " << t2 << "\n";
    // std::cout << "Sequential/OpenMP " << t2_s / t2 << "\n";

    EXPECT_TRUE(C_ccs == C_check);
}

TEST(Bakina_K_ccs_matrix_mult, check_matrix_multiplication_large_size) {
    int n = 1000, m = 500, k = 250;
    std::vector<std::vector<double>> A = get_random_matrix(n, m);
    std::vector<std::vector<double>> B = get_random_matrix(k, n);
    CCS_matrix A_ccs(convet_to_ccs(A));
    CCS_matrix B_ccs(convet_to_ccs(B));

    // double t1_s = omp_get_wtime();
    CCS_matrix C_ccs(ccs_matrix_multplication(A_ccs, B_ccs));
    // double t2_s = omp_get_wtime();

    // double t1 = omp_get_wtime();
    CCS_matrix C_check(ccs_matrix_multplication_omp(A_ccs, B_ccs));
    // double t2 = omp_get_wtime();

    // t2 = t2 - t1;
    // t2_s = t2_s - t1_s;
    // std::cout << "Sequential: " << t2_s << "\n";
    // std::cout << "OpenMP: " << t2 << "\n";
    // std::cout << "Sequential/OpenMP " << t2_s / t2 << "\n";

    EXPECT_TRUE(C_ccs == C_check);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
