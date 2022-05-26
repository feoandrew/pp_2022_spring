  // Copyright 2022 Butescu Vladimir

#include <tbb/tick_count.h>
#include <tbb/parallel_for.h>
#include <gtest/gtest.h>
#include <vector>
#include "../../../modules/task_3/butescu_v_gauss_vert_tbb/vert_gauss.h"

TEST(Gaussian_Filter_TBB, Zero_Rows_Cols) {
    int rows = 0;
    int cols = 0;
    std::vector<double> matrix(rows * cols);

    ASSERT_ANY_THROW(matrix = gauss_filter_parralel_tbb(matrix, rows, cols));
}

TEST(Gaussian_Filter_TBB, Zero_Matrix) {
    int rows = 3;
    int cols = 3;
    std::vector<double> matrix(0);

    ASSERT_ANY_THROW(matrix = gauss_filter_parralel_tbb(matrix, rows, cols));
}

TEST(Gaussian_Filter_TBB, Test_Random_Matrix_Size_25x25) {
    int rows = 25;
    int cols = 25;
    std::vector<double> matrix = createRandomMatrix(rows, cols);

    tbb::tick_count start_seq = tbb::tick_count::now();
    std::vector <double> sq = gauss_filter_sequence(matrix, cols, rows);
    double seq_time = (tbb::tick_count::now() - start_seq).seconds();

    tbb::tick_count start_tbb = tbb::tick_count::now();
    std::vector <double> pp = gauss_filter_parralel_tbb(matrix, cols, rows);
    double tbb_time = (tbb::tick_count::now() - start_tbb).seconds();

    std::cout << "SQ TIME: " << seq_time << std::endl;
    std::cout << "PP TIME: " << tbb_time << std::endl;

    ASSERT_EQ(sq, pp);
}

TEST(Gaussian_Filter_TBB, Test_Random_Matrix_Size_100x100) {
    int rows = 100;
    int cols = 100;
    std::vector<double> matrix = createRandomMatrix(rows, cols);

    tbb::tick_count start_seq = tbb::tick_count::now();
    std::vector <double> sq = gauss_filter_sequence(matrix, cols, rows);
    double seq_time = (tbb::tick_count::now() - start_seq).seconds();

    tbb::tick_count start_tbb = tbb::tick_count::now();
    std::vector <double> pp = gauss_filter_parralel_tbb(matrix, cols, rows);
    double tbb_time = (tbb::tick_count::now() - start_tbb).seconds();

    std::cout << "SQ TIME: " << seq_time << std::endl;
    std::cout << "PP TIME: " << tbb_time << std::endl;

    ASSERT_EQ(sq, pp);
}

TEST(Gaussian_Filter_TBB, Test_Random_Matrix_Size_250x250) {
    int rows = 250;
    int cols = 250;
    std::vector<double> matrix = createRandomMatrix(rows, cols);

    tbb::tick_count start_seq = tbb::tick_count::now();
    std::vector <double> sq = gauss_filter_sequence(matrix, cols, rows);
    double seq_time = (tbb::tick_count::now() - start_seq).seconds();

    tbb::tick_count start_tbb = tbb::tick_count::now();
    std::vector <double> pp = gauss_filter_parralel_tbb(matrix, cols, rows);
    double tbb_time = (tbb::tick_count::now() - start_tbb).seconds();

    std::cout << "SQ TIME: " << seq_time << std::endl;
    std::cout << "PP TIME: " << tbb_time << std::endl;

    ASSERT_EQ(sq, pp);
}

TEST(Gaussian_Filter_TBB, Test_Random_Matrix_Size_500x500) {
    int rows = 500;
    int cols = 500;
    std::vector<double> matrix = createRandomMatrix(rows, cols);

    tbb::tick_count start_seq = tbb::tick_count::now();
    std::vector <double> sq = gauss_filter_sequence(matrix, cols, rows);
    double seq_time = (tbb::tick_count::now() - start_seq).seconds();

    tbb::tick_count start_tbb = tbb::tick_count::now();
    std::vector <double> pp = gauss_filter_parralel_tbb(matrix, cols, rows);
    double tbb_time = (tbb::tick_count::now() - start_tbb).seconds();

    std::cout << "SQ TIME: " << seq_time << std::endl;
    std::cout << "PP TIME: " << tbb_time << std::endl;

    ASSERT_EQ(sq, pp);
}

TEST(Gaussian_Filter_TBB, Test_Random_Matrix_Size_1000x1000) {
    int rows = 1000;
    int cols = 1000;
    std::vector<double> matrix = createRandomMatrix(rows, cols);

    tbb::tick_count start_seq = tbb::tick_count::now();
    std::vector <double> sq = gauss_filter_sequence(matrix, cols, rows);
    double seq_time = (tbb::tick_count::now() - start_seq).seconds();

    tbb::tick_count start_tbb = tbb::tick_count::now();
    std::vector <double> pp = gauss_filter_parralel_tbb(matrix, cols, rows);
    double tbb_time = (tbb::tick_count::now() - start_tbb).seconds();

    std::cout << "SQ TIME: " << seq_time << std::endl;
    std::cout << "PP TIME: " << tbb_time << std::endl;

    ASSERT_EQ(sq, pp);
}

TEST(Gaussian_Filter_TBB, Test_Random_Matrix_Size_5000x5000) {
    int rows = 1000;
    int cols = 1000;
    std::vector<double> matrix = createRandomMatrix(rows, cols);

    tbb::tick_count start_seq = tbb::tick_count::now();
    std::vector <double> sq = gauss_filter_sequence(matrix, cols, rows);
    double seq_time = (tbb::tick_count::now() - start_seq).seconds();

    tbb::tick_count start_tbb = tbb::tick_count::now();
    std::vector <double> pp = gauss_filter_parralel_tbb(matrix, cols, rows);
    double tbb_time = (tbb::tick_count::now() - start_tbb).seconds();

    std::cout << "SQ TIME: " << seq_time << std::endl;
    std::cout << "PP TIME: " << tbb_time << std::endl;

    ASSERT_EQ(sq, pp);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
