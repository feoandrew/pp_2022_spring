// Copyright 2022 Olynin Alexander
#include <gtest/gtest.h>
#include <time.h>
#include <vector>
#include "../../modules/task_3/olynin_a_mult_sparse_cc_complex_mat/mult_sparse_cc_complex_mat.h"

TEST(Class_Complex, Complex_creation) {
    EXPECT_NO_THROW(Complex A(1.2, 2.3));
}

TEST(Class_Complex, Complex_getters) {
    Complex A(1.2, 2.3);

    EXPECT_EQ(A.GetRl(), 1.2);
    EXPECT_EQ(A.GetIm(), 2.3);
}

TEST(Class_Complex, Complex_comparison) {
    Complex A(1.2, 2.3);
    Complex B(A);

    EXPECT_TRUE(A == B);
}

TEST(Class_Complex, Complex_multiplication) {
    Complex A(1.2, 2.3);
    Complex B(3.1, -3.2);
    Complex Res = A * B;
    Complex Res_exp(11.08, 3.29);

    EXPECT_TRUE(Res == Res_exp);
}

TEST(Class_Matrix, Sparse_matrix_creation) {
    int non = 6;
    int size = 4;

    std::vector<Complex> Ent;
    Complex Tmp;
    Ent = Tmp.InitVec({ 9, 3, 8, 15, 7, 16 });
    std::vector<int> irows = { 4, 1, 2, 4, 1, 4 };
    std::vector<int> shtcol = { 1, 2, 3, 5, 7 };

    EXPECT_NO_THROW(Matrix A(size, non, Ent, irows, shtcol));
}

TEST(Class_Matrix, Sparse_matrix_comparison) {
    int non = 6;
    int size = 4;
    Complex Tmp;

    std::vector<Complex> Ent;
    Ent = Tmp.InitVec({ 9, 3, 8, 15, 7, 16 });
    std::vector<int> irows = { 4, 1, 2, 4, 1, 4 };
    std::vector<int> shtcol = { 1, 2, 3, 5, 7 };
    Matrix A(size, non, Ent, irows, shtcol);
    Matrix B(A);

    EXPECT_TRUE(A == B);
}

TEST(Class_Matrix, Sparse_matrix_transposition) {
    int non = 6;
    int size = 4;

    std::vector<Complex> Ent;
    Complex Tmp;
    Ent = Tmp.InitVec({ 9, 3, 8, 15, 7, 16 });
    std::vector<int> irows = { 4, 1, 2, 4, 1, 4 };
    std::vector<int> shtcol = { 1, 2, 3, 5, 7 };

    Matrix A(size, non, Ent, irows, shtcol);

    /*
    * Matrix A is
    *
    * 0  3   0   7
    * 0  0   8   0
    * 0  0   0   0
    * 9  0  15  16
    *
    */
    A = A.T();
    /*
    * Matrix A.T should be
    *
    * 0  0  0   9
    * 3  0  0   0
    * 0  8  0  15
    * 7  0  0  16
    *
    */

    std::vector<Complex> EntRes;
    EntRes = Tmp.InitVec({3, 7, 8, 9, 15, 16});
    std::vector<int> irows_res = { 2, 4, 3, 1, 3, 4 };
    std::vector<int> shtcol_res = { 1, 3, 4, 4, 7 };
    Matrix Res(size, non, EntRes, irows_res, shtcol_res);

    EXPECT_TRUE(A == Res);
}

TEST(Class_Matrix, Sparse_matrix_multiplication_onlyreal) {
    int non = 6;
    int size = 4;
    Complex Tmp;

    std::vector<Complex> Ent;
    Ent = Tmp.InitVec({ 9, 3, 8, 15, 7, 16 });
    std::vector<int> irows = { 4, 1, 2, 4, 1, 4 };
    std::vector<int> shtcol = { 1, 2, 3, 5, 7 };

    Matrix A(size, non, Ent, irows, shtcol);

    /*
    * Matrix A is
    *
    * 0  3   0   7
    * 0  0   8   0
    * 0  0   0   0
    * 9  0  15  16
    *
    */

    Ent = Tmp.InitVec({ 2, 3, 1, 8, 4, 5 });
    irows = { 1, 3, 1, 2, 3, 4 };
    shtcol = { 1, 3, 4, 6, 7 };

    Matrix B(size, non, Ent, irows, shtcol);

    /*
    * Matrix B is
    *
    * 2  1  0  0
    * 0  0  8  0
    * 3  0  4  0
    * 0  0  0  5
    *
    */

    Matrix C = A * B;

    /*
    * Matrix C should be
    *
    * 0   0  24  35
    * 24  0  32   0
    * 0   0   0   0
    * 63  9  60  80
    *
    */

    std::vector<Complex> EntRes;
    EntRes = Tmp.InitVec({ 24, 63, 9, 24, 32, 60, 35, 80 });
    std::vector<int> irows_res = { 2, 4, 4, 1, 2, 4, 1, 4 };
    std::vector<int> shtcol_res = { 1, 3, 4, 7, 9 };
    Matrix Res(size, EntRes.size(), EntRes, irows_res, shtcol_res);

    EXPECT_TRUE(C == Res);
}

TEST(Class_Matrix, Sparse_matrix_multiplication_complex) {
    int non = 5;
    int size = 4;
    Complex Tmp;

    std::vector<Complex> Ent;
    Ent = Tmp.InitVec({ 4, 8, 18, 15, 2 }, { 20, 4, 4, 11, 7 });
    std::vector<int> irows = { 1, 4, 2, 3, 1 };
    std::vector<int> shtcol = { 1, 2, 3, 5, 6 };

    Matrix A(size, non, Ent, irows, shtcol);

    /*
    * Matrix A is
    *
    * 4+20i  0   0    2+7i
    * 0      0   18+4i   0
    * 0      0   15+11i  0
    * 0   8+4i   0       0
    *
    */

    Ent = Tmp.InitVec({ 4, 1, 12, 20, 9 }, { 14, 3, 2, 13, 10 });
    irows = { 2, 2, 4, 1, 1 };
    shtcol = { 1, 2, 4, 5, 6 };

    Matrix B(size, non, Ent, irows, shtcol);

    /*
    * Matrix B is
    *
    * 0      0      20+13i  9+10i
    * 4+14i  1+3i   0           0
    * 0      0      0           0
    * 0      12+2i  0           0
    *
    */

    Matrix C = A * B;

    /*
    * Matrix C should be
    *
    * 0         10+88i  -180+452i  -164+220i
    * 0         0       0                  0
    * 0         0       0                  0
    * -24+128i  -4+28i  0                  0
    *
    */

    std::vector<Complex> EntRes;
    EntRes = Tmp.InitVec({ -24, 10, -4, -180, -164 },
                         { 128, 88, 28, 452, 220 });
    std::vector<int> irows_res = { 4, 1, 4, 1, 1 };
    std::vector<int> shtcol_res = { 1, 2, 4, 5, 6 };
    Matrix Res(size, EntRes.size(), EntRes, irows_res, shtcol_res);

    EXPECT_TRUE(C == Res);
}

TEST(Class_Matrix, Sparse_matrix_random_matrix_creation) {
    Matrix A;
    int size = 10;
    int dist = 100;
    int cnt = 1;
    int seed = 0;

    EXPECT_NO_THROW(A.RandomMatrix(size, dist, cnt, seed));
}

TEST(Class_Matrix, Sparse_matrix_multiplication_complex_small) {
    int size = 10;
    int dist = 1000;
    int cnt = 1;
    Matrix A;
    A.RandomMatrix(size, dist, cnt, 4);
    Matrix B;
    B.RandomMatrix(size, dist, cnt, 1);

    Matrix C_seq = A ^ B;
    Matrix C_par = A * B;
    EXPECT_TRUE(C_seq == C_par);
}

// TEST(Class_Matrix, Sparse_matrix_mult_complex_meduim_time_perfomance) {
//     clock_t start, end;
//     double seq_time, tbb_time;
//     int size = 500;
//     int dist = 1000;
//     int cnt = 5;
//     Matrix A;
//     A.RandomMatrix(size, dist, cnt, 0);
//     Matrix B;
//     B.RandomMatrix(size, dist, cnt, 1);
//     start = clock();
//     Matrix C_seq = A ^ B;
//     end = clock();
//     std::cout << "SEQ Spended time -> " << (end - start + .0) / CLOCKS_PER_SEC
//               << " <- seconds" << std::endl;
//     seq_time = (end - start + .0) / CLOCKS_PER_SEC;

//     start = clock();
//     Matrix C_tbb = A * B;
//     end = clock();
//     std::cout << "TBB Spended time -> " << (end - start + .0) / CLOCKS_PER_SEC
//               << " <- seconds" << std::endl;
//     tbb_time = (end - start + .0) / CLOCKS_PER_SEC;
//     std::cout << "Performance improvement by -| " << seq_time / tbb_time
//               << " |- times" << std::endl;
//     EXPECT_TRUE(C_seq == C_tbb);
// }

// TEST(Class_Matrix, Sparse_matrix_mult_complex_large_time_perfomance) {
//     clock_t start, end;
//     double seq_time, tbb_time;
//     int size = 1000;
//     int dist = 1000;
//     int cnt = 10;
//     Matrix A;
//     A.RandomMatrix(size, dist, cnt, 0);
//     Matrix B;
//     B.RandomMatrix(size, dist, cnt, 1);
//     start = clock();
//     Matrix C_seq = A ^ B;
//     end = clock();
//     std::cout << "SEQ Spended time -> " << (end - start + .0) / CLOCKS_PER_SEC
//               << " <- seconds" << std::endl;
//     seq_time = (end - start + .0) / CLOCKS_PER_SEC;

//     start = clock();
//     Matrix C_tbb = A * B;
//     end = clock();
//     std::cout << "TBB Spended time -> " << (end - start + .0) / CLOCKS_PER_SEC
//               << " <- seconds" << std::endl;
//     tbb_time = (end - start + .0) / CLOCKS_PER_SEC;
//     std::cout << "Performance improvement by -| " << seq_time / tbb_time
//               << " |- times" << std::endl;
//     EXPECT_TRUE(C_seq == C_tbb);
// }

// TEST(Class_Matrix, Sparse_matrix_mult_complex_ext_large_time_perfomance) {
//     clock_t start, end;
//     double seq_time, tbb_time;
//     int size = 2500;
//     int dist = 100000;
//     int cnt = 25;
//     Matrix A;
//     A.RandomMatrix(size, dist, cnt, 0);
//     Matrix B;
//     B.RandomMatrix(size, dist, cnt, 1);
//     start = clock();
//     Matrix C_seq = A ^ B;
//     end = clock();
//     std::cout << "SEQ Spended time -> " << (end - start + .0) / CLOCKS_PER_SEC
//               << " <- seconds" << std::endl;
//     seq_time = (end - start + .0) / CLOCKS_PER_SEC;

//     start = clock();
//     Matrix C_tbb = A * B;
//     end = clock();
//     std::cout << "TBB Spended time -> " << (end - start + .0) / CLOCKS_PER_SEC
//               << " <- seconds" << std::endl;
//     tbb_time = (end - start + .0) / CLOCKS_PER_SEC;
//     std::cout << "Performance improvement by -| " << seq_time / tbb_time
//               << " |- times" << std::endl;
//     EXPECT_TRUE(C_seq == C_tbb);
// }
