// Copyright 2022 Miheev Ivan

#include <gtest/gtest.h>
#include <omp.h>
#include <time.h>

#include <vector>

#include "../../../modules/task_2/miheev_i_mult_matrix_ccs_double_omp/matrix_ccs_omp.h"

TEST(SprMatCCS_Test, Sparse_matrix_multiplication_int) {
  int cap = 3;
  int dim = 3;

  std::vector<double> val{100, -15, 2};
  std::vector<int> rows{2, 1, 3};
  std::vector<int> ptr{1, 2, 3, 4};

  SprMatCCS A(dim, cap, val, rows, ptr);

  val = {3, 6, -23};
  rows = {3, 2, 1};
  ptr = {1, 2, 3, 4};

  SprMatCCS B(dim, cap, val, rows, ptr);

  SprMatCCS C = A.ParallelMult(B);

  std::vector<double> res_val{6, -90, -2300};
  std::vector<int> rows_res{3, 1, 2};
  std::vector<int> ptr_res{1, 2, 3, 4};
  SprMatCCS Res(dim, res_val.size(), res_val, rows_res, ptr_res);

  EXPECT_TRUE(C == Res);
}

TEST(SprMatCCS_Test, Sparse_matrix_multiplication_double1) {
  int cap = 4;
  int dim = 4;

  std::vector<double> val{-10.5, 22.1, 2.4, -3.2};
  std::vector<int> rows{1, 3, 1, 3};
  std::vector<int> ptr{1, 2, 3, 4, 5};

  SprMatCCS A(dim, cap, val, rows, ptr);

  val = {2.4, 6, -39, 108};
  rows = {1, 2, 3, 1};
  ptr = {1, 2, 4, 4, 5};

  SprMatCCS B(dim, cap, val, rows, ptr);

  SprMatCCS C = A.ParallelMult(B);

  std::vector<double> res_val{-25.2, -93.6, 132.6, -1134};
  std::vector<int> rows_res{1, 1, 3, 1};
  std::vector<int> ptr_res{1, 2, 4, 4, 5};
  SprMatCCS Res(dim, res_val.size(), res_val, rows_res, ptr_res);

  EXPECT_TRUE(C == Res);
}

TEST(SprMatCCS_Test, Sparse_matrix_multiplication_double2) {
  int cap = 5;
  int dim = 3;

  std::vector<double> val{-12, 3, 21.4, 8, -4.5};
  std::vector<int> rows{1, 3, 2, 3, 1};
  std::vector<int> ptr{1, 3, 5, 6};

  SprMatCCS A(dim, cap, val, rows, ptr);

  val = {2.4, 6, -39, 108, 44};
  rows = {1, 2, 3, 1, 3};
  ptr = {1, 2, 4, 6};

  SprMatCCS B(dim, cap, val, rows, ptr);

  SprMatCCS C = A.ParallelMult(B);

  std::vector<double> res_val{-28.8, 7.2, 175.5, 128.4, 48, -1494, 324};
  std::vector<int> rows_res{1, 3, 1, 2, 3, 1, 3};
  std::vector<int> ptr_res{1, 3, 6, 8};
  SprMatCCS Res(dim, res_val.size(), res_val, rows_res, ptr_res);

  EXPECT_TRUE(C == Res);
}

TEST(SprMatCCS_Test, Sparse_matrix_multiplication_perf100) {
  double start, end;
  double seq_time, omp_time;
  int dim = 100;
  int spr = 1;

  SprMatCCS A;
  A.randMat(dim, spr);
  SprMatCCS B;
  B.randMat(dim, spr);

  start = omp_get_wtime();
  SprMatCCS C_seq = A * B;
  end = omp_get_wtime();
  std::cout << "SEQ time: " << end - start << " sec" << std::endl;
  seq_time = end - start;

  start = omp_get_wtime();
  SprMatCCS C_omp = A.ParallelMult(B);
  end = omp_get_wtime();
  std::cout << "OMP time: " << end - start << " sec" << std::endl;
  omp_time = end - start;

  std::cout << "Performance improvement is " << seq_time / omp_time
            << std::endl;

  EXPECT_TRUE(C_seq == C_omp);
}

TEST(SprMatCCS_Test, Sparse_matrix_multiplication_perf500) {
  double start, end;
  double seq_time, omp_time;
  int dim = 500;
  int spr = 1;

  SprMatCCS A;
  A.randMat(dim, spr);
  SprMatCCS B;
  B.randMat(dim, spr);

  start = omp_get_wtime();
  SprMatCCS C_seq = A * B;
  end = omp_get_wtime();
  std::cout << "SEQ time: " << end - start << " sec" << std::endl;
  seq_time = end - start;

  start = omp_get_wtime();
  SprMatCCS C_omp = A.ParallelMult(B);
  end = omp_get_wtime();
  std::cout << "OMP time: " << end - start << " sec" << std::endl;
  omp_time = end - start;

  std::cout << "Performance improvement is " << seq_time / omp_time
            << std::endl;

  EXPECT_TRUE(C_seq == C_omp);
}

TEST(SprMatCCS_Test, Sparse_matrix_multiplication_perf1000) {
  double start, end;
  double seq_time, omp_time;
  int dim = 1000;
  int spr = 1;

  SprMatCCS A;
  A.randMat(dim, spr);
  SprMatCCS B;
  B.randMat(dim, spr);

  start = omp_get_wtime();
  SprMatCCS C_seq = A * B;
  end = omp_get_wtime();
  std::cout << "SEQ time: " << end - start << " sec" << std::endl;
  seq_time = end - start;

  start = omp_get_wtime();
  SprMatCCS C_omp = A.ParallelMult(B);
  end = omp_get_wtime();
  std::cout << "OMP time: " << end - start << " sec" << std::endl;
  omp_time = end - start;

  std::cout << "Performance improvement is " << seq_time / omp_time
            << std::endl;

  EXPECT_TRUE(C_seq == C_omp);
}

TEST(SprMatCCS_Test, Sparse_matrix_multiplication_perf2000) {
  double start, end;
  double seq_time, omp_time;
  int dim = 2000;
  int spr = 1;

  SprMatCCS A;
  A.randMat(dim, spr);
  SprMatCCS B;
  B.randMat(dim, spr);

  start = omp_get_wtime();
  SprMatCCS C_seq = A * B;
  end = omp_get_wtime();
  std::cout << "SEQ time: " << end - start << " sec" << std::endl;
  seq_time = end - start;

  start = omp_get_wtime();
  SprMatCCS C_omp = A.ParallelMult(B);
  end = omp_get_wtime();
  std::cout << "OMP time: " << end - start << " sec" << std::endl;
  omp_time = end - start;

  std::cout << "Performance improvement is " << seq_time / omp_time
            << std::endl;

  EXPECT_TRUE(C_seq == C_omp);
}

TEST(SprMatCCS_Test, Sparse_matrix_multiplication_perf5000) {
  double start, end;
  double seq_time, omp_time;
  int dim = 5000;
  int spr = 1;

  SprMatCCS A;
  A.randMat(dim, spr);
  SprMatCCS B;
  B.randMat(dim, spr);

  start = omp_get_wtime();
  SprMatCCS C_seq = A * B;
  end = omp_get_wtime();
  std::cout << "SEQ time: " << end - start << " sec" << std::endl;
  seq_time = end - start;

  start = omp_get_wtime();
  SprMatCCS C_omp = A.ParallelMult(B);
  end = omp_get_wtime();
  std::cout << "OMP time: " << end - start << " sec" << std::endl;
  omp_time = end - start;

  std::cout << "Performance improvement is " << seq_time / omp_time
            << std::endl;

  EXPECT_TRUE(C_seq == C_omp);
}
