// Copyright 2022 Kharunova Alina
#include <gtest/gtest.h>

#include "./kharunova_a_dijkstra_algorithm_omp.h"

TEST(search_path_omp, 1_size_graph_1) {
  ASSERT_NO_THROW(start_parallel_algorithm(0, 1, 1));
}

TEST(search_path_omp, 15_size_graph_10) {
  ASSERT_NO_THROW(start_parallel_algorithm(0, 15, 10));
}

TEST(search_path_omp, 50_size_graph_30) {
  ASSERT_NO_THROW(start_parallel_algorithm(0, 50, 30));
}

TEST(search_path_omp, 100_size_graph_50) {
  ASSERT_NO_THROW(start_parallel_algorithm(0, 100, 50));
}

TEST(search_path_omp, 500_size_graph_80) {
  ASSERT_NO_THROW(start_parallel_algorithm(0, 500, 80));
}

TEST(search_path_omp, 1000_size_graph_100) {
  ASSERT_NO_THROW(start_parallel_algorithm(0, 1000, 100));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
