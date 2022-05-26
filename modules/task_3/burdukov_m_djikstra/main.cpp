// Copyright 2022 Burdukov Mikhail
#include <gtest/gtest.h>
#include <tbb/tbb.h>

#include <vector>

#include "./dijkstra.h"

TEST(burkukov_dijkstra_tbb, create) {
  std::vector<std::vector<int>> a(1, std::vector<int>(1, 0));
  const int num_threads = 4;
  std::vector<int> d = dijkstra(0, a, num_threads);
  ASSERT_EQ(d[0], 0);
}

TEST(burkukov_dijkstra_tbb, simple_path_1v) {
  std::vector<std::vector<int>> a(2, std::vector<int>(2, 0));
  a[1][0] = 1;
  a[0][1] = 1;
  const int num_threads = 4;
  std::vector<int> d = dijkstra(0, a, num_threads);
  ASSERT_EQ(d[1], 1);
}

TEST(burkukov_dijkstra_tbb, simple_path_5000v) {
  int size = 5000;
  std::vector<std::vector<int>> a(size, std::vector<int>(size, 0));
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++) a[i][j] = 10000;
  for (int i = 0; i < size; i++) {
    if (i != size - 1) a[i][i + 1] = 1;
    a[i][i] = 0;
  }
  const int num_threads = 4;
  std::vector<int> d = dijkstra(0, a, num_threads);
  ASSERT_EQ(d[size - 1], size - 1);
}

TEST(burkukov_dijkstra_tbb, tree) {
  int size = 5000;
  std::vector<std::vector<int>> a(size, std::vector<int>(size, 0));
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++) a[i][j] = ((i == j) ? 0 : 100);

  for (int i = 1; i < size; i++) {
    a[0][i] = 1;
  }
  const int num_threads = 4;
  std::vector<int> d = dijkstra(0, a, num_threads);
  ASSERT_EQ(d[2], 1);
}

TEST(burkukov_dijkstra_tbb, tree_simp) {
  std::vector<std::vector<int>> a = {
      {0, 100, 2},
      {3, 0, 100},
      {100, 4, 0},
  };
  const int num_threads = 4;
  std::vector<int> d = dijkstra(0, a, num_threads);
  ASSERT_EQ(d[1], 6);
}

int main(int argc, char **argv) {
  tbb::task_scheduler_init init(4);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
