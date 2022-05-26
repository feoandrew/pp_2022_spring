// Copyright 2022 Tyrina Anastasia
#include <gtest/gtest.h>

#include "./dijkstra.h"

TEST(DIJKSTRA_OMP, TEST_1) {
  int V = 9;
  Graph graph = {{0, 4, 0, 0, 0, 0, 0, 8, 0},  {4, 0, 8, 0, 0, 0, 0, 11, 0},
                 {0, 8, 0, 7, 0, 4, 0, 0, 2},  {0, 0, 7, 0, 9, 14, 0, 0, 0},
                 {0, 0, 0, 9, 0, 10, 0, 0, 0}, {0, 0, 4, 14, 10, 0, 2, 0, 0},
                 {0, 0, 0, 0, 0, 2, 0, 1, 6},  {8, 11, 0, 0, 0, 0, 1, 0, 7},
                 {0, 0, 2, 0, 0, 0, 6, 7, 0}};

  Graph result_sequential = sequentialDijkstra(graph, V);
  Graph result_parallel = parallelDijkstra(graph, V);

  for (int i = 0; i < V; i++) {
    ASSERT_EQ(result_sequential[i], result_parallel[i]);
  }
}

TEST(DIJKSTRA_OMP, TEST_2) {
  int V = 10;
  Graph graph = getRandomGraph(V);
  Graph result_sequential = sequentialDijkstra(graph, V);
  Graph result_parallel = parallelDijkstra(graph, V);

  for (int i = 0; i < V; i++) {
    ASSERT_EQ(result_sequential[i], result_parallel[i]);
  }
}

TEST(DIJKSTRA_OMP, TEST_3) {
  int V = 20;
  Graph graph = getRandomGraph(V);
  Graph result_sequential = sequentialDijkstra(graph, V);
  Graph result_parallel = parallelDijkstra(graph, V);

  for (int i = 0; i < V; i++) {
    ASSERT_EQ(result_sequential[i], result_parallel[i]);
  }
}

TEST(DIJKSTRA_OMP, TEST_4) {
  int V = 30;
  Graph graph = getRandomGraph(V);
  Graph result_sequential = sequentialDijkstra(graph, V);
  Graph result_parallel = parallelDijkstra(graph, V);

  for (int i = 0; i < V; i++) {
    ASSERT_EQ(result_sequential[i], result_parallel[i]);
  }
}

int V = 50;

TEST(DIJKSTRA_OMP, TEST_BIG_OMP) {
  Graph graph = getRandomGraph(V);
  Graph result_parallel = parallelDijkstra(graph, V);
}

TEST(DIJKSTRA_OMP, TEST_BIG_SEQ) {
  Graph graph = getRandomGraph(V);
  Graph result_sequential = sequentialDijkstra(graph, V);
}
