// Copyright 2022 Kharunova Alina
#ifndef MODULES_TASK_2_KHARUNOVA_A_DIJKSTRA_ALGORITHM_OMP_KHARUNOVA_A_DIJKSTRA_ALGORITHM_OMP_H_
#define MODULES_TASK_2_KHARUNOVA_A_DIJKSTRA_ALGORITHM_OMP_KHARUNOVA_A_DIJKSTRA_ALGORITHM_OMP_H_
#include <vector>
std::vector<std::vector<int>> create_random_graph(int num_of_vertex,
                                                  int max_destination);
void start_parallel_algorithm(int start_vertex, int num_of_vertex,
                              int max_destination);
#endif  // MODULES_TASK_2_KHARUNOVA_A_DIJKSTRA_ALGORITHM_OMP_KHARUNOVA_A_DIJKSTRA_ALGORITHM_OMP_H_
