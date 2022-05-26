// Copyright 2022 Burdukov Mikhail
#ifndef MODULES_TASK_3_BURDUKOV_M_DJIKSTRA_DIJKSTRA_H_
#define MODULES_TASK_3_BURDUKOV_M_DJIKSTRA_DIJKSTRA_H_
#include <vector>

std::vector<int> dijkstra(const int start,
                          const std::vector<std::vector<int>>& matrix,
                          const int num_threads);
#endif  // MODULES_TASK_3_BURDUKOV_M_DJIKSTRA_DIJKSTRA_H_
