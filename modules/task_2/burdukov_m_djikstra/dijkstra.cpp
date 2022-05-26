// Copyright 2022 Burdukov Mikhail
#include "../../../modules/task_2/burdukov_m_djikstra/dijkstra.h"

#include <iostream>
#include <vector>

std::vector<int> dijkstra(const int start,
                          const std::vector<std::vector<int>>& matrix,
                          const int threads) {
  int size = static_cast<int>(matrix.size());
  const int INF = 1000000;
  std::vector<int> dist(size, INF);

  dist[start] = 0;
  int curr_v = start;
  std::vector<int> updated(size, 0);
  int timer = 1;
  while (true) {
    int min = INF, pos = -1;
    #pragma omp parallel num_threads(threads)
    {
      #pragma omp for
      for (int i = 0; i < size; i++) {
        if (i == curr_v) continue;
        if (dist[curr_v] + matrix[curr_v][i] < dist[i]) {
          dist[i] = dist[curr_v] + matrix[curr_v][i];
          updated[i] = timer;
        }
      }
      int local_min = INF;
      int local_pos = -1;
      #pragma omp for
      for (int i = 0; i < size; i++) {
        if (updated[i] == timer && dist[i] < local_min) {
          local_min = dist[i];
          local_pos = i;
        }
      }
      #pragma omp critical
      {
        if (local_min < min) {
          pos = local_pos;
          min = local_min;
        }
      }
    }
    if (pos == -1) break;
    curr_v = pos;
    timer++;
  }
  return dist;
}
