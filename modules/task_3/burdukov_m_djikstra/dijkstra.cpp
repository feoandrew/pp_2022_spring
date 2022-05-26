// Copyright 2022 Burdukov Mikhail
#include "../../../modules/task_3/burdukov_m_djikstra/dijkstra.h"

#include <tbb/tbb.h>

#include <algorithm>
#include <iostream>
#include <utility>
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
    tbb::enumerable_thread_specific<std::pair<int, int>> res{INF, -1};
    tbb::parallel_for(
        tbb::blocked_range<int>(0, size),
        [&](const tbb::blocked_range<int>& range) {
          for (int i = range.begin(); i < range.end(); i++) {
            if (dist[curr_v] + matrix[curr_v][i] < dist[i]) {
              dist[i] = dist[curr_v] + matrix[curr_v][i];
              updated[i] = timer;
            }
            if (updated[i] == timer && dist[i] < res.local().first) {
              res.local().first = dist[i];
              res.local().second = i;
            }
          }
        });
    std::pair<int, int> tmp =
        res.combine([&](std::pair<int, int> x, std::pair<int, int> y) {
          if (x.first < y.first)
            return x;
          else
            return y;
        });
    curr_v = tmp.second;
    timer++;
    if (curr_v == -1) break;
  }
  return dist;
}
