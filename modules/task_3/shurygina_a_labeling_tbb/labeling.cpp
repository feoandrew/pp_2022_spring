// Copyright 2022 Shurygina A

#include "../../../modules/task_3/shurygina_a_labeling_tbb/labeling.h"
#include <tbb/tbb.h>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>

void getRandomImg(std::vector<std::vector<int>>* ptr) {
    int h = ptr->size();
    int w = (*ptr)[0].size();
  std::mt19937 gen(time(0));
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j)
      if (gen() % 2 == 0)
        (*ptr)[i][j] = 1;
      else
        (*ptr)[i][j] = 0;
  }
}

std::vector<std::vector<int>> labelingTbb(
    const std::vector<std::vector<int>>& img) {
    std::vector<std::vector<int>> res(
      img.size(), std::vector<int>(img[0].size(), 0));
  std::vector<int> sec_start;
  int img_size = img.size();
  int data_size = img[0].size();
  int comp_counter = 1;
  tbb::task_scheduler_init init;
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, img_size, 5),
      [&res, img, &comp_counter](tbb::blocked_range<size_t> r) {
          int start = r.begin();
          int end = r.end();
          int first_row = -1;
        bool lock = true;
        int w = res[0].size();
        for (int i = start; i != end; ++i) {
          for (int j = 0; j < w; ++j) {
              int up_value = 0;
              int left_value = 0;
              int sel_v = 0;

            if (lock == false) up_value = img[i - 1][j];
            if (j > 0) left_value = img[i][j - 1];
            sel_v = img[i][j];

            if (sel_v == 0) {
              continue;
            } else if (up_value == 1 && left_value == 1) {
              if (res[i - 1][j] == res[i][j - 1]) {
                res[i][j] = res[i - 1][j];
              } else {
               int min =
                    (res[i - 1][j] < res[i][j - 1] ? res[i - 1][j]
                                                   : res[i][j - 1]);
               int max =
                    (res[i - 1][j] > res[i][j - 1] ? res[i - 1][j]
                                                   : res[i][j - 1]);
                merge(&res, first_row, max, min, i, j);
                res[i][j] = min;
              }
            } else if (up_value == 1 && left_value == 0) {
              res[i][j] = res[i - 1][j];
            } else if (up_value == 0 && left_value == 1) {
              res[i][j] = res[i][j - 1];
            } else if (up_value == 0 && left_value == 0) {
              {
                res[i][j] = comp_counter;
                comp_counter++;
              }
            }
          }

          if (lock == true) {
            lock = false;
            first_row = i;
          }
        }
      });
  init.terminate();
  for (int i = 0; i < img_size; ++i) {
    for (int j = 0; j < data_size; ++j) {
        int sel_v = res[i][j];
        int up_value = 0;

      if (i > 0) up_value = res[i - 1][j];

      if (sel_v == 0 || up_value == 0) {
        continue;
      } else if (sel_v != up_value) {
        merge(&res, 0, up_value, sel_v, i, data_size);
      }
    }
  }
  return res;
}

void merge(std::vector<std::vector<int>>* ptr,
    int str_row, int pr_num, int req_num,
    int end_x, int end_y) {
    int h = (*ptr).size();
    int w = (*ptr)[0].size();
  for (int i = str_row; i < h; ++i)
    for (int j = 0; j < w; ++j) {
      if ((*ptr)[i][j] == pr_num) (*ptr)[i][j] = req_num;
      if (i == end_x && j == end_y) return;
    }
}
