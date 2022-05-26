// Copyright 2022 Shurygina A
#ifndef MODULES_TASK_3_SHURYGINA_A_LABELING_TBB_LABELING_H_
#define MODULES_TASK_3_SHURYGINA_A_LABELING_TBB_LABELING_H_
#include <omp.h>
#include <string>
#include <vector>
#include <iostream>
#include <random>
#include <ctime>


std::vector<std::vector<int>> labelingTbb(
    const std::vector<std::vector<int>>& pic);
void merge(std::vector<std::vector<int>>* ptr, int str_row,
    int pr_num, int req_num, int end_x,
    int end_y);
void getRandomImg(std::vector<std::vector<int>>* ptr);


#endif  // MODULES_TASK_3_SHURYGINA_A_LABELING_TBB_LABELING_H_
