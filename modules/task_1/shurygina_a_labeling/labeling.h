// Copyright 2022 Shurygina A

#ifndef MODULES_TASK_1_SHURYGINA_A_LABELING_LABELING_H_
#define MODULES_TASK_1_SHURYGINA_A_LABELING_LABELING_H_

#include <vector>
#include <utility>

std::vector<int> getRandomImg(int rows, int cols);
std::pair<std::vector<int>, std::vector<int> >
         firstPass(std::vector<int>, int, int);
std::vector<int>
         secondPass(std::vector<int>, int, int, std::vector<int>);
std::vector<int> labeling(const std::vector<int>&, int, int);

#endif  //  MODULES_TASK_1_SHURYGINA_A_LABELING_LABELING_H_
