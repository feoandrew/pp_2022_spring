// Copyright 2022 Shurygina A
#include <gtest/gtest.h>
#include <vector>
#include "../../../modules/task_3/shurygina_a_labeling_tbb/labeling.h"

TEST(Labeling_tbb, 5x5) {
    std::vector<std::vector<int>> img(5, std::vector<int>(5));
    img = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 1, 0},
        {0, 1, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {0, 0, 0, 1, 0}
    };
    std::vector<std::vector<int>> expres(5, std::vector<int>(5));
    expres = {
        {0, 0, 0, 0, 0 },
        {0, 0, 0, 1, 0 },
        {0, 2, 0, 0, 0 },
        {0, 2, 0, 0, 0 },
        {0, 0, 0, 3, 0 }
    };
    std::vector<std::vector<int>>  res = labelingTbb(img);
    ASSERT_EQ(expres, res);
}

TEST(Labeling_tbb, 5x6) {
    std::vector<std::vector<int>> img(5, std::vector<int>(5));
    img = {
        {0, 0, 0, 0, 0, 0},
        {0, 1, 0, 1, 0, 0},
        {0, 1, 0, 1, 0, 0},
        {0, 1, 1, 1, 1, 0},
        {0, 0, 0, 0, 0, 0}
    };
    std::vector<std::vector<int>> expres(5, std::vector<int>(5));
    expres = {
        {0, 0, 0, 0, 0, 0},
        {0, 1, 0, 1, 0, 0},
        {0, 1, 0, 1, 0, 0},
        {0, 1, 1, 1, 1, 0},
        {0, 0, 0, 0, 0, 0}
    };
    std::vector<std::vector<int>>  res = labelingTbb(img);
    ASSERT_EQ(expres, res);
}

TEST(Labeling, 2x2_no_change) {
    std::vector<std::vector<int>> img(2, std::vector<int>(2));
    img = {
        {0, 0},
        {0, 0}
    };
    std::vector<std::vector<int>> expres(2, std::vector<int>(2));
    expres = {
        {0, 0},
        {0, 0}
    };
    std::vector<std::vector<int>>  res = labelingTbb(img);
    ASSERT_EQ(expres, res);
}


TEST(Labeling, 2x2) {
    std::vector<std::vector<int>> img(2, std::vector<int>(2));
    img = {
        {0, 0},
        {0, 1}
    };
    std::vector<std::vector<int>> expres(2, std::vector<int>(2));
    expres = {
        {0, 0},
        {0, 1}
    };
    std::vector<std::vector<int>>  res = labelingTbb(img);
    ASSERT_EQ(expres, res);
}

TEST(Labeling_tbb, No_Throw) {
  std::vector<std::vector<int>> A(4, std::vector<int>(4));
  A = {{1, 1, 1, 0},
       {0, 1, 0, 1},
       {0, 1, 0, 1},
       {1, 1, 0, 1}};
  ASSERT_NO_THROW(labelingTbb(A));
}
