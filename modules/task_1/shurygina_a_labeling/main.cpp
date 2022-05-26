// Copyright 2022 Shurygina A

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "./labeling.h"

TEST(Clabeling_Matrix_gen, invalid_size) {
    int rows = -5;
    int cols = 7;
    ASSERT_ANY_THROW(getRandomImg(rows, cols));
}

TEST(labeling, 2x2_simple) {
    int rows = 2;
    int cols = 2;
    const std::vector<int> img = {
        0, 0,
        0, 0,
    };
    std::vector<int> expres = {
        0, 0,
        0, 0
    };
    std::vector<int> res = labeling(img, rows, cols);
    ASSERT_EQ(expres, res);
}

TEST(labeling, 2x2) {
    int rows = 2;
    int cols = 2;
    const std::vector<int> img = {
        0, 0,
        0, 1,
    };
    std::vector<int> expres = {
        0, 0,
        0, 1
    };
    std::vector<int> res = labeling(img, rows, cols);
    ASSERT_EQ(expres, res);
}


TEST(labeling, 5x5) {
    int rows = 5;
    int cols = 5;
    const std::vector<int> img = {
        0, 0, 0, 0, 0,
        0, 0, 0, 1, 0,
        0, 1, 1, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 0, 0, 0
    };
    std::vector<int> expres = {
        0, 0, 0, 0, 0,
        0, 0, 0, 1, 0,
        0, 2, 2, 0, 0,
        0, 2, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    std::vector<int> res = labeling(img, rows, cols);
    ASSERT_EQ(expres, res);
}

TEST(labeling, 6x6) {
    int rows = 6;
    int cols = 6;
    const std::vector<int> img = {
        0, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 0,
        0, 1, 0, 1, 0, 0,
        0, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0
    };
    std::vector<int> expres = {
        0, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 0,
        0, 1, 0, 1, 0, 0,
        0, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0
    };
    std::vector<int> res = labeling(img, rows, cols);
    ASSERT_EQ(expres, res);
}

TEST(labeling, 6x12) {
    int rows = 6;
    int cols = 12;
    const std::vector<int> img = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0,
        0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
        0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    std::vector<int> expres = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 2, 2, 0, 0, 3, 0,
        0, 4, 4, 0, 1, 0, 2, 0, 2, 0, 3, 0,
        0, 4, 0, 0, 1, 0, 2, 0, 2, 0, 0, 0,
        0, 4, 4, 0, 1, 0, 2, 2, 2, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    std::vector<int> res = labeling(img, rows, cols);
    ASSERT_EQ(expres, res);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
