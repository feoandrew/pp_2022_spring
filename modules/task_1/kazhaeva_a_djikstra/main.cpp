// Copyright 2022 Kazhaeva Anastasia
#include <gtest/gtest.h>

#include <vector>

#include "./djikstra_seq.h"

TEST(Sequential_Deicstra, Test_ASSERT_NO_THROW) {
    std::vector<std::vector<int>> b = {
                  {0, 1, 7, 6, 3},
                  {1, 0, 3, 5, 4},
                  {7, 3, 0, 10, 5},
                  {6, 5, 10, 0, 4},
                  {3, 4, 5, 4, 0}
    };

  ASSERT_NO_THROW(djikstra(0, b));
}

TEST(Sequential_Deicstra, Test_Deicstra_1) {
    std::vector<std::vector<int>> b = {
                    {0, 1, 7, 6, 3},
                    {1, 0, 3, 5, 4},
                    {7, 3, 0, 10, 5},
                    {6, 5, 10, 0, 4},
                    {3, 4, 5, 4, 0}
    };
    std::vector<int> answer = { 0, 1, 4, 6, 3 };
    const std::vector<int> result = djikstra(0, b);

    ASSERT_EQ(answer == result, true);
}

TEST(Sequential_Deicstra, Test_Deicstra_2) {
    std::vector<std::vector<int>> b = {
                    {0, 1, 7, 6, 3},
                    {1, 0, 3, 5, 4},
                    {7, 3, 0, 10, 5},
                    {6, 5, 10, 0, 4},
                    {3, 4, 5, 4, 0}
    };
    std::vector<int> answer = { 6, 5, 8, 0, 4 };
    const std::vector<int> result = djikstra(3, b);
    ASSERT_EQ(answer == result, true);
}

TEST(Sequential_Deicstra, Test_Deicstra_3) {
    std::vector<std::vector<int>> b = {
                    {0, 7, 9, 0, 0, 14},
                    {7, 0, 10, 15, 0, 0},
                    {9, 10, 0, 11, 0, 2},
                    {0, 15, 11, 0, 6, 0},
                    {0, 0, 0, 6, 0, 9},
                    {14, 0, 2, 0, 9, 0}
    };
    std::vector<int> answer = { 0, 7, 9, 20, 20, 11 };
    const std::vector<int> result = djikstra(0, b);
    ASSERT_EQ(answer == result, true);
}

TEST(Sequential_Deicstra, Test_Deicstra_4) {
    std::vector<std::vector<int>> b = {
                    {0, 1, 7, 6, 3},
                    {1, 0, 3, 5, 4},
                    {7, 3, 0, 10, 5},
                    {6, 5, 10, 0, 4},
                    {3, 4, 5, 4, 0}
    };
    std::vector<int> answer = { 4, 3, 0, 8, 5 };
    const std::vector<int> result = djikstra(2, b);
    ASSERT_EQ(answer == result, true);
}
