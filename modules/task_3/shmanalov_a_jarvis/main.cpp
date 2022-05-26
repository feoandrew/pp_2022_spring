// Copyright 2022 Shmanalov Alexander
#include <gtest/gtest.h>
#include <iostream>
#include "../../modules/task_3/shmanalov_a_jarvis/jarvis_tbb.h"

TEST(Tbb_Jarvis, PointsCount_0) {
    std::vector<Point> points(0);
    ASSERT_ANY_THROW(tbbJarvis(points));
}

TEST(Tbb_Jarvis, PointsCount_2500) {
    std::vector<Point> points = getRandomCombinationPoints(2500);
    std::vector<Point> resultSeq = sequentinalJarvis(points);
    std::vector<Point> resultTbb = tbbJarvis(points);
    ASSERT_EQ(resultSeq, resultTbb);
}

TEST(Tbb_Jarvis, PointsCount_5000) {
    std::vector<Point> points = getRandomCombinationPoints(5000);
    std::vector<Point> resultSeq = sequentinalJarvis(points);
    std::vector<Point> resultTbb = tbbJarvis(points);
    ASSERT_EQ(resultSeq, resultTbb);
}

TEST(Tbb_Jarvis, PointsCount_75000) {
    std::vector<Point> points = getRandomCombinationPoints(75000);
    std::vector<Point> resultSeq = sequentinalJarvis(points);
    std::vector<Point> resultTbb = tbbJarvis(points);
    ASSERT_EQ(resultSeq, resultTbb);
}

TEST(Tbb_Jarvis, PointsCount_1000000) {
    std::vector<Point> points = getRandomCombinationPoints(1000000);
    // tbb::tick_count startSeq = tbb::tick_count::now();
    std::vector<Point> resultSeq = sequentinalJarvis(points);
    // tbb::tick_count endSeq = tbb::tick_count::now();
    // double timeSeq = (endSeq - startSeq).seconds();
    // std::cout << "Sequential = " << timeSeq << std::endl;
    // tbb::tick_count startTbb = tbb::tick_count::now();
    std::vector<Point> resultTbb = tbbJarvis(points);
    // tbb::tick_count endTbb = tbb::tick_count::now();
    // double timeTbb = (endTbb - startTbb).seconds();
    // std::cout << "Parallel [TBB] = " << timeTbb << std::endl;
    // std::cout << "Efficiency = " << timeSeq / timeTbb << std::endl;
    ASSERT_EQ(resultSeq, resultTbb);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // tbb::task_scheduler_init init(tbb::task_scheduler_init::automatic);
    return RUN_ALL_TESTS();
}
