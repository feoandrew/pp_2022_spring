
//  Copyright 2022 Poplavsky Anton
#include <gtest/gtest.h>

#include "../../../modules/task_3/poplavsky_a_bin_convex_hull/conv.h"

TEST(ConvexHull, timer) {
  double time1, time2, time3, time4;
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> urd(-10000.f, 10000.f);
  std::vector<Coordinates> points;
  Coordinates pt;
  for (int i = 0; i < SIZE; i++) {
    pt.set_x(urd(gen));
    pt.set_y(urd(gen));
    points.push_back(pt);
  }
  time1 = omp_get_wtime();
  std::vector<Coordinates> hull_seq = graham_seq(points);
  time2 = omp_get_wtime();
  time3 = omp_get_wtime();
  std::vector<Coordinates> hull_tbb = graham_tbb(points);
  time4 = omp_get_wtime();

  printf("seq time : %lf\n\n", (time2 - time1));
  printf("tbb time : %lf\n\n", (time4 - time3));
  printf("hull time : %lf\n\n", (time2 - time1) / (time4 - time3));
}

TEST(ConvexHull, ttt) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> urd(-1000, 1000);
  std::vector<Coordinates> points;
  Coordinates pt;
  for (int i = 0; i < 3; i++) {
    pt.set_x(urd(gen));
    pt.set_y(urd(gen));
    points.push_back(pt);
  }
  std::vector<Coordinates> hull = graham_seq(points);
}

TEST(ConvexHull, iii) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> urd(-1000, 1000);
  std::vector<Coordinates> points;
  Coordinates pt;
  for (int i = 0; i < 5; i++) {
    pt.set_x(urd(gen));
    pt.set_y(urd(gen));
    points.push_back(pt);
  }
  std::vector<Coordinates> hull = graham_seq(points);
}

TEST(ConvexHull, mmm) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> urd(-1000, 1000);
  std::vector<Coordinates> points;
  Coordinates pt;
  for (int i = 0; i < 6; i++) {
    pt.set_x(urd(gen));
    pt.set_y(urd(gen));
    points.push_back(pt);
  }
  std::vector<Coordinates> hull = graham_seq(points);
}

TEST(ConvexHull, eee) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> urd(-1000, 1000);
  std::vector<Coordinates> points;
  Coordinates pt;
  for (int i = 0; i < 9; i++) {
    pt.set_x(urd(gen));
    pt.set_y(urd(gen));
    points.push_back(pt);
  }
  std::vector<Coordinates> hull = graham_seq(points);
}
