// Copyright 2022 Poplavsky Anton
#include "../../../modules/task_1/poplavsky_a_bin_convex_hull/conv.h"

int find_minimal_coordinates(
    const std::vector<Coordinates>& input_coordinates) {
  int number = 0;
  int minimal_x = input_coordinates[0].X;
  int minimal_y = input_coordinates[0].Y;
  int size = input_coordinates.size();
  for (int i = 1; i < size; i++) {
    if ((input_coordinates[i].X < minimal_x) ||
        (input_coordinates[i].X == minimal_x &&
         input_coordinates[i].Y < minimal_y)) {
      minimal_x = input_coordinates[i].X;
      minimal_y = input_coordinates[i].Y;
      number = i;
    }
  }
  return number;
}

std::vector<Coordinates> graham(
    const std::vector<Coordinates>& _input_coordinates) {
  std::vector<Coordinates> input_coordinates = _input_coordinates;
  int size = input_coordinates.size();
  if (size < 3) {
    printf("Convex hull doesn't exist!!\n");
    return std::vector<Coordinates>();
  }
  Coordinates minimal_coordinate;
  int minimal = find_minimal_coordinates(input_coordinates);
  Coordinates tmp = input_coordinates[0];
  input_coordinates[0] = input_coordinates[minimal];
  input_coordinates[minimal] = tmp;
  minimal_coordinate = input_coordinates[0];
  std::sort(input_coordinates.begin() + 1, input_coordinates.end(),
            [&](const Coordinates& p1, const Coordinates& p2) {
              if (minimal_coordinate.turn(p1, p2) == COLLINEAR) {
                return (minimal_coordinate.vector_norm(p1) <
                        minimal_coordinate.vector_norm(p2));
              }
              return (minimal_coordinate.turn(p1, p2) == ANTICLOCKWISE);
            });
  std::vector<Coordinates> without_collinear;
  without_collinear.push_back(input_coordinates[0]);
  for (int i = 1; i < size; i++) {
    while (i < size - 1 &&
           input_coordinates[i - 1].turn(
               input_coordinates[i], input_coordinates[i + 1]) == COLLINEAR) {
      i++;
    }
    without_collinear.push_back(input_coordinates[i]);
  }
  input_coordinates.clear();
  input_coordinates = without_collinear;
  size = input_coordinates.size();
  if (size < 3) {
    printf("Convex hull doesn't exist!!\n");
    return std::vector<Coordinates>();
  }
  Stack stack_coordinates;
  stack_coordinates.push(0);
  stack_coordinates.push(1);
  stack_coordinates.push(2);
  for (int i = 3; i < size; i++) {
    while (input_coordinates[stack_coordinates.next_to_top()].turn(
               input_coordinates[stack_coordinates.peek()],
               input_coordinates[i]) != ANTICLOCKWISE) {
      stack_coordinates.pop();
    }
    stack_coordinates.push(i);
  }
  std::vector<Coordinates> hull;
  while (!stack_coordinates.empty()) {
    hull.push_back(input_coordinates[stack_coordinates.pop()]);
  }
  std::reverse(hull.begin(), hull.end());
  return hull;
}
