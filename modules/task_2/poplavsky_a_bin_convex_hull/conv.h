
//   Copyright 2022 Poplavsky Anton
#ifndef MODULES_TASK_2_POPLAVSKY_A_BIN_CONVEX_HULL_CONV_H_
#define MODULES_TASK_2_POPLAVSKY_A_BIN_CONVEX_HULL_CONV_H_

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <list>
#include <random>
#include <stack>
#include <vector>
#define SIZE 110
#define COLLINEAR 0
#define CLOCKWISE -1
#define ANTICLOCKWISE 1
#include <iostream>

class Stack {
  std::vector<int> stack_;
  int capacity;
  int size;
  int top;

 public:
  Stack() {
    capacity = SIZE;
    stack_.resize(capacity);
    top = -1;
    size = 0;
  }
  void push(int elem) {
    try {
      if (size == capacity) {
        throw std::logic_error("Stack overflow");
      }
      top++;
      size++;
      stack_[top] = elem;
    } catch (const char* msg) {
      printf("%s\n", msg);
      exit(0);
    }
  }
  int pop() {
    try {
      if (size == 0) {
        throw std::logic_error("Stack pop");
      }
      top--;
      size--;
      return stack_[top + 1];
    } catch (const char* msg) {
      printf("%s\n", msg);
      exit(0);
    }
  }
  int peek() {
    try {
      if (size == 0) {
        throw std::logic_error("Stack peek");
      }
      return stack_[top];
    } catch (const char* msg) {
      printf("%s\n", msg);
      exit(0);
    }
  }
  bool empty() { return size == 0; }
  int next_to_top() {
    try {
      if (size == 1) {
        throw std::logic_error("Stack undeflow");
      }
      return stack_[top - 1];
    } catch (const char* msg) {
      printf("%s\n", msg);
      exit(0);
    }
  }
};

struct Coordinates {
  int X;
  int Y;
  Coordinates() {}
  Coordinates(int _X, int _Y) : X(_X), Y(_Y) {}
  int turn(const Coordinates& p1, const Coordinates& p2) {
    int value = (p1.Y - Y) * (p2.X - p1.X) - (p1.X - X) * (p2.Y - p1.Y);
    if (value == 0)
      return COLLINEAR;
    else if (value > 0)
      return ANTICLOCKWISE;
    else
      return CLOCKWISE;
  }
  int vector_norm(const Coordinates& pt) {
    return (X - pt.X) * (X - pt.X) + (Y - pt.Y) * (Y - pt.Y);
  }
  void set_x(int _x) { X = _x; }
  void set_y(int _y) { Y = _y; }
};

int find_minimal_coordinates(const std::vector<Coordinates>& input_coordinates);

int find_minimal_coordinates(const std::vector<Coordinates>& input_coordinates);
std::vector<Coordinates> graham_omp(
    const std::vector<Coordinates>& _input_coordinates);
std::vector<Coordinates> graham_seq(
    const std::vector<Coordinates>& _input_coordinates);

#endif  // MODULES_TASK_2_POPLAVSKY_A_BIN_CONVEX_HULL_CONV_H_
