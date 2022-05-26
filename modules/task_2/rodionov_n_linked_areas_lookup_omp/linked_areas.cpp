// Copyright 2022 Nikita Rodionov
#include "../../../modules/task_2/rodionov_n_linked_areas_lookup_omp/linked_areas.h"

#include <omp.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <random>
#include <vector>

BinaryImage GenerateBinrayImage(int size) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(0, 1);

  bool** image = new bool*[size];
  for (int x = 0; x < size; x++) {
    image[x] = new bool[size];
    for (int y = 0; y < size; y++) {
      image[x][y] = dist(rng);
    }
  }
  return BinaryImage(image, size);
}

BinaryImageAreas* FindAreas(BinaryImage image) {
  BinaryImageAreas* areas = new BinaryImageAreas();
  areas->size = image.size;
  areas->image = new int*[image.size];
  for (int x = 0; x < image.size; x++) {
    areas->image[x] = new int[image.size];
    for (int y = 0; y < image.size; y++) {
      areas->image[x][y] = -1;
    }
  }

  Equivalents equivalents;
  int label = 0;
  // First pass
  for (int x = 0; x < image.size; x++)
    for (int y = 0; y < image.size; y++) {
      if (!image.Get(x, y)) {
        continue;
      }

      int _1 = areas->Get(x, y + 1);
      int _2 = areas->Get(x + 1, y);
      int _3 = areas->Get(x, y - 1);
      int _4 = areas->Get(x - 1, y);

      int total_non_bg =
          static_cast<int>(_1 != -1) + static_cast<int>(_2 != -1) +
          static_cast<int>(_3 != -1) + static_cast<int>(_4 != -1);
      if (total_non_bg == 0) {
        areas->image[x][y] = label;
        label++;
      } else if (total_non_bg == 1) {
        if (_1 != -1) {
          areas->image[x][y] = _1;
        }
        if (_2 != -1) {
          areas->image[x][y] = _2;
        }
        if (_3 != -1) {
          areas->image[x][y] = _3;
        }
        if (_4 != -1) {
          areas->image[x][y] = _4;
        }
      } else {
        int nonzero[4] = {-1};
        int sz = 0;
        if (_1 != -1) {
          nonzero[sz] = _1;
          sz++;
        }
        if (_2 != -1) {
          nonzero[sz] = _2;
          sz++;
        }
        if (_3 != -1) {
          nonzero[sz] = _3;
          sz++;
        }
        if (_4 != -1) {
          nonzero[sz] = _4;
          sz++;
        }
        areas->image[x][y] = nonzero[0];

        for (int i = 0; i < sz; i++) {
          if (nonzero[i] == -1) {
            continue;
          }
          equivalents.Add(nonzero[0], nonzero[i]);
        }
      }
    }

  // Second pass
  for (int x = 0; x < image.size; x++) {
    for (int y = 0; y < image.size; y++) {
      int _00 = areas->Get(x, y);
      areas->image[x][y] = equivalents.Get(_00);
    }
  }
  areas->components = equivalents.equivalents.size();
  return areas;
}

BinaryImageAreas* FindSubArea(BinaryImage image, int xStart, int yStart,
                              int size, int startLabel) {
  BinaryImageAreas* areas = new BinaryImageAreas();
  areas->size = size;
  areas->image = new int*[size];
  for (int x = 0; x < size; x++) {
    areas->image[x] = new int[size];
    for (int y = 0; y < size; y++) {
      areas->image[x][y] = -1;
    }
  }
  int label = startLabel;
  Equivalents equivalents;
  // First pass
  for (int x = 0; x < size; x++)
    for (int y = 0; y < size; y++) {
      if (!image.Get(xStart + x, yStart + y)) {
        continue;
      }

      int _1 = areas->Get(x, y + 1);
      int _2 = areas->Get(x + 1, y);
      int _3 = areas->Get(x, y - 1);
      int _4 = areas->Get(x - 1, y);

      int total_non_bg =
          static_cast<int>(_1 != -1) + static_cast<int>(_2 != -1) +
          static_cast<int>(_3 != -1) + static_cast<int>(_4 != -1);
      if (total_non_bg == 0) {
        areas->image[x][y] = label;
        label++;
      } else if (total_non_bg == 1) {
        if (_1 != -1) {
          areas->image[x][y] = _1;
        }
        if (_2 != -1) {
          areas->image[x][y] = _2;
        }
        if (_3 != -1) {
          areas->image[x][y] = _3;
        }
        if (_4 != -1) {
          areas->image[x][y] = _4;
        }
      } else {
        int nonzero[4] = {-1};
        int sz = 0;
        if (_1 != -1) {
          nonzero[sz] = _1;
          sz++;
        }
        if (_2 != -1) {
          nonzero[sz] = _2;
          sz++;
        }
        if (_3 != -1) {
          nonzero[sz] = _3;
          sz++;
        }
        if (_4 != -1) {
          nonzero[sz] = _4;
          sz++;
        }
        areas->image[x][y] = nonzero[0];
        for (int i = 0; i < sz; i++) {
          if (nonzero[i] == -1) {
            continue;
          }
          equivalents.Add(nonzero[0], nonzero[i]);
        }
      }
    }

  // Second pass
  for (int x = 0; x < size; x++) {
    for (int y = 0; y < size; y++) {
      int _00 = areas->Get(x, y);
      areas->image[x][y] = equivalents.Get(_00);
    }
  }
  areas->components = label;
  return areas;
}

BinaryImageAreas* FindAreasOmp(BinaryImage image) {
  BinaryImageAreas* areas = new BinaryImageAreas();
  areas->size = image.size;
  areas->image = new int*[image.size];
  for (int x = 0; x < image.size; x++) {
    areas->image[x] = new int[image.size];
    for (int y = 0; y < image.size; y++) {
      areas->image[x][y] = -1;
    }
  }

  int onedimblocks = 5;
  int blockSize = (image.size / onedimblocks);
  if (blockSize == 0) {
    onedimblocks = 1;
    blockSize = (image.size / onedimblocks);
  }
  int blockSizeArea =
      (image.size / onedimblocks) * (image.size / onedimblocks) + 1;

  std::map<int, BinaryImageAreas*> blocks;
#pragma omp parallel for
  for (int block = 0; block < onedimblocks * onedimblocks; block++) {
    int offsetX = (block / onedimblocks) * image.size / onedimblocks;
    int offsetY = (block % onedimblocks) * image.size / onedimblocks;
    int startLabel = block * blockSizeArea;
    auto subarea = FindSubArea(image, offsetX, offsetY, blockSize, startLabel);
#pragma omp critical
    blocks.insert(std::make_pair(
        block, subarea));
  }

  // Merge vertical
  for (int row = 0; row < onedimblocks - 1; row++) {
    for (int column = 0; column < onedimblocks; column++) {
      BinaryImageAreas* current = blocks[row * onedimblocks + column];
      BinaryImageAreas* bottom = blocks[(row + 1) * onedimblocks + column];
      Equivalents equivalents;
      // Go over merge line and collect equivalents
      for (int i = 0; i < blockSize; i++) {
        int first = current->Get(i, current->size - 1);
        int second = bottom->Get(i, 0);
        if ((first != -1) && (second != -1)) {
          equivalents.Add(first, second);
        }
      }
      // Apply equivalents to first block
      for (int x = 0; x < blockSize; x++) {
        for (int y = 0; y < blockSize; y++) {
          int _00 = current->Get(x, y);
          current->image[x][y] = equivalents.Get(_00);
        }
      }
      // Apply equivalents to second block
      for (int x = 0; x < blockSize; x++) {
        for (int y = 0; y < blockSize; y++) {
          int _00 = bottom->Get(x, y);
          bottom->image[x][y] = equivalents.Get(_00);
        }
      }
    }
  }
  // Merge horizontal
  for (int row = 0; row < onedimblocks; row++) {
    for (int column = 0; column < onedimblocks - 1; column++) {
      BinaryImageAreas* current = blocks[row * onedimblocks + column];
      BinaryImageAreas* bottom = blocks[row * onedimblocks + column + 1];
      Equivalents equivalents;
      // Go over merge line and collect equivalents
      for (int i = 0; i < blockSize; i++) {
        int first = current->Get(i, current->size - 1);
        int second = bottom->Get(i, 0);
        if ((first != -1) && (second != -1)) {
          equivalents.Add(first, second);
        }
      }
      // Apply equivalents to first block
      for (int x = 0; x < blockSize; x++) {
        for (int y = 0; y < blockSize; y++) {
          int _00 = current->Get(x, y);
          current->image[x][y] = equivalents.Get(_00);
        }
      }
      // Apply equivalents to second block
      for (int x = 0; x < blockSize; x++) {
        for (int y = 0; y < blockSize; y++) {
          int _00 = bottom->Get(x, y);
          bottom->image[x][y] = equivalents.Get(_00);
        }
      }
    }
  }

  // Restore initial areas from blocks
  for (int block = 0; block < onedimblocks * onedimblocks; block++) {
    int offsetX = (block / onedimblocks) * image.size / onedimblocks;
    int offsetY = (block % onedimblocks) * image.size / onedimblocks;
    BinaryImageAreas* current = blocks.at(block);
    for (int x = 0; x < blockSize; x++) {
      for (int y = 0; y < blockSize; y++) {
        areas->image[offsetX + x][offsetY + y] = current->Get(x, y);
      }
    }
  }

  areas->components = 0;
  return areas;
}
