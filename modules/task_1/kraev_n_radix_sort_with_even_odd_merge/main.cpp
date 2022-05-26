// Copyright 2022 Kraev Nikita
#include <gtest/gtest.h>
#include "./radix_sort_with_even_odd_merge.h"

TEST(Generate_Vector, Test_Generate_Random_Vector) {
    std::vector<double> random_vector = getRandomVector(100);
    ASSERT_TRUE(!random_vector.empty());
}

TEST(Sequential_Sort, Test_Radix_Sort_1) {
    std::vector<double> vec = { 1.0, -6.2, 7.45, -22.3457, 0.0, 45.69876, -0.03 };
    std::vector<double> sort_vec = { -22.3457, -6.2, -0.03, 0.0, 1.0, 7.45, 45.69876 };
    radixSort(&vec);
    ASSERT_EQ(sort_vec, vec);
}

TEST(Sequential_Sort, Test_Radix_Sort_2) {
    std::vector<double> random_vector = getRandomVector(100);
    std::vector<double> result_vector = random_vector;
    radixSort(&random_vector);
    std::sort(result_vector.begin(), result_vector.end());
    ASSERT_EQ(result_vector, random_vector);
}

TEST(Sequential_Sort, Test_Normal_Size_Vector) {
    std::vector<double> random_vector = getRandomVector(10000);
    std::vector<double> result_vector = random_vector;
    radixSort(&random_vector);
    std::sort(result_vector.begin(), result_vector.end());
    ASSERT_EQ(result_vector, random_vector);
}

TEST(Sequential_Sort, Test_Big_Size_Vector) {
    std::vector<double> random_vector = getRandomVector(100000);
    std::vector<double> result_vector = random_vector;
    radixSort(&random_vector);
    std::sort(result_vector.begin(), result_vector.end());
    ASSERT_EQ(result_vector, random_vector);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
