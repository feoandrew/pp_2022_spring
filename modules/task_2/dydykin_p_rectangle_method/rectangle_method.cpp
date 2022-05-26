// Copyright 2022 Dydykin Pavel
#include "../../../modules/task_2/dydykin_p_rectangle_method/rectangle_method.h"

double Rectangle_Method_Seq(std::vector <int> start_integ,
    std::vector <int> end_integ,
    const int count, std::function<double(std::vector<double>)> function) {
    if (start_integ.size() != end_integ.size()) {
        throw "Integration limits are not equal";
    }

    if (count <= 0) {
        throw "Incorrect step size";
    }

    int degree_of_integration = start_integ.size();

    for (int i = 0; i < degree_of_integration; i++) {
        if (start_integ[i] > end_integ[i]) {
            throw "Incorrect integration limits";
        }
    }

    double result = 1;
    double sum = 0.0;
    std::vector<double> h;
    std::vector<double> res;

    for (int i = 0; i < degree_of_integration; i++) {
        h.push_back(static_cast<double> (end_integ[i] - start_integ[i])
            /static_cast<double>(count));
    }

    std::vector<double> combinations(degree_of_integration);
    for (int i = 0; i < degree_of_integration; i++) {
        for (int j = 0; j < count; j++) {
            combinations[i] = (start_integ[i] + j * h[i]);
            sum += function(combinations) * h[i];
        }
        res.push_back(sum);
    }

    for (int i = 0; i < degree_of_integration; i++) {
        result *= res[i];
    }

    return result;
}

double Rectangle_Method_Parallel(std::vector <int> start_integ,
    std::vector <int> end_integ,
    const int count, std::function<double(std::vector<double>)> function) {
    if (start_integ.size() != end_integ.size()) {
        throw "Integration limits are not equal";
    }

    if (count <= 0) {
        throw "Incorrect step size";
    }

    int degree_of_integration = start_integ.size();

    for (int i = 0; i < degree_of_integration; i++) {
        if (start_integ[i] > end_integ[i]) {
            throw "Incorrect integration limits";
        }
    }

    double result = 1;
    double sum = 0.0;
    std::vector<double> h;
    std::vector<double> res;

    for (int i = 0; i < degree_of_integration; i++) {
        h.push_back(static_cast<double> (end_integ[i] - start_integ[i])
            / static_cast<double>(count));
    }

    std::vector<double> combinations(degree_of_integration);
    for (int i = 0; i < degree_of_integration; i++) {
#pragma omp parallel for schedule(static, 1) reduction(+:sum)
        for (int j = 0; j < count; j++) {
            combinations[i] = (start_integ[i] + j * h[i]);
            sum += function(combinations) * h[i];
        }
        res.push_back(sum);
    }

#pragma omp parallel for reduction(*:result)
    for (int i = 0; i < degree_of_integration; i++) {
        result *= res[i];
    }

    return result;
}
