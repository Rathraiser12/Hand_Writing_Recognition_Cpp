#pragma once
#include <Eigen/Dense>
#include <cmath>

#ifndef EPSILON
#define EPSILON 1e-10
#endif

class CrossEntropyLoss {
public:
    CrossEntropyLoss() = default;
    double forward(const Eigen::MatrixXd &pred, const Eigen::MatrixXd &label) {
        cache_ = pred;
        double loss = - (label.array() * (pred.array() + EPSILON).log()).sum();
        return loss / static_cast<double>(pred.rows());
    }
    Eigen::MatrixXd backward(const Eigen::MatrixXd &label) {
        return (cache_ - label) / static_cast<double>(cache_.rows());
    }
private:
    Eigen::MatrixXd cache_;
};
