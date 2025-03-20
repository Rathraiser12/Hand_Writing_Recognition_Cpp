#pragma once
#include <Eigen/Dense>

class Relu {
public:
    Relu() = default;
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) {
        cache_ = input;
        return input.cwiseMax(0.0);
    }
    Eigen::MatrixXd backward(const Eigen::MatrixXd &grad) {
        return grad.array() * (cache_.array() > 0.0).cast<double>().array();
    }
private:
    Eigen::MatrixXd cache_;
};
