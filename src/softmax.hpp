#pragma once
#include <Eigen/Dense>
#include <iostream>

class Softmax {
public:
    Softmax() = default;
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) {
        cache_ = input;
        Eigen::VectorXd rowMax = input.rowwise().maxCoeff();
        Eigen::MatrixXd shifted = input.array().colwise() - rowMax.array();
        Eigen::MatrixXd expTensor = shifted.array().exp();
        Eigen::VectorXd rowSum = expTensor.rowwise().sum();
        output_ = (expTensor.array().colwise() / rowSum.array()).matrix();
        return output_;
    }
    Eigen::MatrixXd backward(const Eigen::MatrixXd &grad) {
        Eigen::VectorXd sumWeighted = (grad.array() * output_.array()).rowwise().sum();
        Eigen::MatrixXd sumMat = sumWeighted.replicate(1, grad.cols());
        return output_.array() * (grad.array() - sumMat.array());
    }
private:
    Eigen::MatrixXd cache_, output_;
};
