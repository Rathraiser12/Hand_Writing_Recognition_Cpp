#pragma once

#include <Eigen/Dense>

class Relu
{
private:
    Eigen::MatrixXd inputTensorCache;
public:
    Relu();
    ~Relu();

    Eigen::MatrixXd forward(const Eigen::MatrixXd &inputTensor);
    Eigen::MatrixXd backward(const Eigen::MatrixXd &errorTensor);
};

Relu::Relu() {}
Relu::~Relu() {}

Eigen::MatrixXd Relu::forward(const Eigen::MatrixXd &inputTensor)
{
    inputTensorCache = inputTensor;
    return inputTensor.cwiseMax(0.0);
}

Eigen::MatrixXd Relu::backward(const Eigen::MatrixXd &errorTensor)
{
    Eigen::MatrixXd mask = (inputTensorCache.array() > 0.0).cast<double>();
    return errorTensor.array() * mask.array();
}
