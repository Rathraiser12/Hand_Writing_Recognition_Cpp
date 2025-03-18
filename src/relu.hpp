#pragma once

#include <Eigen/Dense>

class Relu
{
private:
    // Caches the input to use during backprop
    Eigen::MatrixXd inputTensorCache;

public:
    Relu();
    ~Relu();

    // forward(inputTensor): shape [batch_size x features]
    Eigen::MatrixXd forward(const Eigen::MatrixXd &inputTensor);

    // backward(errorTensor):
    //   shape [batch_size x features], returns gradient to pass to prior layer
    Eigen::MatrixXd backward(const Eigen::MatrixXd &errorTensor);
};

Relu::Relu() {}
Relu::~Relu() {}

Eigen::MatrixXd Relu::forward(const Eigen::MatrixXd &inputTensor)
{
    // Save for backprop
    inputTensorCache = inputTensor;

    // ReLU(x) = max(0, x) elementwise
    Eigen::MatrixXd output = inputTensor.cwiseMax(0.0);
    return output;
}

Eigen::MatrixXd Relu::backward(const Eigen::MatrixXd &errorTensor)
{
    // We assume errorTensor has same shape as inputTensorCache
    // mask: 1 where input > 0, else 0
    Eigen::MatrixXd mask = (inputTensorCache.array() > 0.0).cast<double>();

    // dL/dx = dL/dy * dy/dx
    // For ReLU, dy/dx = 1 if x>0 else 0
    Eigen::MatrixXd gradInput = errorTensor.array() * mask.array();

    return gradInput;
}
