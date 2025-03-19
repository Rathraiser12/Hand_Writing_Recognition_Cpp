#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

#ifndef EPSILON
#define EPSILON 1e-10
#endif

// CrossEntropyLoss for a batch of data (expects softmax outputs).
// Shapes: [batch_size x num_classes]
class CrossEntropyLoss {
private:
    Eigen::MatrixXd predTensorCache; // Cache softmax outputs.
public:
    CrossEntropyLoss();
    ~CrossEntropyLoss();

    // Computes average cross-entropy loss.
    double forward(const Eigen::MatrixXd &inputTensor, const Eigen::MatrixXd &labelTensor);

    // Returns gradient: (yhat - y)/N.
    Eigen::MatrixXd backward(const Eigen::MatrixXd &labelTensor);
};

CrossEntropyLoss::CrossEntropyLoss() {}
CrossEntropyLoss::~CrossEntropyLoss() {}

double CrossEntropyLoss::forward(const Eigen::MatrixXd &inputTensor, const Eigen::MatrixXd &labelTensor)
{
    predTensorCache = inputTensor;
    double loss = - (labelTensor.array() * (inputTensor.array() + EPSILON).log()).sum();
    loss /= static_cast<double>(inputTensor.rows());
    return loss;
}

Eigen::MatrixXd CrossEntropyLoss::backward(const Eigen::MatrixXd &labelTensor)
{
    return (predTensorCache - labelTensor) / static_cast<double>(predTensorCache.rows());
}
