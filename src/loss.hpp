#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

#ifndef EPSILON
#define EPSILON 1e-10
#endif

// CrossEntropyLoss for a batch of data, shape [batch_size x num_classes]
class CrossEntropyLoss {
private:
    // We store the network's predictions from the forward pass here
    Eigen::MatrixXd predTensorCache;

public:
    CrossEntropyLoss();
    ~CrossEntropyLoss();

    // forward(inputTensor, labelTensor):
    //   - inputTensor: predicted probabilities (e.g., from softmax),
    //       shape [batch_size x num_classes]
    //   - labelTensor: one-hot ground-truth labels,
    //       shape [batch_size x num_classes]
    // returns the scalar cross-entropy loss (sum or mean).
    double forward(const Eigen::MatrixXd &inputTensor, const Eigen::MatrixXd &labelTensor);

    // backward(labelTensor):
    //   - returns dL/dInput of shape [batch_size x num_classes].
    Eigen::MatrixXd backward(const Eigen::MatrixXd &labelTensor);
};

CrossEntropyLoss::CrossEntropyLoss() {}
CrossEntropyLoss::~CrossEntropyLoss() {}

double CrossEntropyLoss::forward(const Eigen::MatrixXd &inputTensor, const Eigen::MatrixXd &labelTensor)
{
    // Cache the predicted probabilities for backprop.
    predTensorCache = inputTensor;

    // cross-entropy = - sum( y * log(yhat+eps) ) 
    // for each sample, then optionally average by dividing by batch_size
    // We'll do a sum here, you can divide if you prefer the mean.
    // shape [batch_size x num_classes].
    Eigen::ArrayXXd clippedInput = (inputTensor.array() + EPSILON).log();
    double loss = -(labelTensor.array() * clippedInput).sum();
    return loss;
}

Eigen::MatrixXd CrossEntropyLoss::backward(const Eigen::MatrixXd &labelTensor)
{
    // dL/dyhat = - y / (yhat + eps)
    // But since we added EPSILON in forward, we can do the same here or rely on input not being zero:
    Eigen::MatrixXd grad = -(labelTensor.array() / (predTensorCache.array() + EPSILON)).matrix();

    // shape is [batch_size x num_classes], same as inputTensor
    return grad;
}
