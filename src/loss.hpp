#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

#ifndef EPSILON
#define EPSILON 1e-10
#endif

// CrossEntropyLoss for a batch of data, where inputTensor is assumed to be the softmax output.
// Shapes: [batch_size x num_classes]
class CrossEntropyLoss {
private:
    // Cache the predictions (softmax outputs) from the forward pass.
    Eigen::MatrixXd predTensorCache;

public:
    CrossEntropyLoss();
    ~CrossEntropyLoss();

    // forward(inputTensor, labelTensor):
    //  - inputTensor: predicted probabilities (softmax outputs),
    //      shape [batch_size x num_classes]
    //  - labelTensor: one-hot ground-truth labels,
    //      shape [batch_size x num_classes]
    // Returns the average cross-entropy loss.
    double forward(const Eigen::MatrixXd &inputTensor, const Eigen::MatrixXd &labelTensor);

    // backward(labelTensor):
    // Returns dL/dInput (gradient with respect to softmax inputs),
    // with shape [batch_size x num_classes].
    Eigen::MatrixXd backward(const Eigen::MatrixXd &labelTensor);
};

CrossEntropyLoss::CrossEntropyLoss() {}
CrossEntropyLoss::~CrossEntropyLoss() {}

double CrossEntropyLoss::forward(const Eigen::MatrixXd &inputTensor, const Eigen::MatrixXd &labelTensor)
{
    // Cache predictions for backpropagation.
    predTensorCache = inputTensor;

    // Compute the cross-entropy loss:
    // loss = -1/N * sum( y * log(yhat + eps) )
    double loss = - (labelTensor.array() * (inputTensor.array() + EPSILON).log()).sum();
    loss /= static_cast<double>(inputTensor.rows());
    return loss;
}

Eigen::MatrixXd CrossEntropyLoss::backward(const Eigen::MatrixXd &labelTensor)
{
    // The gradient of the combined softmax and cross-entropy loss is:
    // (yhat - y) / N, where N is the batch size.
    return (predTensorCache - labelTensor) / static_cast<double>(predTensorCache.rows());
}
