#pragma once

#include <Eigen/Dense>
#include <iostream>

class Softmax
{
private:
    Eigen::MatrixXd inputTensorCache; // stores z (the logits) from forward pass
    Eigen::MatrixXd yHat;             // stores softmax output

public:
    Softmax();
    ~Softmax();

    // forward(inputTensor):
    //   shape [batch_size x num_classes], returns softmax probabilities
    Eigen::MatrixXd forward(const Eigen::MatrixXd &inputTensor);

    // backward(errorTensor):
    //   shape [batch_size x num_classes], returns gradient w.r.t. input logits
    Eigen::MatrixXd backward(const Eigen::MatrixXd &errorTensor);
};

Softmax::Softmax() {}
Softmax::~Softmax() {}

Eigen::MatrixXd Softmax::forward(const Eigen::MatrixXd &inputTensor)
{
    inputTensorCache = inputTensor; // might be helpful if you want to do derivative directly

    // For numerical stability, subtract max in each row
    // shape: [batch_size x 1]
    Eigen::VectorXd rowMax = inputTensor.rowwise().maxCoeff();
    // broadcasting subtract
    Eigen::MatrixXd shifted = inputTensor.colwise() - rowMax;

    // exponentiate
    Eigen::MatrixXd expTensor = shifted.array().exp();

    // sum each row
    Eigen::VectorXd rowSum = expTensor.rowwise().sum();

    // broadcast divide
    yHat = expTensor.array().colwise() / rowSum.array();

    return yHat;
}

Eigen::MatrixXd Softmax::backward(const Eigen::MatrixXd &errorTensor)
{
    // The typical formula for softmax derivative can be implemented in vectorized form:
    //
    // dL/dz_i = yhat_i * ( error_i - sum_j( error_j * yhat_j ) )
    //
    // shape [batch_size x num_classes]

    // rowwise sum of (errorTensor * yHat) across columns
    Eigen::VectorXd weightedErrorSum = (errorTensor.array() * yHat.array()).rowwise().sum();

    // replicate that sum across columns so we can subtract it
    // shape: [batch_size x num_classes]
    Eigen::MatrixXd sumMatrix = weightedErrorSum.replicate(1, errorTensor.cols());

    // adjustedError = errorTensor - sumMatrix
    Eigen::MatrixXd adjustedError = errorTensor.array() - sumMatrix.array();

    // multiply by yHat
    Eigen::MatrixXd gradInput = yHat.array() * adjustedError.array();

    return gradInput;
}
