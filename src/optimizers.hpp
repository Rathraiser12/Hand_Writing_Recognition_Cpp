#pragma once

#include <Eigen/Dense>
#include <random>
#include <cmath>

// A simple SGD optimizer
class SGD {
private:
    double learningRate;
public:
    SGD();
    explicit SGD(double lr);
    ~SGD();

    // weights: the current weights
    // gradient: dL/dW for each weight
    Eigen::MatrixXd updateWeights(Eigen::MatrixXd &weights, const Eigen::MatrixXd &gradient);
};

SGD::SGD() : learningRate(0.001) {}

SGD::SGD(double lr) : learningRate(lr) {}

SGD::~SGD() {}

Eigen::MatrixXd SGD::updateWeights(Eigen::MatrixXd &weights, const Eigen::MatrixXd &gradient) {
    // w_new = w_old - lr * grad
    return weights - learningRate * gradient;
}

//------------------------------------------------------------------------------
//  HE (Kaiming) WEIGHT INITIALIZATION
//
//  Typically used for layers followed by ReLU. 
//  outDim = number of output neurons
//  inDim  = number of input  neurons
//------------------------------------------------------------------------------
inline Eigen::MatrixXd heUniformInit(int outDim, int inDim, unsigned int seed = 1337)
{
    static std::mt19937 rng(seed);
    
    // For ReLU, He initialization sets bounds ±sqrt(6 / inDim).
    // Sometimes it's sqrt(2/inDim) for normal, or sqrt(6/inDim) for uniform. 
    double limit = std::sqrt(6.0 / double(inDim));

    std::uniform_real_distribution<double> dist(-limit, limit);

    Eigen::MatrixXd W(outDim, inDim);
    for(int r = 0; r < outDim; ++r) {
        for(int c = 0; c < inDim; ++c) {
            W(r, c) = dist(rng);
        }
    }
    return W;
}
