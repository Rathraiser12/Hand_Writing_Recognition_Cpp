#pragma once
#include <Eigen/Dense>
#include <random>
#include <cmath>

class SGD {
public:
    explicit SGD(double lr = 0.001) : lr_(lr) {}
    Eigen::MatrixXd updateWeights(const Eigen::MatrixXd &weights, const Eigen::MatrixXd &grad) const {
        return weights - lr_ * grad;
    }
private:
    double lr_;
};

inline Eigen::MatrixXd heUniformInit(int outDim, int inDim, unsigned int seed = 1337) {
    std::mt19937 rng(seed);
    double limit = std::sqrt(6.0 / static_cast<double>(inDim));
    std::uniform_real_distribution<double> dist(-limit, limit);
    Eigen::MatrixXd W(outDim, inDim);
    for (int r = 0; r < outDim; ++r)
        for (int c = 0; c < inDim; ++c)
            W(r, c) = dist(rng);
    return W;
}
