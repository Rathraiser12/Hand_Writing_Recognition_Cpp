#ifndef FULLY_CONNECTED_HPP
#define FULLY_CONNECTED_HPP
#include <Eigen/Dense>
#include "optimizers.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>

class FullyConnected {
public:
    FullyConnected(size_t in, size_t out) : in_size(in), out_size(out) {
        weights_.resize(in_size + 1, out_size);
        weights_.topRows(in_size) = heUniformInit(in_size, out_size);
        weights_.row(in_size).setZero();
    }
    void setWeights(const Eigen::MatrixXd &w) { weights_ = w; }
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) {
        size_t batch = input.rows();
        input_aug_.resize(batch, in_size + 1);
        input_aug_.block(0, 0, batch, in_size) = input;
        input_aug_.col(in_size) = Eigen::VectorXd::Ones(batch);
        return input_aug_ * weights_;
    }
    Eigen::MatrixXd backward(const Eigen::MatrixXd &grad, const SGD &sgd) {
        Eigen::MatrixXd gradWeights = input_aug_.transpose() * grad;
        Eigen::MatrixXd prevGrad = grad * weights_.topRows(in_size).transpose();
        weights_ = sgd.updateWeights(weights_, gradWeights);
        return prevGrad;
    }
private:
    size_t in_size, out_size;
    Eigen::MatrixXd weights_, input_aug_;
};

#endif // FULLY_CONNECTED_HPP
