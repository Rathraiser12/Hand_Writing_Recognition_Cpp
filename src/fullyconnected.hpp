#ifndef ADVPT_MPU_FULLY_CONNECTED_HPP
#define ADVPT_MPU_FULLY_CONNECTED_HPP

#include "Eigen/Dense"
#include "optimizers.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>

// FullyConnected layer with bias incorporated
class FullyConnected
{
private:
    Eigen::MatrixXd weights; // shape: (input_size + 1) x output_size
    size_t input_size;
    size_t output_size;
    double range;

    // Cache the augmented input (with bias column) for backward pass
    Eigen::MatrixXd input_tensor;

public:
    FullyConnected() {}
    FullyConnected(size_t in, size_t out) : input_size(in), output_size(out)
    {
        // He initialization: typically for ReLU you want bounds based on in
        range = 1.0 / std::sqrt(input_size);
        // weights: (in + 1) x out, with an extra row for bias
        weights = Eigen::MatrixXd::Random(input_size + 1, output_size) * range;
    }

    // Optional setter for testing
    void setWeights(const Eigen::MatrixXd &w) {
        weights = w;
    }

    // Forward pass:
    // Input: Matrix [batch_size x input_size]
    // Output: Matrix [batch_size x output_size]
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) {
        size_t batch_size = input.rows();
        // Create augmented input: first input_size columns from input, last column = ones (for bias)
        input_tensor.resize(batch_size, input_size + 1);
        input_tensor.block(0, 0, batch_size, input_size) = input;
        input_tensor.col(input_size) = Eigen::VectorXd::Ones(batch_size);
        // Multiply: [batch_size x (input_size+1)] * [(input_size+1) x output_size] 
        Eigen::MatrixXd output = input_tensor * weights;
        return output;
    }

    // Backward pass:
    // error_tensor: gradient from next layer [batch_size x output_size]
    // Returns: gradient for previous layer [batch_size x input_size] (excluding bias derivative)
    Eigen::MatrixXd backward(const Eigen::MatrixXd &error_tensor, SGD &sgd) {
        // Compute gradient w.r.t. weights: (input_tensor^T [ (in+1) x batch_size ] * error_tensor [batch_size x out])
        Eigen::MatrixXd gradient_weights = input_tensor.transpose() * error_tensor;
        // Update weights using SGD
        weights = sgd.updateWeights(weights, gradient_weights);
        // Compute error for previous layer:
        // next_error = error_tensor * weights^T gives shape [batch_size x (input_size+1)]
        Eigen::MatrixXd next_error = error_tensor * weights.transpose();
        // Exclude the bias column: return only the first 'input_size' columns
        Eigen::MatrixXd next_error_no_bias = next_error.leftCols(input_size);
        return next_error_no_bias;
    }

    ~FullyConnected() {}
};

#endif // ADVPT_MPU_FULLY_CONNECTED_HPP
