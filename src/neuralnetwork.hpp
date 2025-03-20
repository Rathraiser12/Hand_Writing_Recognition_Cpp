#pragma once
#include <iostream>
#include <chrono>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>
#include <omp.h>
#include <numeric>
#include <algorithm>
#include <random>

#include "loss.hpp"
#include "optimizers.hpp"
#include "relu.hpp"
#include "softmax.hpp"
#include "fullyconnected.hpp"
#include "mnist_data_loader.hpp"  // Integrated loader for images & labels

class NeuralNetwork {
public:
    NeuralNetwork(double lr, int epochs, int batch, int hidden,
                  std::string trainData, std::string trainLabels,
                  std::string testData, std::string testLabels,
                  std::string logPath)
        : learning_rate(lr), num_epochs(epochs), batch_size(batch), hidden_size(hidden),
          train_data_path(trainData), train_labels_path(trainLabels),
          test_data_path(testData), test_labels_path(testLabels),
          log_file_path(logPath), input_size(784),
          fc1(input_size, hidden_size), fc2(hidden_size, 10),
          sgd(lr) {}

    void train() {
        auto start = std::chrono::steady_clock::now();
        // Use the integrated data loader for training data.
        MNISTDataLoader trainLoader(train_data_path, train_labels_path, batch_size);
        trainLoader.loadDataset();
        size_t numBatches = trainLoader.getNumBatches();
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::cout << "Epoch " << epoch << " / " << num_epochs << "\n";
            std::vector<size_t> indices(numBatches);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), std::default_random_engine(epoch));
            for (auto idx : indices) {
                Eigen::MatrixXd images = trainLoader.getImageBatch(idx);
                Eigen::MatrixXd labels = trainLoader.getLabelBatch(idx);
                Eigen::MatrixXd predictions = forward(images);
                double loss = loss_.forward(predictions, labels);
                Eigen::MatrixXd dLoss = loss_.backward(labels);
                backward(dLoss);
            }
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Total training time: " << elapsed.count() << " seconds\n";
    }

void test() {
    // Use the integrated data loader for test data.
    MNISTDataLoader testLoader(test_data_path, test_labels_path, batch_size);
    testLoader.loadDataset();
    std::ostringstream buffer;
    int total = 0, correct = 0;
    for (size_t b = 0; b < testLoader.getNumBatches(); ++b) {
        // Print the header with the exact expected text:
        buffer << "Current batch: " << b << "\n";
        Eigen::MatrixXd images = testLoader.getImageBatch(b);
        Eigen::MatrixXd predictions = forward(images);
        Eigen::MatrixXd labels = testLoader.getLabelBatch(b);
        for (int i = 0; i < predictions.rows(); ++i) {
            Eigen::Index pred, actual;
            predictions.row(i).maxCoeff(&pred);
            labels.row(i).maxCoeff(&actual);
            buffer << " - image " << (b * batch_size + i)
                   << ": Prediction=" << pred << ". Label=" << actual << "\n";
            ++total;
            if (pred == actual)
                ++correct;
        }
    }
    std::ofstream logFile(log_file_path);
    if (!logFile.is_open()) {
        std::cerr << "Error: Cannot open log file: " << log_file_path << "\n";
        return;
    }
    logFile << buffer.str();
    logFile.close();
    std::cout << "Test accuracy: " << 100.0 * correct / total << "%\n";
}

    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) {
        Eigen::MatrixXd a1 = fc1.forward(input);
        Eigen::MatrixXd r = relu.forward(a1);
        Eigen::MatrixXd a2 = fc2.forward(r);
        return softmax.forward(a2);
    }
    Eigen::MatrixXd backward(const Eigen::MatrixXd &gradLoss) {
        Eigen::MatrixXd grad2 = fc2.backward(gradLoss, sgd);
        Eigen::MatrixXd gradRelu = relu.backward(grad2);
        return fc1.backward(gradRelu, sgd);
    }

private:
    double learning_rate;
    int num_epochs, batch_size, hidden_size, input_size;
    std::string train_data_path, train_labels_path, test_data_path, test_labels_path, log_file_path;
    FullyConnected fc1, fc2;
    Relu relu;
    Softmax softmax;
    CrossEntropyLoss loss_;
    SGD sgd;
};
