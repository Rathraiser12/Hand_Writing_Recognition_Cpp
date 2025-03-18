#pragma once

#include <iostream>
#include <fstream>
#include <Eigen/Dense>

#include "loss.hpp"
#include "optimizers.hpp"
#include "relu.hpp"
#include "softmax.hpp"
#include "fullyconnected.hpp"
#include "data.hpp"
#include "label.hpp"

class NeuralNetwork
{
private:
    double learningRate;
    int numEpochs;
    int batchSize;
    int hiddenLayerSize;
    int inputSize = 784; // MNIST images (28x28)

    FullyConnected fc1;  // [784 -> hiddenLayerSize]
    FullyConnected fc2;  // [hiddenLayerSize -> 10]

    Relu relu;
    Softmax softmax;
    CrossEntropyLoss celoss;
    SGD sgd;

    std::string trainDataPath;
    std::string trainLabelsPath;
    std::string testDataPath;
    std::string testLabelsPath;
    std::string predictionLogFilePath;

public:
    NeuralNetwork(double lr,
                  int nEpochs,
                  int bSize,
                  int hLayerSize,
                  std::string tDataPath,
                  std::string tLabelsPath,
                  std::string tstDataPath,
                  std::string tstLabelsPath,
                  std::string predLogPath)
        : learningRate(lr), numEpochs(nEpochs), batchSize(bSize), hiddenLayerSize(hLayerSize),
          trainDataPath(tDataPath), trainLabelsPath(tLabelsPath),
          testDataPath(tstDataPath), testLabelsPath(tstLabelsPath),
          predictionLogFilePath(predLogPath), sgd(lr)
    {
        // Initialize FullyConnected layers with He initialization inside their constructors.
        fc1 = FullyConnected(inputSize, hiddenLayerSize);
        fc2 = FullyConnected(hiddenLayerSize, 10);
    }

    ~NeuralNetwork() {
        // Destructor (if any cleanup is needed)
    }

    // Forward pass through FC1 -> ReLU -> FC2 -> Softmax
    Eigen::MatrixXd forward(const Eigen::MatrixXd &inputTensor)
    {
        Eigen::MatrixXd out_fc1 = fc1.forward(inputTensor);
        Eigen::MatrixXd out_relu = relu.forward(out_fc1);
        Eigen::MatrixXd out_fc2 = fc2.forward(out_relu);
        Eigen::MatrixXd out_softmax = softmax.forward(out_fc2);
        return out_softmax;
    }

    // Backward pass: softmax -> FC2 -> ReLU -> FC1
    Eigen::MatrixXd backward(const Eigen::MatrixXd &errorTensor)
    {
        Eigen::MatrixXd grad_softmax = softmax.backward(errorTensor);
        Eigen::MatrixXd grad_fc2 = fc2.backward(grad_softmax, sgd);
        Eigen::MatrixXd grad_relu = relu.backward(grad_fc2);
        Eigen::MatrixXd grad_fc1 = fc1.backward(grad_relu, sgd);
        return grad_fc1;
    }

    // Train the network using the provided training data and labels
    void train()
    {
        DataSetImages trainData(batchSize);
        trainData.readImageData(trainDataPath);

        DatasetLabels trainLabels(batchSize);
        trainLabels.readLabelData(trainLabelsPath);

        size_t numBatches = trainData.getNoOfBatches();

        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            for (size_t b = 0; b < numBatches; b++)
            {
                Eigen::MatrixXd batchImages = trainData.getBatch(b);
                Eigen::MatrixXd batchLabels = trainLabels.getBatch(b);

                Eigen::MatrixXd predictions = forward(batchImages);

                double lossVal = celoss.forward(predictions, batchLabels);
                // Optionally print or log the loss

                Eigen::MatrixXd dLoss = celoss.backward(batchLabels);
                backward(dLoss);
            }
        }
    }

    // Test the network and log predictions to file
    void test()
    {
        DataSetImages testDataObj(batchSize);
        testDataObj.readImageData(testDataPath);

        DatasetLabels testLabelsObj(batchSize);
        testLabelsObj.readLabelData(testLabelsPath);

        std::ofstream predictionLogFile(predictionLogFilePath);
        if (!predictionLogFile.is_open())
        {
            std::cerr << "Error: Cannot open prediction log file: " << predictionLogFilePath << std::endl;
            return;
        }

        size_t numTestBatches = testDataObj.getNoOfBatches();

        for (size_t b = 0; b < numTestBatches; b++)
        {
            predictionLogFile << "Current batch: " << b << "\n";

            Eigen::MatrixXd batchImages = testDataObj.getBatch(b);
            Eigen::MatrixXd predictions = forward(batchImages);
            Eigen::MatrixXd batchLabels = testLabelsObj.getBatch(b);

            for (int i = 0; i < predictions.rows(); i++)
            {
                Eigen::Index predLabel;
                predictions.row(i).maxCoeff(&predLabel);

                Eigen::Index actualLabel;
                batchLabels.row(i).maxCoeff(&actualLabel);

                predictionLogFile << " - image " << (b * batchSize + i)
                                  << ": Prediction=" << predLabel
                                  << ". Label=" << actualLabel << "\n";
            }
        }
        predictionLogFile.close();
    }
};
