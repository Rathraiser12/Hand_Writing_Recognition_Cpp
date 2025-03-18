#include <iostream>
#include <string>
#include "neuralnetwork.hpp"
#include "Eigen/Dense"

int main(int argc, char** argv)
{
    // We expect 10 total arguments:
    // argv[0] = program name
    // argv[1] = learningRate
    // argv[2] = numEpochs
    // argv[3] = batchSize
    // argv[4] = hiddenLayerSize
    // argv[5] = trainDataPath
    // argv[6] = trainLabelsPath
    // argv[7] = testDataPath
    // argv[8] = testLabelsPath
    // argv[9] = predictionLogFilePath

    if (argc != 10)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <learningRate> <numEpochs> <batchSize> <hiddenLayerSize>"
                  << " <trainDataPath> <trainLabelsPath> <testDataPath> <testLabelsPath> <predictionLogFilePath>\n"
                  << "Example:\n  " << argv[0]
                  << " 0.001 5 64 128 "
                     "mnist-datasets/train-images.idx3-ubyte mnist-datasets/train-labels.idx1-ubyte "
                     "mnist-datasets/t10k-images.idx3-ubyte mnist-datasets/t10k-labels.idx1-ubyte "
                     "prediction_log.txt\n";
        return 1;
    }

    // Parse arguments
    double learningRate      = std::stod(argv[1]);
    int    numEpochs         = std::stoi(argv[2]);
    int    batchSize         = std::stoi(argv[3]);
    int    hiddenLayerSize   = std::stoi(argv[4]);
    std::string trainDataPath    = argv[5];
    std::string trainLabelsPath  = argv[6];
    std::string testDataPath     = argv[7];
    std::string testLabelsPath   = argv[8];
    std::string predictionLogFilePath = argv[9];

    // Construct and run the neural network
    NeuralNetwork nn(
        learningRate,
        numEpochs,
        batchSize,
        hiddenLayerSize,
        trainDataPath,
        trainLabelsPath,
        testDataPath,
        testLabelsPath,
        predictionLogFilePath
    );

    std::cout << "Starting training with the following hyperparameters:\n"
              << "  Learning rate: " << learningRate << "\n"
              << "  Epochs:        " << numEpochs    << "\n"
              << "  Batch size:    " << batchSize    << "\n"
              << "  Hidden size:   " << hiddenLayerSize << "\n\n";

    nn.train();

    std::cout << "\nTraining completed. Now running test phase...\n";
    nn.test();

    std::cout << "Test completed. Predictions logged to: " << predictionLogFilePath << "\n";
    return 0;
}
