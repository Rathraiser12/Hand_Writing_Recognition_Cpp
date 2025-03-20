#include <iostream>
#include <string>
#include "neuralnetwork.hpp"

int main(int argc, char **argv) {
    if (argc != 10) {
        std::cerr << "Usage: " << argv[0]
                  << " <learningRate> <numEpochs> <batchSize> <hiddenLayerSize>"
                     " <trainDataPath> <trainLabelsPath> <testDataPath> <testLabelsPath> <predictionLogFilePath>\n";
        return 1;
    }
    double lr = std::stod(argv[1]);
    int epochs = std::stoi(argv[2]), batch = std::stoi(argv[3]), hidden = std::stoi(argv[4]);
    std::string trainData = argv[5], trainLabels = argv[6],
                testData = argv[7], testLabels = argv[8], logPath = argv[9];

    NeuralNetwork nn(lr, epochs, batch, hidden, trainData, trainLabels, testData, testLabels, logPath);
    std::cout << "Starting training with:\n"
              << " Learning rate: " << lr << "\n Epochs: " << epochs
              << "\n Batch size: " << batch << "\n Hidden size: " << hidden << "\n";
    nn.train();
    std::cout << "Training complete. Running test phase...\n";
    nn.test();
    std::cout << "Test completed. Predictions logged to: " << logPath << "\n";
    return 0;
}
