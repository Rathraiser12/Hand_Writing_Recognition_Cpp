#include <iostream>
#include <string>
#include <stdexcept>
#include "mnist_reader.h"
#include "tensor.hpp"

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset_input> <tensor_output> <index>\n\n"
                  << "Example (images): " << argv[0]
                  << " mnist-datasets/train-images.idx3-ubyte image_out.txt 0\n"
                  << "Example (labels): " << argv[0]
                  << " mnist-datasets/train-labels.idx1-ubyte label_out.txt 0\n";
        return 1;
    }
    std::string inputFile = argv[1], outputFile = argv[2];
    int index = std::stoi(argv[3]);
    bool isImage = (inputFile.find("images") != std::string::npos) ||
                   (inputFile.find("idx3-ubyte") != std::string::npos);
    try {
        if (isImage) {
            Tensor<double> imageTensor = MNISTReader::readImageData(inputFile, index);
            writeTensorToFile(imageTensor, outputFile);
            std::cout << "Successfully wrote image tensor to " << outputFile << "\n";
        } else {
            Tensor<double> labelTensor = MNISTReader::readLabelData(inputFile, index);
            writeTensorToFile(labelTensor, outputFile);
            std::cout << "Successfully wrote label tensor to " << outputFile << "\n";
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
