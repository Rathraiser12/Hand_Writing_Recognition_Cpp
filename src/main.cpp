#include <iostream>
#include <string>
#include <stdexcept>
#include "mnist_data_loader.hpp"  // Using the integrated loader
#include "tensor.hpp"             // Your custom Tensor class

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
            // Read a single image using the integrated loader.
            Eigen::MatrixXd imageMat = MNISTDataLoader::readSingleImage(inputFile, index);
            // Create a Tensor with the same dimensions as imageMat.
            std::vector<size_t> shape = { static_cast<size_t>(imageMat.rows()), static_cast<size_t>(imageMat.cols()) };
            Tensor<double> imageTensor(shape);
            for (size_t r = 0; r < shape[0]; ++r)
                for (size_t c = 0; c < shape[1]; ++c)
                    imageTensor({ r, c }) = imageMat(r, c);
            writeTensorToFile(imageTensor, outputFile);
            std::cout << "Successfully wrote image tensor to " << outputFile << "\n";
        } else {
            // Read a single label using the integrated loader.
            Eigen::MatrixXd labelMat = MNISTDataLoader::readSingleLabel(inputFile, index);
            // If labelMat has one column, treat it as a 1D vector.
            std::vector<size_t> shape;
            if (labelMat.cols() == 1)
                shape = { static_cast<size_t>(labelMat.rows()) };
            else
                shape = { static_cast<size_t>(labelMat.rows()), static_cast<size_t>(labelMat.cols()) };
            
            Tensor<double> labelTensor(shape);
            if (labelMat.cols() == 1) {
                for (size_t i = 0; i < shape[0]; ++i)
                    labelTensor({ i }) = labelMat(i, 0);
            } else {
                for (size_t r = 0; r < shape[0]; ++r)
                    for (size_t c = 0; c < shape[1]; ++c)
                        labelTensor({ r, c }) = labelMat(r, c);
            }
            writeTensorToFile(labelTensor, outputFile);
            std::cout << "Successfully wrote label tensor to " << outputFile << "\n";
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
