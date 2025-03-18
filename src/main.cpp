#include <iostream>
#include <string>
#include <stdexcept>
#include "mnist_reader.h"    // Your MNISTReader class
#include "tensor.hpp"        // For writeTensorToFile (or include wherever it's defined)

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        // The project description requires exactly 3 arguments:
        //   <input_file> <output_file> <index>
        // For example:
        //   ./mnist_reader mnist-datasets/train-images.idx3-ubyte image_out.txt 0
        //   ./mnist_reader mnist-datasets/train-labels.idx1-ubyte label_out.txt 0
        std::cerr << "Usage: " << argv[0]
                  << " <dataset_input> <tensor_output> <index>\n\n"
                  << "Example (images): " << argv[0]
                  << " mnist-datasets/train-images.idx3-ubyte image_out.txt 0\n"
                  << "Example (labels): " << argv[0]
                  << " mnist-datasets/train-labels.idx1-ubyte label_out.txt 0\n";
        return 1;
    }

    // Parse the arguments
    std::string inputFile  = argv[1];
    std::string outputFile = argv[2];
    int index              = std::stoi(argv[3]);

    // Detect if we are reading images or labels by checking the filename
    // (One approach: "images" => readImageData, "labels" => readLabelData)
    // You can also check ".idx3-ubyte" vs. ".idx1-ubyte".
    bool isImage = (inputFile.find("images") != std::string::npos)
                || (inputFile.find("idx3-ubyte") != std::string::npos);

    try
    {
        if (isImage)
        {
            // Read a single MNIST image at 'index'
            Tensor<double> imageTensor = MNISTReader::readImageData(inputFile, index);

            // Write the resulting image tensor to file
            writeTensorToFile(imageTensor, outputFile);

            std::cout << "Successfully wrote image tensor to " << outputFile << std::endl;
        }
        else
        {
            // Read a single MNIST label at 'index'
            Tensor<double> labelTensor = MNISTReader::readLabelData(inputFile, index);

            // Write the resulting label tensor to file
            writeTensorToFile(labelTensor, outputFile);

            std::cout << "Successfully wrote label tensor to " << outputFile << std::endl;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
