#include "mnist_reader.h"  // Declares MNISTReader class with readImageData/readLabelData
#include "tensor.hpp"      // Declares/defines Tensor<T>, writeTensorToFile, etc.

#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    // Expect 5 arguments: 
    //   1) mode (image|label)
    //   2) inputFile
    //   3) outputFile
    //   4) index
    //
    // Example for image:
    //   ./mnist_reader.exe image mnist-datasets/train-images.idx3-ubyte image_out.txt 0
    // Example for label:
    //   ./mnist_reader.exe label mnist-datasets/train-labels.idx1-ubyte label_out.txt 0
    //
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <mode: image|label> <dataset_input> <tensor_output> <index>\n";
        return 1;
    }

    // Parse command line
    std::string mode       = argv[1];
    std::string inputFile  = argv[2];
    std::string outputFile = argv[3];
    int index              = std::stoi(argv[4]);  // which image/label in the dataset

    try
    {
        if (mode == "image")
        {
            // Use your existing function to read an MNIST image at `index`
            Tensor<double> image = MNISTReader::readImageData(inputFile, index);

            // Write the image (28x28) to file using the inline function in tensor.hpp
            writeTensorToFile(image, outputFile);

            std::cout << "Wrote image " << index << " to " << outputFile << "\n";
        }
        else if (mode == "label")
        {
            // Use your newly added function to read a single label (one-hot) at `index`
            Tensor<double> labelOneHot = MNISTReader::readLabelData(inputFile, index);

            // Write the 1D, shape={10} tensor to file
            writeTensorToFile(labelOneHot, outputFile);

            std::cout << "Wrote label " << index << " (one-hot) to " << outputFile << "\n";
        }
        else
        {
            std::cerr << "Unknown mode: " << mode << "\n";
            return 1;
        }

        return 0; // success
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
