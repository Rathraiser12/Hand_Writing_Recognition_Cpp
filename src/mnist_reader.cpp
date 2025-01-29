// mnist_reader.cpp

#include "mnist_reader.h"

#include <fstream>

#include <iostream>



int MNISTReader::reverseInt(int i) {

    // This function swaps the bytes to handle endianness

    unsigned char c1, c2, c3, c4;

    c1 = i & 255;

    c2 = (i >> 8) & 255;

    c3 = (i >> 16) & 255;

    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;

}



// mnist_reader.cpp (continued)

Tensor<double> MNISTReader::readImageData(const std::string& filename, int imageIndex) {
    // Open the file
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Read the magic number
    int magicNumber = 0;
    file.read((char*)&magicNumber, sizeof(magicNumber));
    magicNumber = reverseInt(magicNumber);

    // Read number of images
    int numberOfImages = 0;
    file.read((char*)&numberOfImages, sizeof(numberOfImages));
    numberOfImages = reverseInt(numberOfImages);

    // Read number of rows
    int numberOfRows = 0;
    file.read((char*)&numberOfRows, sizeof(numberOfRows));
    numberOfRows = reverseInt(numberOfRows);

    // Read number of columns
    int numberOfColumns = 0;
    file.read((char*)&numberOfColumns, sizeof(numberOfColumns));
    numberOfColumns = reverseInt(numberOfColumns);

    // Check if the requested image index is valid
    if (imageIndex < 0 || imageIndex >= numberOfImages) {
        throw std::runtime_error("Image index out of range");
    }

    // Create tensor for single image (28x28)
    Tensor<double> imageTensor({static_cast<size_t>(numberOfRows), static_cast<size_t>(numberOfColumns)});

    // Skip to the requested image
    file.seekg(16 + imageIndex * numberOfRows * numberOfColumns, std::ios::beg);

    // Read the image data
    unsigned char pixel = 0;
    for (int r = 0; r < numberOfRows; ++r) {
        for (int c = 0; c < numberOfColumns; ++c) {
            file.read((char*)&pixel, 1);
            // Convert to double in range [0.0, 1.0]
            imageTensor({static_cast<size_t>(r), static_cast<size_t>(c)}) = static_cast<double>(pixel) / 255.0;
        }
    }

    file.close();
    return imageTensor;
}

Tensor<double> MNISTReader::readLabelData(const std::string& filename, int labelIndex)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Read the magic number (should be 2049 for labels)
    int magicNumber = 0;
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    magicNumber = reverseInt(magicNumber);
    if (magicNumber != 2049) {
        throw std::runtime_error("Invalid label file (magic != 2049)");
    }

    // Number of labels
    int numLabels = 0;
    file.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
    numLabels = reverseInt(numLabels);

    // Check range
    if (labelIndex < 0 || labelIndex >= numLabels) {
        throw std::runtime_error("Label index out of range");
    }

    // Skip to the requested label position.
    // Header is 8 bytes total; each label is 1 byte.
    file.seekg(8 + labelIndex, std::ios::beg);

    // Read the single label
    unsigned char labelByte = 0;
    file.read(reinterpret_cast<char*>(&labelByte), 1);
    if (!file.good()) {
        throw std::runtime_error("Error reading label at index " + std::to_string(labelIndex));
    }

    // Construct one-hot vector for the digit [0..9].
    // For instance, if labelByte=3, then oneHot = [0,0,0,1,0,0,0,0,0,0].
    Tensor<double> oneHot({10}); // shape = {10}
    for (int i = 0; i < 10; ++i) {
        oneHot({static_cast<size_t>(i)}) = 0.0;
    }
    oneHot({static_cast<size_t>(labelByte)}) = 1.0;

    file.close();
    return oneHot;
}