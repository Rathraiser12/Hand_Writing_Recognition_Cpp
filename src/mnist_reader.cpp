#pragma once

#include "mnist_reader.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

int MNISTReader::reverseInt(int i)
{
    // This function swaps the bytes to handle endianness
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

Tensor<double> MNISTReader::readImageData(const std::string& filename, int imageIndex)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    int magicNumber = 0;
    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    magicNumber = reverseInt(magicNumber);  // handle endianness

    // For MNIST images, magicNumber should be 2051
    if (magicNumber != 2051)
        throw std::runtime_error("Invalid MNIST image file (magic != 2051)");

    int numberOfImages = 0;
    file.read(reinterpret_cast<char*>(&numberOfImages), 4);
    numberOfImages = reverseInt(numberOfImages);

    int numberOfRows = 0;
    file.read(reinterpret_cast<char*>(&numberOfRows), 4);
    numberOfRows = reverseInt(numberOfRows);

    int numberOfColumns = 0;
    file.read(reinterpret_cast<char*>(&numberOfColumns), 4);
    numberOfColumns = reverseInt(numberOfColumns);

    // Debug printing
    std::cout << "Read image file: " << filename << std::endl
              << "Magic: " << magicNumber
              << ", #images: " << numberOfImages
              << ", rows: " << numberOfRows
              << ", cols: " << numberOfColumns << std::endl;

    // Validate index
    if (imageIndex < 0 || imageIndex >= numberOfImages)
        throw std::runtime_error("Image index out of range");

    // Create a 2D tensor shape
    Tensor<double> imageTensor({
        static_cast<size_t>(numberOfRows),
        static_cast<size_t>(numberOfColumns)
    });

    // Seek to the start of the requested image
    const size_t imageSize = static_cast<size_t>(numberOfRows) * numberOfColumns;
    // 16 bytes for header; skip to imageIndex
    file.seekg(16 + imageIndex * imageSize, std::ios::beg);

    // Read each pixel into the 2D tensor
    for (int r = 0; r < numberOfRows; ++r)
    {
        for (int c = 0; c < numberOfColumns; ++c)
        {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            double value = static_cast<double>(pixel) / 255.0; // normalized
            imageTensor({(size_t)r, (size_t)c}) = value;
        }
    }

    file.close(); // close after reading
    return imageTensor;
}

Tensor<double> MNISTReader::readLabelData(const std::string& filename, int labelIndex)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    // Read the magic number (should be 2049 for labels)
    int magicNumber = 0;
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    magicNumber = reverseInt(magicNumber);
    if (magicNumber != 2049)
        throw std::runtime_error("Invalid label file (magic != 2049)");

    // Number of labels
    int numLabels = 0;
    file.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
    numLabels = reverseInt(numLabels);

    // Check range
    if (labelIndex < 0 || labelIndex >= numLabels)
        throw std::runtime_error("Label index out of range");

    // Skip to the requested label position.
    // Header is 8 bytes total; each label is 1 byte.
    file.seekg(8 + labelIndex, std::ios::beg);

    // Read the single label
    unsigned char labelByte = 0;
    file.read(reinterpret_cast<char*>(&labelByte), 1);
    if (!file.good())
        throw std::runtime_error("Error reading label at index " + std::to_string(labelIndex));

    // Construct one-hot vector for the digit [0..9].
    Tensor<double> oneHot({10}); // shape = {10}
    for (int i = 0; i < 10; ++i)
    {
        oneHot({(size_t)i}) = 0.0;
    }
    oneHot({(size_t)labelByte}) = 1.0;

    file.close();
    return oneHot;
}
