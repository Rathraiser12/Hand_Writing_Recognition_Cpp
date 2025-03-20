#include "mnist_reader.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

int MNISTReader::reverseInt(int i) {
    unsigned char c1 = i & 0xFF, c2 = (i >> 8) & 0xFF, c3 = (i >> 16) & 0xFF, c4 = (i >> 24) & 0xFF;
    return (int(c1) << 24) + (int(c2) << 16) + (int(c3) << 8) + c4;
}

Tensor<double> MNISTReader::readImageData(const std::string &filename, int imageIndex) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filename);
    int magic = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    magic = reverseInt(magic);
    if (magic != 2051)
        throw std::runtime_error("Invalid MNIST image file (magic != 2051)");
    int numImages = 0, numRows = 0, numCols = 0;
    file.read(reinterpret_cast<char*>(&numImages), 4); numImages = reverseInt(numImages);
    file.read(reinterpret_cast<char*>(&numRows), 4); numRows = reverseInt(numRows);
    file.read(reinterpret_cast<char*>(&numCols), 4); numCols = reverseInt(numCols);
    std::cout << "Read image file: " << filename << "\nMagic: " << magic
              << ", #images: " << numImages << ", rows: " << numRows << ", cols: " << numCols << "\n";
    if (imageIndex < 0 || imageIndex >= numImages)
        throw std::runtime_error("Image index out of range");
    Tensor<double> imageTensor({ static_cast<size_t>(numRows), static_cast<size_t>(numCols) });
    size_t imgSize = numRows * numCols;
    file.seekg(16 + imageIndex * imgSize, std::ios::beg);
    for (int r = 0; r < numRows; ++r)
        for (int c = 0; c < numCols; ++c) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            imageTensor({ (size_t)r, (size_t)c }) = static_cast<double>(pixel) / 255.0;
        }
    file.close();
    return imageTensor;
}

Tensor<double> MNISTReader::readLabelData(const std::string &filename, int labelIndex) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filename);
    int magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    magic = reverseInt(magic);
    if (magic != 2049)
        throw std::runtime_error("Invalid label file (magic != 2049)");
    int numLabels = 0;
    file.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
    numLabels = reverseInt(numLabels);
    if (labelIndex < 0 || labelIndex >= numLabels)
        throw std::runtime_error("Label index out of range");
    file.seekg(8 + labelIndex, std::ios::beg);
    unsigned char labelByte = 0;
    file.read(reinterpret_cast<char*>(&labelByte), 1);
    Tensor<double> oneHot({ 10 });
    for (int i = 0; i < 10; ++i)
        oneHot({ (size_t)i }) = 0.0;
    oneHot({ (size_t)labelByte }) = 1.0;
    file.close();
    return oneHot;
}
