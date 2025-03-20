#include "mnist_data_loader.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstring>

// Define the constructor.
MNISTDataLoader::MNISTDataLoader(const std::string &imageFile, const std::string &labelFile, size_t batchSize)
    : imageFilePath(imageFile), labelFilePath(labelFile), batchSize(batchSize),
      numImages(0), numRows(0), numCols(0), numLabels(0)
{
    // The constructor initializes file paths, batch size, and numeric properties to zero.
}
// Define reverseInt as a static member function.
int MNISTDataLoader::reverseInt(int i) {
    unsigned char c1 = i & 0xFF, c2 = (i >> 8) & 0xFF, c3 = (i >> 16) & 0xFF, c4 = (i >> 24) & 0xFF;
    return (int(c1) << 24) + (int(c2) << 16) + (int(c3) << 8) + c4;
}

void MNISTDataLoader::loadDataset() {
    loadImages();
    loadLabels();
}

void MNISTDataLoader::loadImages() {
    std::ifstream in(imageFilePath, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open image file: " + imageFilePath);

    int magic = 0;
    in.read(reinterpret_cast<char*>(&magic), 4);
    magic = reverseInt(magic);
    if (magic != 2051)
        throw std::runtime_error("Invalid MNIST image file (magic != 2051)");

    in.read(reinterpret_cast<char*>(&numImages), 4);
    numImages = reverseInt(numImages);
    in.read(reinterpret_cast<char*>(&numRows), 4);
    numRows = reverseInt(numRows);
    in.read(reinterpret_cast<char*>(&numCols), 4);
    numCols = reverseInt(numCols);

    std::cout << "Image File: " << imageFilePath << "\n"
              << "Number of Images: " << numImages
              << ", Rows: " << numRows << ", Cols: " << numCols << "\n";

    size_t imgSize = numRows * numCols;
    Eigen::MatrixXd imageMatrix(batchSize, imgSize);
    size_t fillCount = 0;
    std::vector<unsigned char> imgBin(imgSize);

    for (size_t i = 0; i < numImages; ++i) {
        in.read(reinterpret_cast<char*>(imgBin.data()), imgSize);
        for (size_t j = 0; j < imgSize; ++j)
            imageMatrix(fillCount, j) = static_cast<double>(imgBin[j]) / 255.0;
        fillCount++;

        if (fillCount == batchSize || i == numImages - 1) {
            imageBatches.push_back(imageMatrix.topRows(fillCount));
            fillCount = 0;
        }
    }
    in.close();
}

void MNISTDataLoader::loadLabels() {
    std::ifstream in(labelFilePath, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open label file: " + labelFilePath);

    int magic = 0;
    in.read(reinterpret_cast<char*>(&magic), 4);
    magic = reverseInt(magic);
    if (magic != 2049)
        throw std::runtime_error("Invalid MNIST label file (magic != 2049)");

    in.read(reinterpret_cast<char*>(&numLabels), 4);
    numLabels = reverseInt(numLabels);

    std::cout << "Label File: " << labelFilePath << "\n"
              << "Number of Labels: " << numLabels << "\n";

    Eigen::MatrixXd labelMatrix(batchSize, 10);
    labelMatrix.setZero();
    size_t fillCount = 0;

    for (size_t i = 0; i < numLabels; ++i) {
        unsigned char label = 0;
        in.read(reinterpret_cast<char*>(&label), 1);
        labelMatrix(fillCount, label) = 1.0;
        fillCount++;

        if (fillCount == batchSize || i == numLabels - 1) {
            labelBatches.push_back(labelMatrix.topRows(fillCount));
            labelMatrix.setZero();
            fillCount = 0;
        }
    }
    in.close();
}

Eigen::MatrixXd MNISTDataLoader::getImageBatch(size_t index) const {
    if (index >= imageBatches.size())
        throw std::runtime_error("Image batch index out of range");
    return imageBatches[index];
}

Eigen::MatrixXd MNISTDataLoader::getLabelBatch(size_t index) const {
    if (index >= labelBatches.size())
        throw std::runtime_error("Label batch index out of range");
    return labelBatches[index];
}

size_t MNISTDataLoader::getNumBatches() const {
    return imageBatches.size(); // Assumes images and labels have the same number of batches.
}

// --- Static Methods for Single Sample Reading ---

Eigen::MatrixXd MNISTDataLoader::readSingleImage(const std::string &filename, int imageIndex) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open image file: " + filename);
    
    int magic = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    magic = reverseInt(magic);
    if (magic != 2051)
        throw std::runtime_error("Invalid MNIST image file (magic != 2051)");

    int numImages = 0, numRows = 0, numCols = 0;
    file.read(reinterpret_cast<char*>(&numImages), 4); numImages = reverseInt(numImages);
    file.read(reinterpret_cast<char*>(&numRows), 4); numRows = reverseInt(numRows);
    file.read(reinterpret_cast<char*>(&numCols), 4); numCols = reverseInt(numCols);

    if (imageIndex < 0 || imageIndex >= numImages)
        throw std::runtime_error("Image index out of range");

    size_t imgSize = static_cast<size_t>(numRows) * numCols;
    file.seekg(16 + imageIndex * imgSize, std::ios::beg);
    Eigen::MatrixXd imageMat(numRows, numCols);
    for (int r = 0; r < numRows; ++r) {
        for (int c = 0; c < numCols; ++c) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            imageMat(r, c) = static_cast<double>(pixel) / 255.0;
        }
    }
    file.close();
    return imageMat;
}

Eigen::MatrixXd MNISTDataLoader::readSingleLabel(const std::string &filename, int labelIndex) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open label file: " + filename);
    
    int magic = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    magic = reverseInt(magic);
    if (magic != 2049)
        throw std::runtime_error("Invalid MNIST label file (magic != 2049)");

    int numLabels = 0;
    file.read(reinterpret_cast<char*>(&numLabels), 4); numLabels = reverseInt(numLabels);
    if (labelIndex < 0 || labelIndex >= numLabels)
        throw std::runtime_error("Label index out of range");

    file.seekg(8 + labelIndex, std::ios::beg);
    unsigned char labelByte = 0;
    file.read(reinterpret_cast<char*>(&labelByte), 1);

    Eigen::MatrixXd labelMat(10, 1);
    labelMat.setZero();
    labelMat(static_cast<int>(labelByte), 0) = 1.0;
    file.close();
    return labelMat;
}
