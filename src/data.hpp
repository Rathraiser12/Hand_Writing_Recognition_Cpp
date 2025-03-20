#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <Eigen/Dense>

class DataSetImages {
public:
    explicit DataSetImages(size_t batchSize)
        : batch_size(batchSize), num_images(0), num_rows(0), num_cols(0) {}

    void readImageData(const std::string &filepath);
    void writeImageToFile(const std::string &filepath, size_t index);
    Eigen::MatrixXd getBatch(size_t index) const { return batches.at(index); }
    size_t getBatchCount() const { return batches.size(); }

private:
    size_t batch_size, num_images, num_rows, num_cols;
    std::vector<Eigen::MatrixXd> batches;
};

void DataSetImages::readImageData(const std::string &filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Unable to open file: " << filepath << "\n";
        return;
    }
    char buffer[4];
    in.read(buffer, 4); std::reverse(buffer, buffer + 4);
    std::memcpy(&num_images, buffer, sizeof(int));

    in.read(buffer, 4); std::reverse(buffer, buffer + 4);
    std::memcpy(&num_images, buffer, sizeof(int));

    in.read(buffer, 4); std::reverse(buffer, buffer + 4);
    std::memcpy(&num_rows, buffer, sizeof(int));

    in.read(buffer, 4); std::reverse(buffer, buffer + 4);
    std::memcpy(&num_cols, buffer, sizeof(int));

    size_t imgSize = num_rows * num_cols;
    std::vector<unsigned char> imgBin(imgSize);
    Eigen::MatrixXd imageMatrix(batch_size, imgSize);
    size_t fillCount = 0;
    for (size_t i = 0; i < num_images; ++i) {
        in.read(reinterpret_cast<char*>(imgBin.data()), imgSize);
        for (size_t j = 0; j < imgSize; ++j)
            imageMatrix(fillCount, j) = static_cast<double>(imgBin[j]) / 255.0;
        fillCount++;
        if (fillCount == batch_size || i == num_images - 1) {
            batches.push_back(imageMatrix.topRows(fillCount));
            fillCount = 0;
        }
    }
    in.close();
}

void DataSetImages::writeImageToFile(const std::string &filepath, size_t index) {
    size_t batch_no = index / batch_size;
    size_t row_in_batch = index % batch_size;
    if (batch_no >= batches.size() || row_in_batch >= batches[batch_no].rows()) {
        std::cerr << "Index out of range.\n";
        return;
    }
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "Unable to open file for writing: " << filepath << "\n";
        return;
    }
    out << 2 << "\n" << num_rows << "\n" << num_cols << "\n";
    size_t imgSize = num_rows * num_cols;
    for (size_t i = 0; i < imgSize; ++i)
        out << batches[batch_no](row_in_batch, i) << "\n";
    out.close();
}
