#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <Eigen/Dense>

class DatasetLabels {
public:
    explicit DatasetLabels(size_t batchSize)
        : batch_size(batchSize), num_labels(0) {}

    void readLabelData(const std::string &filepath);
    void writeLabelToFile(const std::string &filepath, size_t index);
    Eigen::MatrixXd getBatch(size_t index) const { return batches.at(index); }
    size_t getBatchCount() const { return batches.size(); }

private:
    size_t batch_size, num_labels;
    std::vector<Eigen::MatrixXd> batches;
};

void DatasetLabels::readLabelData(const std::string &filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Unable to open file " << filepath << "\n";
        return;
    }
    char buffer[4];
    in.read(buffer, 4); std::reverse(buffer, buffer + 4);
    int magic = 0; std::memcpy(&magic, buffer, sizeof(int));

    in.read(buffer, 4); std::reverse(buffer, buffer + 4);
    std::memcpy(&num_labels, buffer, sizeof(int));

    Eigen::MatrixXd labelMatrix(batch_size, 10);
    labelMatrix.setZero();
    size_t fillCount = 0;
    for (size_t i = 0; i < num_labels; ++i) {
        uint8_t byte = 0;
        in.read(reinterpret_cast<char*>(&byte), 1);
        labelMatrix(fillCount, byte) = 1.0;
        fillCount++;
        if (fillCount == batch_size || i == num_labels - 1) {
            batches.push_back(labelMatrix.topRows(fillCount));
            labelMatrix.setZero();
            fillCount = 0;
        }
    }
    in.close();
}

void DatasetLabels::writeLabelToFile(const std::string &filepath, size_t index) {
    size_t batch_no = index / batch_size;
    size_t row_in_batch = index % batch_size;
    if (batch_no >= batches.size() || row_in_batch >= batches[batch_no].rows()) {
        std::cerr << "Index out of range.\n";
        return;
    }
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "Unable to open file " << filepath << "\n";
        return;
    }
    out << 1 << "\n" << 10 << "\n";
    for (int i = 0; i < 10; ++i)
        out << batches[batch_no](row_in_batch, i) << "\n";
    out.close();
}
