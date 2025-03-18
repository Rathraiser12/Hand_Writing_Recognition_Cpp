#pragma once

#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <Eigen/Dense>

class DatasetLabels
{
private:
    size_t batch_size_;
    size_t number_of_labels_;

    // Each batch is a matrix of shape [batch_size_ x 10] for one-hot
    // The last batch might have fewer rows.
    std::vector<Eigen::MatrixXd> batches_;

public:
    explicit DatasetLabels(size_t batch_size);
    ~DatasetLabels();

    void readLabelData(const std::string &filepath);
    void writeLabelToFile(const std::string &filepath, size_t index);

    // Return the i-th batch
    Eigen::MatrixXd getBatch(size_t index);
    size_t getNumBatches() const { return batches_.size(); }
};

DatasetLabels::DatasetLabels(size_t batch_size)
    : batch_size_(batch_size),
      number_of_labels_(0)
{}

DatasetLabels::~DatasetLabels() {}

Eigen::MatrixXd DatasetLabels::getBatch(size_t index)
{
    return batches_[index];
}

void DatasetLabels::readLabelData(const std::string &input_filepath)
{
    std::ifstream input_file(input_filepath, std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Unable to open file " << input_filepath << std::endl;
        return;
    }

    char bin_data[4];
    // Magic number
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    int magic_number = 0;
    std::memcpy(&magic_number, bin_data, sizeof(int));

    // number of labels
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    int number_of_labels = 0;
    std::memcpy(&number_of_labels, bin_data, sizeof(int));
    number_of_labels_ = number_of_labels;

    // We'll build each batch as a [batch_size_ x 10] matrix
    Eigen::MatrixXd label_matrix(batch_size_, 10);
    label_matrix.setZero();

    size_t batchFillCount = 0;
    for(size_t i = 0; i < number_of_labels_; i++)
    {
        uint8_t byte = 0;
        input_file.read(reinterpret_cast<char *>(&byte), 1);
        int label = static_cast<int>(byte);

        // One-hot: set label_matrix(row, label_index)=1
        label_matrix(batchFillCount, label) = 1.0;
        batchFillCount++;

        // If batch is full or we are at the last label:
        if(batchFillCount == batch_size_ || (i == number_of_labels_ - 1))
        {
            // If partial, push only top 'batchFillCount' rows
            size_t validRows = batchFillCount;
            batches_.push_back(label_matrix.topRows(validRows));

            // Reset
            label_matrix.setZero();
            batchFillCount = 0;
        }
    }

    input_file.close();
}

void DatasetLabels::writeLabelToFile(const std::string &output_filepath, size_t index)
{
    // index is absolute: figure out which batch, which row
    size_t batch_no = index / batch_size_;
    size_t row_in_batch = index % batch_size_;

    if (batch_no >= batches_.size()) {
        std::cerr << "Index out of range: no such batch for label." << std::endl;
        return;
    }
    if (row_in_batch >= batches_[batch_no].rows()) {
        std::cerr << "Index out of range: row beyond last batch size." << std::endl;
        return;
    }

    std::ofstream output_file(output_filepath);
    if(!output_file.is_open()) {
        std::cerr << "Unable to open file " << output_filepath << std::endl;
        return;
    }

    // rank=1
    output_file << 1 << "\n";
    // shape=10
    output_file << 10 << "\n";

    for(int i = 0; i < 10; i++)
    {
        double val = batches_[batch_no](row_in_batch, i);
        output_file << val << "\n";
    }
    output_file.close();
}
