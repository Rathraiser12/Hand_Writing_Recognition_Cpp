#pragma once
#include <string>
#include "tensor.hpp"

class MNISTReader {
public:
    static Tensor<double> readImageData(const std::string &filename, int imageIndex);
    static Tensor<double> readLabelData(const std::string &filename, int labelIndex);
private:
    static int reverseInt(int i);
};
