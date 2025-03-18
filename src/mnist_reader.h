// mnist_reader.h

#pragma once

#include <string>
#include "tensor.hpp" // The provided tensor class

class MNISTReader {
public:
    // Existing image-reading method
    static Tensor<double> readImageData(const std::string& filename, int imageIndex);

    // NEW: Label-reading method (reads ONE label at labelIndex and one-hot encodes it)
    static Tensor<double> readLabelData(const std::string& filename, int labelIndex);

private:
    static int reverseInt(int i);
};
