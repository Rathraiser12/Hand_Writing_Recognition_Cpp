#pragma once
#include <Eigen/Dense>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <type_traits>

// Compute flat (linear) index from multi-dimensional indices.
inline constexpr size_t linearIndex(const std::vector<size_t>& shape, const std::vector<size_t>& idx) {
    assert(shape.size() == idx.size());
    size_t rank = idx.size();
    if (rank == 0) return 0;
    if (rank == 1) return idx[0];
    if (rank == 2) return idx[0] * shape[1] + idx[1];
    size_t result = 0;
    for (size_t i = 0; i < rank; ++i) {
        size_t prod = 1;
        for (size_t j = i + 1; j < rank; ++j)
            prod *= shape[j];
        result += idx[i] * prod;
    }
    return result;
}

inline size_t numElements(const std::vector<size_t>& shape) {
    size_t prod = 1;
    for (auto d : shape)
        prod *= d;
    return prod;
}

template<typename T>
T stringToScalar(const std::string &str) {
    std::istringstream iss(str);
    T val;
    iss >> val;
    return val;
}

template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template<Arithmetic T>
class Tensor {
public:
    Tensor() : shape_{}, data_(1) { data_(0) = T(0); }
    Tensor(const std::vector<size_t>& s) : shape_(s), data_(::numElements(s)) { data_.setZero(); }
    Tensor(const std::vector<size_t>& s, const T &fillVal) : shape_(s), data_(::numElements(s)) { data_.setConstant(fillVal); }

    Tensor(const Tensor&) = default;
    Tensor(Tensor&& other) noexcept
        : shape_(std::move(other.shape_)), data_(std::move(other.data_)) {
        other.shape_.clear();
        other.data_.resize(1);
        other.data_(0) = T(0);
    }
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            shape_ = std::move(other.shape_);
            data_  = std::move(other.data_);
            other.shape_.clear();
            other.data_.resize(1);
            other.data_(0) = T(0);
        }
        return *this;
    }

    size_t rank() const { return shape_.size(); }
    std::vector<size_t> shape() const { return shape_; }
    size_t numElements() const { return ::numElements(shape_); }

    const T& operator()(const std::vector<size_t>& idx) const { return data_.coeff(linearIndex(shape_, idx)); }
    T& operator()(const std::vector<size_t>& idx) { return data_.coeffRef(linearIndex(shape_, idx)); }

private:
    std::vector<size_t> shape_;
    Eigen::Matrix<T, Eigen::Dynamic, 1> data_;
};

template<Arithmetic T>
bool operator==(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape() != b.shape())
        return false;
    for (size_t i = 0; i < a.numElements(); ++i)
        if (a({ i }) != b({ i }))
            return false;
    return true;
}

template<Arithmetic T>
T readScalarLine(std::ifstream &file) {
    std::string line;
    if (!std::getline(file, line))
        throw std::runtime_error("Not enough lines in file");
    return stringToScalar<T>(line);
}

template<Arithmetic T>
Tensor<T> readTensorFromFile(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << "\n";
        std::exit(1);
    }
    std::string line;
    if (!std::getline(file, line))
        throw std::runtime_error("No rank line found");
    size_t rnk = stringToScalar<size_t>(line);
    std::vector<size_t> shp(rnk);
    for (size_t i = 0; i < rnk; ++i) {
        if (!std::getline(file, line))
            throw std::runtime_error("Shape line missing");
        shp[i] = stringToScalar<size_t>(line);
    }
    Tensor<T> tensor(shp);
    if (rnk == 0)
        tensor({}) = readScalarLine<T>(file);
    else if (rnk == 1)
        for (size_t i = 0; i < tensor.numElements(); ++i)
            tensor({ i }) = readScalarLine<T>(file);
    else if (rnk == 2) {
        size_t rows = shp[0], cols = shp[1];
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c)
                tensor({ r, c }) = readScalarLine<T>(file);
    } else {
        throw std::runtime_error("Rank >2 not supported");
    }
    file.close();
    return tensor;
}

template<Arithmetic T>
void writeTensorToFile(const Tensor<T> &tensor, const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file for writing: " << filename << "\n";
        std::exit(1);
    }
    file << tensor.rank() << "\n";
    for (auto d : tensor.shape())
        file << d << "\n";
    size_t rnk = tensor.rank();
    if (rnk == 0)
        file << tensor({}) << "\n";
    else if (rnk == 1)
        for (size_t i = 0; i < tensor.numElements(); ++i)
            file << tensor({ i }) << "\n";
    else if (rnk == 2) {
        size_t rows = tensor.shape()[0], cols = tensor.shape()[1];
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c)
                file << tensor({ r, c }) << "\n";
    } else {
        throw std::runtime_error("Rank >2 not supported");
    }
    file.close();
}
