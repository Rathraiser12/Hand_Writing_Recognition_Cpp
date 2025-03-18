#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

//------------------------------
// Helpers
//------------------------------
inline constexpr size_t flatIdx(const std::vector<size_t>& shape,
                                const std::vector<size_t>& idx)
{
    assert(shape.size() == idx.size());
    const size_t rank = idx.size();

    if (rank == 0)
    {
        return 0;
    }
    else if (rank == 1)
    {
        return idx[0];
    }
    else if (rank == 2)
    {
        return idx[0] * shape[1] + idx[1];
    }
    else
    {
        // Fallback for rank > 2 (generic n-d)
        size_t result = 0;
        for (size_t i = 0; i < rank; i++)
        {
            size_t dimProduct = 1;
            for (size_t ii = i + 1; ii < rank; ii++)
            {
                dimProduct *= shape[ii];
            }
            result += idx[i] * dimProduct;
        }
        return result;
    }
}

inline size_t numTensorElements(const std::vector<size_t>& shape)
{
    size_t size = 1;
    for (auto d : shape)
    {
        size *= d;
    }
    return size;
}

template<typename T>
T stringToScalar(const std::string& str)
{
    std::stringstream s(str);
    T scalar;
    s >> scalar;
    return scalar;
}

//------------------------------
// Concept
//------------------------------
template<class T>
concept Arithmetic = std::is_arithmetic_v<T>;

//------------------------------
// Tensor Class
//------------------------------
template<Arithmetic ComponentType>
class Tensor
{
public:
    Tensor();  // rank=0 constructor
    Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, const ComponentType& fillValue);

    Tensor(const Tensor<ComponentType>& other) = default;            // copy-ctor
    Tensor(Tensor<ComponentType>&& other) noexcept;                  // move-ctor
    Tensor<ComponentType>& operator=(const Tensor<ComponentType>&) = default;  // copy-assign
    Tensor<ComponentType>& operator=(Tensor<ComponentType>&& other) noexcept;  // move-assign
    ~Tensor() = default;

    [[nodiscard]] size_t rank() const;
    [[nodiscard]] std::vector<size_t> shape() const;
    [[nodiscard]] size_t numElements() const;

    // Read-only element access
    const ComponentType& operator()(const std::vector<size_t>& idx) const;
    // Mutable element access
    ComponentType&       operator()(const std::vector<size_t>& idx);

private:
    std::vector<size_t>                              shape_;
    Eigen::Matrix<ComponentType, Eigen::Dynamic, 1>  data_;
};

//------------------------------
// Implementations
//------------------------------
template<Arithmetic ComponentType>
Tensor<ComponentType>::Tensor()
    : shape_(0)
    , data_(1)  // for rank=0, we have exactly one scalar
{
    data_(0) = ComponentType(0);
}

template<Arithmetic ComponentType>
Tensor<ComponentType>::Tensor(const std::vector<size_t>& shape)
    : shape_(shape)
    , data_(numTensorElements(shape))
{
    // Initialize all elements to 0
    data_.setZero();
}

template<Arithmetic ComponentType>
Tensor<ComponentType>::Tensor(const std::vector<size_t>& shape,
                              const ComponentType& fillValue)
    : shape_(shape)
    , data_(numTensorElements(shape))
{
    data_.setConstant(fillValue);
}

// Move constructor
template<Arithmetic ComponentType>
Tensor<ComponentType>::Tensor(Tensor<ComponentType>&& other) noexcept
    : shape_(std::move(other.shape_))
    , data_(std::move(other.data_))
{
    // After move, old object is left in a valid but unspecified state
    other.shape_.clear();
    other.data_.resize(1);
    other.data_(0) = ComponentType(0);
}

// Move assignment
template<Arithmetic ComponentType>
Tensor<ComponentType>&
Tensor<ComponentType>::operator=(Tensor<ComponentType>&& other) noexcept
{
    if (this != &other)
    {
        shape_ = std::move(other.shape_);
        data_  = std::move(other.data_);
        // Clean up the moved-from object
        other.shape_.clear();
        other.data_.resize(1);
        other.data_(0) = ComponentType(0);
    }
    return *this;
}

template<Arithmetic ComponentType>
size_t Tensor<ComponentType>::rank() const
{
    return shape_.size();
}

template<Arithmetic ComponentType>
std::vector<size_t> Tensor<ComponentType>::shape() const
{
    return shape_;
}

template<Arithmetic ComponentType>
size_t Tensor<ComponentType>::numElements() const
{
    return numTensorElements(shape_);
}

template<Arithmetic ComponentType>
const ComponentType&
Tensor<ComponentType>::operator()(const std::vector<size_t>& idx) const
{
    const size_t linearIndex = flatIdx(shape_, idx);
    return data_.coeff(linearIndex); // read-only
}

template<Arithmetic ComponentType>
ComponentType&
Tensor<ComponentType>::operator()(const std::vector<size_t>& idx)
{
    const size_t linearIndex = flatIdx(shape_, idx);
    return data_.coeffRef(linearIndex); // writable
}

//------------------------------
// Comparison (==)
//------------------------------
template<Arithmetic ComponentType>
bool operator==(const Tensor<ComponentType>& a,
                const Tensor<ComponentType>& b)
{
    if (a.shape() != b.shape())
        return false;

    // Compare all elements
    for (size_t i = 0; i < a.numElements(); i++)
    {
        if (a({i}) != b({i}))
            return false;
    }
    return true;
}

//------------------------------
// File IO
//------------------------------
//
// We'll handle rank=0,1,2 explicitly. If you need rank>2, you could either
// implement a generic n-d indexing loop or throw an error. We'll do the latter
// for brevity.

// Generic helper for reading a single line as a scalar
template<Arithmetic ComponentType>
ComponentType readScalarLine(std::ifstream& file)
{
    std::string line;
    if (!std::getline(file, line))
    {
        throw std::runtime_error("readTensorFromFile: not enough lines");
    }
    return stringToScalar<ComponentType>(line);
}

template<Arithmetic ComponentType>
Tensor<ComponentType> readTensorFromFile(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Could not open file: " << filename << "\n";
        std::exit(1);
    }

    // 1) Read rank
    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("readTensorFromFile: no rank line found");
    }
    size_t rankVal = stringToScalar<size_t>(line);

    // 2) Read shape
    std::vector<size_t> shp(rankVal);
    for (size_t i = 0; i < rankVal; i++)
    {
        if (!std::getline(file, line)) {
            throw std::runtime_error("readTensorFromFile: shape line missing");
        }
        shp[i] = stringToScalar<size_t>(line);
    }

    Tensor<ComponentType> tensor(shp);

    // 3) Read elements depending on rank
    if (rankVal == 0)
    {
        // single scalar
        tensor({}) = readScalarLine<ComponentType>(file);
    }
    else if (rankVal == 1)
    {
        size_t total = tensor.numElements();
        for (size_t i = 0; i < total; i++)
        {
            tensor({i}) = readScalarLine<ComponentType>(file);
        }
    }
    else if (rankVal == 2)
    {
        // read row-by-row
        size_t rows = shp[0];
        size_t cols = shp[1];
        for (size_t r = 0; r < rows; r++)
        {
            for (size_t c = 0; c < cols; c++)
            {
                tensor({r, c}) = readScalarLine<ComponentType>(file);
            }
        }
    }
    else
    {
        throw std::runtime_error("readTensorFromFile: rank>2 not supported in this snippet");
    }

    file.close();
    return tensor;
}

// Write the tensor to file. For rank=2, do row+col iteration
template<Arithmetic ComponentType>
void writeTensorToFile(const Tensor<ComponentType>& tensor,
                       const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Could not open file for writing: " << filename << "\n";
        std::exit(1);
    }

    // 1) Write rank
    file << tensor.rank() << "\n";
    // 2) Write shape
    for (auto d : tensor.shape())
    {
        file << d << "\n";
    }

    // 3) Write elements
    size_t rankVal = tensor.rank();
    if (rankVal == 0)
    {
        // single scalar
        file << tensor({}) << "\n";
    }
    else if (rankVal == 1)
    {
        size_t total = tensor.numElements();
        for (size_t i = 0; i < total; i++)
        {
            file << tensor({i}) << "\n";
        }
    }
    else if (rankVal == 2)
    {
        size_t rows = tensor.shape()[0];
        size_t cols = tensor.shape()[1];
        for (size_t r = 0; r < rows; r++)
        {
            for (size_t c = 0; c < cols; c++)
            {
                file << tensor({r, c}) << "\n";
            }
        }
    }
    else
    {
        throw std::runtime_error("writeTensorToFile: rank>2 not supported in this snippet");
    }

    file.close();
}
