#pragma once

#include "tensor.hpp"  // the revised tensor.hpp above
#include <cstdlib>     // for std::exit

template<typename ComponentType>
class Vector
{
public:
    Vector() = default;

    explicit Vector(size_t size);
    Vector(size_t size, const ComponentType& fillValue);
    Vector(const std::string& filename);

    [[nodiscard]] size_t size() const;

    // Element access
    const ComponentType& operator()(size_t idx) const;
    ComponentType&       operator()(size_t idx);

    // Direct access to underlying tensor
    Tensor<ComponentType>& tensor();

private:
    Tensor<ComponentType> tensor_;
};

template<typename ComponentType>
class Matrix
{
public:
    Matrix() = default;

    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, const ComponentType& fillValue);
    Matrix(const std::string& filename);

    [[nodiscard]] size_t rows() const;
    [[nodiscard]] size_t cols() const;

    // Element access
    const ComponentType& operator()(size_t row, size_t col) const;
    ComponentType&       operator()(size_t row, size_t col);

    Tensor<ComponentType>& tensor();

private:
    Tensor<ComponentType> tensor_;
};

//-----------------------------------------
// Implementations: Vector
//-----------------------------------------
template<typename ComponentType>
Vector<ComponentType>::Vector(size_t size)
    : tensor_({size})
{
}

template<typename ComponentType>
Vector<ComponentType>::Vector(size_t size, const ComponentType& fillValue)
    : tensor_({size}, fillValue)
{
}

template<typename ComponentType>
Vector<ComponentType>::Vector(const std::string& filename)
{
    tensor_ = readTensorFromFile<ComponentType>(filename);
    if (tensor_.rank() != 1)
    {
        std::cerr << "Error: loaded tensor is not rank-1.\n";
        std::exit(1);
    }
}

template<typename ComponentType>
size_t Vector<ComponentType>::size() const
{
    return tensor_.numElements();
}

template<typename ComponentType>
const ComponentType& Vector<ComponentType>::operator()(size_t idx) const
{
    return tensor_({idx});
}

template<typename ComponentType>
ComponentType& Vector<ComponentType>::operator()(size_t idx)
{
    return tensor_({idx});
}

template<typename ComponentType>
Tensor<ComponentType>& Vector<ComponentType>::tensor()
{
    return tensor_;
}

//-----------------------------------------
// Implementations: Matrix
//-----------------------------------------
template<typename ComponentType>
Matrix<ComponentType>::Matrix(size_t rows, size_t cols)
    : tensor_({rows, cols})
{
}

template<typename ComponentType>
Matrix<ComponentType>::Matrix(size_t rows, size_t cols, const ComponentType& fillValue)
    : tensor_({rows, cols}, fillValue)
{
}

template<typename ComponentType>
Matrix<ComponentType>::Matrix(const std::string& filename)
{
    tensor_ = readTensorFromFile<ComponentType>(filename);
    if (tensor_.rank() != 2)
    {
        std::cerr << "Error: loaded tensor is not rank-2.\n";
        std::exit(1);
    }
}

template<typename ComponentType>
size_t Matrix<ComponentType>::rows() const
{
    return tensor_.shape()[0];
}

template<typename ComponentType>
size_t Matrix<ComponentType>::cols() const
{
    return tensor_.shape()[1];
}

template<typename ComponentType>
const ComponentType& Matrix<ComponentType>::operator()(size_t row, size_t col) const
{
    return tensor_({row, col});
}

template<typename ComponentType>
ComponentType& Matrix<ComponentType>::operator()(size_t row, size_t col)
{
    return tensor_({row, col});
}

template<typename ComponentType>
Tensor<ComponentType>& Matrix<ComponentType>::tensor()
{
    return tensor_;
}

//-----------------------------------------
// MatVec multiplication
//-----------------------------------------
template<typename ComponentType>
Vector<ComponentType> matvec(const Matrix<ComponentType>& mat,
                             const Vector<ComponentType>& vec)
{
    if (mat.cols() != vec.size())
    {
        std::cerr << "Error: dimension mismatch in matvec.\n";
        std::exit(1);
    }

    Vector<ComponentType> out(mat.rows(), ComponentType(0));

    for (size_t row = 0; row < mat.rows(); row++)
    {
        for (size_t col = 0; col < mat.cols(); col++)
        {
            out(row) += mat(row, col) * vec(col);
        }
    }
    return out;
}
