#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>

class MNISTDataLoader {
public:
    // Existing constructor and batch loading methods…
    MNISTDataLoader(const std::string &imageFile, const std::string &labelFile, size_t batchSize);

    void loadDataset();
    // Batch getters...
    Eigen::MatrixXd getImageBatch(size_t index) const;
    Eigen::MatrixXd getLabelBatch(size_t index) const;
    size_t getNumBatches() const;

    // --- NEW STATIC METHODS FOR SINGLE SAMPLE READING ---
    static Eigen::MatrixXd readSingleImage(const std::string &filename, int imageIndex);
    static Eigen::MatrixXd readSingleLabel(const std::string &filename, int labelIndex);

private:
    std::string imageFilePath;
    std::string labelFilePath;
    size_t batchSize;

    size_t numImages, numRows, numCols;
    size_t numLabels;

    std::vector<Eigen::MatrixXd> imageBatches;
    std::vector<Eigen::MatrixXd> labelBatches;

    // Make reverseInt static so it can be used in static methods.
    static int reverseInt(int i);

    void loadImages();
    void loadLabels();
};
