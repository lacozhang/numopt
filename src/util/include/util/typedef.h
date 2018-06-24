/*
include widely used library, define available types
*/
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::SparseVector<float, Eigen::ColMajor> SparseVector;
typedef Eigen::VectorXf DenseVector;
typedef Eigen::VectorXi LabelVector;
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> DataSamples;
typedef Eigen::MatrixXd DenseMatrix;

// For nn lib
typedef Eigen::SparseVector<double, Eigen::ColMajor> RealSparseVector;
typedef Eigen::VectorXd RealVector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign | Eigen::RowMajor> RowRealMatrix;
