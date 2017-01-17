/*
include widely used library, define available types
*/

#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::SparseVector<float, Eigen::ColMajor> SparseVector;
typedef Eigen::VectorXf DenseVector;
typedef Eigen::VectorXi LabelVector;
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> DataSamples;