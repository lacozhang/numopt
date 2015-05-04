/*
include widely used library, define available types
*/

#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::SparseVector<double, Eigen::ColMajor> SparseVector;
typedef Eigen::VectorXd DenseVector;
typedef Eigen::VectorXi ClsLabelVector;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> DataSamples;