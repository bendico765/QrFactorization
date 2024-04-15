#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

VectorXd getHouseholder(VectorXd x);
std::tuple<MatrixXd, MatrixXd> qr(MatrixXd A);
MatrixXd backSubstitution(MatrixXd A, VectorXd b);
VectorXd solveLinearSystem(MatrixXd A, VectorXd b, int nHiddenNodes);
float sigmoid(float x);
MatrixXd initializeInputLayer(MatrixXd x, int nSamples, int nHiddenNodes, int nFeatures);
