#include <iostream>
#include <Eigen/Dense>
#include <armadillo>
#include <cmath>
#include <sstream>
#include <chrono>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::seq;
using Eigen::last;
using Eigen::all;
using chrono::high_resolution_clock;
using chrono::duration;

/*
	Compute the householder vector

	@param x: the vector whose householder vector to calculate
	@return the householder vector for x
*/
VectorXd getHouseholder(VectorXd x){
	VectorXd y, v;

	// compute target vector
	y = VectorXd::Constant(x.size(), 0);
	y(0) = x.norm();

	// compute householder
	v = x - y;
	return  v / v.norm();
}

/*
	Compute the QR factorization of a matrix A = QR using householder reflectors

	@param A: the matrix to factorize
	@return a tuple containing the Q and R matrices
*/
tuple<MatrixXd, MatrixXd> qr(MatrixXd A){
	int m = A.rows();
	int n = A.cols();

	// initialize matrices R and Q
	MatrixXd R = A;
	MatrixXd Q = MatrixXd::Identity(m, m);

	int last_iteration;
	if( m != n ) last_iteration = n;
	else last_iteration = n-1;

	//  compute factorization iteratibely
	for(int k = 0; k < last_iteration; k++){
		// compute the householder reflector and householder matrix H for the k-th column of matrix A
		VectorXd u = getHouseholder(R(seq(k, last), k));
		MatrixXd H = MatrixXd::Identity(m-k, m-k) - 2*( u * u.transpose() );

		// compute submatrix R and Q*H
		R.bottomRightCorner(m-k, n-k) = R.bottomRightCorner(m-k, n-k).eval() - 2 * u * (u.transpose() * R.bottomRightCorner(m-k, n-k).eval());
		Q.bottomRightCorner(m, m-k) =  Q.bottomRightCorner(m, m - k).eval() * H;
	}

	return make_tuple(Q, R);
}


/*
	Compute the solution of the linear system Ax = b using backsubstitution.

	@param A: an upper triangular matrix
	@param b: a vector of targers
	@return the vector x that solves Ax = b
*/
MatrixXd backSubstitution(MatrixXd A, VectorXd b){
	int m = A.rows(); // rows of A
	int n = A.cols(); // cols of A

	VectorXd x(m); // solution vector

	// starting from the bottom row, iterate and solve
	for(int i = m - 1; i >= 0; i--){
		double acc = b(i);

		for(int j = i + 1; j < n; j++){
			acc -= A(i, j)*x(j);
		}
		x(i) = acc / A(i,i);
	}
	return x;
}

VectorXd solveLinearSystem(MatrixXd A, VectorXd b, int nHiddenNodes){
	/*
		Given matrices A and b solves the linear system Ax = b and returns
		the solution x. The solution is computed by using QR factorization
		(via householder reflectors) and back substitution.
	*/
	MatrixXd Q, R, Q_0, R_0;
	int m = A.rows();
	int n = A.cols();

	// factorize A using QR
	tie(Q, R) = qr(A);

	// keeping thinner matrices Q_0 and R_0
	Q_0 = Q.bottomLeftCorner(m, nHiddenNodes);
	R_0 = R.topLeftCorner(nHiddenNodes, nHiddenNodes);

	// solving R_0 x = Q_0^T b via back substitution
	VectorXd c = Q_0.transpose() * b;

	return backSubstitution(R_0, c);
}

double sigmoid(float x){ return 1 / (1 + exp(-x)); }

/*
	Initialize the hidden layer output matrix
*/
MatrixXd initializeInputLayer(MatrixXd x, int nSamples, int nHiddenNodes, int nFeatures){
	MatrixXd w = MatrixXd::Random(nHiddenNodes, nFeatures);
	VectorXd bias = VectorXd::Random(nHiddenNodes);

	MatrixXd H(nSamples, nHiddenNodes);
        for(int i = 0; i < nSamples; i++){ // iterate over samples
                for(int j = 0; j < nHiddenNodes; j++){ // iterate over hidden nodes
                        H(i, j) = sigmoid( w(j, seq(0, last))*x(i, seq(0, last)).transpose() + bias(j) );
                }
        }
	return H;
}

