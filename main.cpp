#include <iostream>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::seq;
using Eigen::last;
using Eigen::all;

VectorXd getHouseholder(VectorXd x){
	/*
		Returns householder vector for vector x
	*/
	VectorXd y, v;

	// compute target vector
	y = VectorXd::Constant(x.size(), 0);
	y(0) = x.norm();

	// compute householder
	v = x - y;
	return  v / v.norm();
}

tuple<MatrixXd, MatrixXd> qr(MatrixXd A){
	/*
		Compute QR factorization of matrix A and returns a tuple containing
		matrices Q and R
	*/
	int m = A.rows();
	int n = A.cols();

	// initialize matrices R and Q
	MatrixXd R = A;
	MatrixXd Q = MatrixXd::Identity(m, m);

	int last_iteration;
	if( m > n ) last_iteration = n;
	else last_iteration = n-1;

	//  compute factorization iteratibely
	for(int k = 0; k < last_iteration; k++){
		// compute the householder reflector and householder matrix H for the k-th column of matrix A
		VectorXd u = getHouseholder(R(seq(k, last), k));
		MatrixXd H = MatrixXd::Identity(m-k, m-k) - 2*( u * u.transpose() );

		// compute submatrix R and Q*H
		R.bottomRightCorner(m-k, n-k) = R.bottomRightCorner(m-k, n-k).eval() - 2*(u * u.transpose() * R.bottomRightCorner(m-k, n-k).eval());
		Q.bottomRightCorner(m, n-k) =  Q.bottomRightCorner(m, n - k).eval() * H;
	}

	return make_tuple(Q, R);
}

MatrixXd backSubstitution(MatrixXd A, VectorXd b){
	/*
		Compute the solution x of Ax = b using back substitution 
		assuming A upper triangular.
	*/
	int m = A.rows();
	int n = A.cols();

	VectorXd x(m);

	for(int i = m - 1; i >= 0; i--){
		double acc = b(i);
		for(int j = i + 1; j < n; j++){
			acc -= A(i, j)*x(j);
		}
		x(i) = acc / A(i,i);
	}
	return x;
}

VectorXd solveLinearSystem(MatrixXd A, VectorXd b){
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
	Q_0 = Q.bottomLeftCorner(m, n);
	R_0 = R.topLeftCorner(n, n);

	// solving R_0 x = Q_0^T b via back substitution
	VectorXd c = Q_0.transpose() * b;
	return backSubstitution(R, c);
}

int main(){
	MatrixXd A(3,3);
	A(0,0) = 1;
	A(0,1) = 2;
	A(0,2) = 3;
	A(1,0) = 0;
	A(1,1) = 1;
	A(1,2) = 4;
	A(2,0) = 5;
	A(2,1) = 6;
	A(2,2) = 0;

	VectorXd b(3);
	b(0) = 9;
	b(1) = 6;
	b(2) = 22;

	MatrixXd Q, R;
	tie(Q, R) = qr(A);

	cout << "Q*R" << endl << Q*R << endl;

	VectorXd x = solveLinearSystem(A, b);
	cout << "Solution x" << endl << x << endl;
}
