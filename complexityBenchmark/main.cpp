#include <iostream>
#include <Eigen/Dense>
#include <armadillo>
#include <sstream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

const int M = 10000;
const int N = 100;

/**/
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
void save_csv(string name, MatrixXd matrix)
{
    ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
}

void vector_save_csv(string name, VectorXd vector)
{
    ofstream file(name.c_str());
    file << vector.format(CSVFormat);
}


int main(){
	MatrixXd A(M, N);
	VectorXd x(M);

	A = MatrixXd::Random(M, N);
	x = VectorXd::Random(N);

	VectorXd b = A*x;

	save_csv("MatrixA.csv", A);
	vector_save_csv("VectorB.csv", b);
	vector_save_csv("VectorX.csv", x);
}
