#include <iostream>
#include <Eigen/Dense>
#include <armadillo>
#include <cmath>
#include <sstream>
#include <chrono>
#include "extremeLearning.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::seq;
using Eigen::last;
using Eigen::all;
using chrono::high_resolution_clock;
using chrono::duration;

/*
	Loads data from csv file

	@param the path of the csv file to load
*/
template <typename M>
M load_csv_arma (const std::string & path) {
    	arma::mat X;
    	X.load(path, arma::csv_ascii);
    	return Eigen::Map<const M>(X.memptr(), X.n_rows, X.n_cols);
}

/**/
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
void save_csv(string name, MatrixXd matrix)
{
    ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
}

void complexityBenchmark(){
	MatrixXd A;
	int m_values[] = {100, 200, 400, 800};
	int n_values[] = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};

	A = load_csv_arma<MatrixXd>("complexityBenchmark/MatrixA.csv");

	for(int i = 0; i < size(m_values); i ++){
		int m = m_values[i];
		MatrixXd measurements(size(n_values), 2);
		for(int j = 0; j < size(n_values); j++){
			MatrixXd Q,R;
			int n = n_values[j];

			MatrixXd submatrix(m, n);
			for(int r = 0; r < m; r++){
				for(int c = 0; c < n; c++){
					submatrix(r, c) = A(r, c);
				}
			}

			auto t1 = high_resolution_clock::now(); // begin measurement
			tie(Q,R) = qr(submatrix);
			auto t2 = high_resolution_clock::now(); // end measurement

			measurements(j, 0) = n;
			measurements(j, 1) = duration<double, milli>(t2 - t1).count();
		}
		ostringstream filepath;
		filepath << m << " rows.csv";
		save_csv(filepath.str(), measurements);
	}
}

int main(){
	complexityBenchmark();

	/*
	int nHiddenNodes;
	int nFeatures;
	int nSamples;
	int const N_LABELS = 1; // assuming single label problem
	int m,n;

	// read the dataset
	MatrixXd A = load_csv_arma<MatrixXd>("dataset.csv");

	m = A.rows(); // total numer of samples
	n = A.cols();

	nSamples = m;
	nFeatures = n - N_LABELS;
	nHiddenNodes = m -1;

	MatrixXd x = A.leftCols(nFeatures); // x features
	VectorXd y = A.rightCols(N_LABELS); // y labels

	cout << "Matrix x" << endl << x << endl;

	MatrixXd H = initializeInputLayer(x, nSamples, nHiddenNodes, nFeatures);

	cout << "Matrix H" << endl << H << endl;
	//cout << "Vector y" << endl << y << endl;

	VectorXd beta = solveLinearSystem(H, y);
	//cout << "Vector beta" << endl << beta << endl;
	//cout << "Predicted values" << endl << H * beta << endl;
	*/
}
