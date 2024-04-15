#include <iostream>
#include <Eigen/Dense>
#include <armadillo>
#include <cmath>
#include <sstream>
#include <chrono>
#include "extremeLearning.h"

#ifndef LOG
	#define LOG false
#endif

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::seq;
using Eigen::last;
using Eigen::all;
using Eigen::CompleteOrthogonalDecomposition;
using chrono::high_resolution_clock;
using chrono::duration;

/*
	Loads data from csv file

	@param the path of the csv file to load
	@return the matrix object containing the csv file data
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

void complexityRowsBenchmark(){
	MatrixXd A = load_csv_arma<MatrixXd>("complexityBenchmark/MatrixA.csv");
	int m_values[] = {100}; // values for m
	int n_values[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100}; // values for n
	int n_measurements = 1;

	if( LOG ) cout << "Running complexity rows benchmark" << endl;
	for(int i = 0; i < size(m_values); i ++){
		int m = m_values[i];

		// create a matrix containing the elapsed time for each simulation
		MatrixXd measurements(size(n_values), 2);
		for(int j = 0; j < size(n_values); j++){
			int n = n_values[j];
			MatrixXd submatrix(m, n);
			MatrixXd Q,R;

			//
			for(int r = 0; r < m; r++){
				for(int c = 0; c < n; c++){
					submatrix(r, c) = A(r, c);
				}
			}

			// measure the execution time of QR factorization
			if( LOG ) cout << "Running QR for matrix " << m << "x" << n << endl;
			double mean_exec_time = 0;

			for(int l = 0; l < n_measurements; l++){
				if( LOG ) cout << "Taking measurement "<< l+1 <<"/"<<n_measurements << endl;
				auto t1 = high_resolution_clock::now(); // begin measurement
				tie(Q,R) = qr(submatrix);
				auto t2 = high_resolution_clock::now(); // end measurement

				mean_exec_time += duration<double, milli>(t2 - t1).count();
			}
			// save the results into the table
			measurements(j, 0) = n;
			measurements(j, 1) = mean_exec_time / n_measurements;
		}
		// export the results onto a csv file
		ostringstream filepath;
		filepath << m << " rows.csv";
		save_csv(filepath.str(), measurements);
	}
}

void complexityColumnsBenchmark(){
        MatrixXd A = load_csv_arma<MatrixXd>("complexityBenchmark/MatrixA.csv");
        int m_values[] = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}; // values for m
        int n_values[] = {10}; // values for n
        int n_measurements = 10;

        if( LOG ) cout << "Running complexity columns benchmark" << endl;
        for(int i = 0; i < size(n_values); i ++){
                int n = n_values[i];

                // create a matrix containing the elapsed time for each simulation
                MatrixXd measurements(size(m_values), 2);
                for(int j = 0; j < size(m_values); j++){
                        int m = m_values[j];
                        MatrixXd submatrix(m, n);
                        MatrixXd Q,R;

                        //
                        for(int r = 0; r < m; r++){
                                for(int c = 0; c < n; c++){
                                        submatrix(r, c) = A(r, c);
                                }
                        }

                        // measure the execution time of QR factorization
                        if( LOG ) cout << "Running QR for matrix " << m << "x" << n << endl;
                        double mean_exec_time = 0;
                        for(int l = 0; l < n_measurements; l++){
                                if( LOG ) cout << "Taking measurement "<< l+1 <<"/"<<n_measurements << endl;
                                auto t1 = high_resolution_clock::now(); // begin measurement
                                tie(Q,R) = qr(submatrix);
                                auto t2 = high_resolution_clock::now(); // end measurement

                                mean_exec_time += duration<double, milli>(t2 - t1).count();
                        }
                        // save the results into the table
                        measurements(j, 0) = m;
                        measurements(j, 1) = mean_exec_time / n_measurements;
                }
                // export the results onto a csv file
                ostringstream filepath;
                filepath << n << " columns.csv";
                save_csv(filepath.str(), measurements);
        }
}

void firstResidualTest(MatrixXd A){
	//MatrixXd A = load_csv_arma<MatrixXd>("complexityBenchmark/MatrixA.csv");
	//int m_values = {100, 1000, 10000};
	//int n_values = {10, 100, 100};
	//int m_values[] = {10000};
	//int n_values[] = {100};

	//for(int i = 0; i < size(m_values); i++){
		MatrixXd Q, R;
		//int m = m_values[i];
		//int n = n_values[i];
		/*
		MatrixXd submatrix(m, n);

		for(int r = 0; r < m; r++){
			for(int c = 0; c < n; c++){
				submatrix(r, c) = A(r, c);
			}
		}
		*/
		//tie(Q, R) = qr(submatrix);
		tie(Q, R) = qr(A);

		//float residual = (Q*R - submatrix).norm() / submatrix.norm();
		float residual = (Q*R - A).norm() / A.norm();
		cout << "|| QR - A || / ||A|| : " << residual << endl;
	//}
}

void secondResidualTest(MatrixXd submatrix, VectorXd subvector){
	int m = submatrix.rows();
	int n = submatrix.cols();
	VectorXd x_tilde = solveLinearSystem(submatrix, subvector, n);

	double r_tilde_norm = (submatrix * x_tilde - subvector).norm();
        cout << "Residual r_tilde = A x_tilde - b for matrix : " << r_tilde_norm << endl;

	CompleteOrthogonalDecomposition<MatrixXd> cqr(submatrix);
	MatrixXd pinv = cqr.pseudoInverse();
	double condition_number_matrix = submatrix.norm() * pinv.norm();
	cout << "Condition number of matrix: " << condition_number_matrix << endl;

	double bound =  condition_number_matrix * r_tilde_norm / subvector.norm();
	cout << "Bound K(A) * ( ||r_tilde|| / ||b|| ) : "  << bound << endl;
}

int main(){
	//complexityRowsBenchmark();
	//complexityColumnsBenchmark();
	MatrixXd A = load_csv_arma<MatrixXd>("complexityBenchmark/MatrixA.csv");
	VectorXd b = load_csv_arma<VectorXd>("complexityBenchmark/VectorB.csv");
	int N_RESIDUAL_TESTS = 4;
	int m_values[] = { 100, 100, 1000, 1000 };
	int n_values[] = { 10, 100, 100, 1000 };

	for(int i = 0; i < N_RESIDUAL_TESTS; i++){
		int m = m_values[i];
		int n = n_values[i];

		MatrixXd submatrix(m, n);
		VectorXd subvector(m);
		for(int r = 0; r < m; r++){
			subvector(r) = b(r);
                	for(int c = 0; c < n; c++){
                        	submatrix(r, c) = A(r, c);
                	}
        	}
		cout << "RUNNING RESIDUALS TESTS FOR MATRIX " << m << "x" << n << endl;
		firstResidualTest(submatrix);
		secondResidualTest(submatrix, subvector);
		cout << "=====================================================" << endl;
	}
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
	nHiddenNodes = m;

	MatrixXd x = A.leftCols(nFeatures); // x features
	VectorXd y = A.rightCols(N_LABELS); // y labels

	cout << "Matrix x" << endl << x << endl;

	MatrixXd H = initializeInputLayer(x, nSamples, nHiddenNodes, nFeatures);

	cout << "Matrix H" << endl << H << endl;
	cout << "Vector y" << endl << y << endl;

	VectorXd beta = solveLinearSystem(H, y, nHiddenNodes);
	cout << "Vector beta" << endl << beta << endl;
	cout << "Predicted values" << endl << H * beta << endl;
	*/
}
