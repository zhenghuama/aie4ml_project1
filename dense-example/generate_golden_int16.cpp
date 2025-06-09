/*
*	How to run:
* 	g++ -o generate_golden_int16 generate_golden_int16.cpp
*
*/

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "aie/kernels/include.h"

int main(){


	auto matA = new int16_t [single_M * single_K][mult_X * mult_Y];
	auto matB = new int16_t [single_K * single_N][mult_Y * mult_Z];
	auto matC = new int [single_M * single_N][mult_X * mult_Z];

	// each chuck_C computes the single AI MatMul kernel
	// Then reduced on mult_Y dimension
	auto chunk_C = new int [single_M * single_N][mult_Y];


	std::fstream a_file_array[mult_X * mult_Y];
	std::fstream b_file_array[mult_Y * mult_Z];
	std::fstream c_file_array[mult_X * mult_Z];


	for (int i = 0; i < mult_X * mult_Y; i++){
		a_file_array[i].open("./data/matA" + std::to_string(i) + ".txt", std::ios::out);
	}

	for (int i = 0; i < mult_Y * mult_Z; i++){
		b_file_array[i].open("./data/matB" + std::to_string(i) + ".h", std::ios::out);
	}

	for (int i = 0; i < mult_X * mult_Z; i++){
		c_file_array[i].open("./data/matC" + std::to_string(i) + ".txt", std::ios::out);
	}


	// B matrix
	for (int yz = 0; yz < mult_Y * mult_Z; yz++){

		// generate matB in blocked format
		for (int k = 0; k < single_K/K_API; k++){
			for (int j = 0; j < single_N/N_API; j++){

				for (int k_a = 0; k_a < K_API; k_a++){
					for (int n_a = 0; n_a < N_API; n_a++){

						matB[k*(single_N/N_API)*K_API*N_API + j*K_API*N_API + k_a*N_API + n_a][yz] = (rand()-128)%256;
					}
				}
			}
		}
		// write matB to matB.h
		b_file_array[yz] << "alignas(32) const signed short matB[" << int(single_K*single_N) << "] = { ";
		for (int i = 0; i < single_K*single_N; i++){
			b_file_array[yz] << int(matB[i][yz]);
			if (i == single_K*single_N-1) b_file_array[yz] << "};";
			else                          b_file_array[yz] << ", ";
		}
	}

	// seed
	srand(time(NULL));


	for (int batch = 0; batch < 10; batch++){

		// A matrix
		for (int xy = 0; xy < mult_X * mult_Y; xy++){

			// generate matA in blocked format
			for (int i = 0; i < single_M/M_API; i++){
				for (int k = 0; k < single_K/K_API; k++){

					for (int m_a = 0; m_a < M_API; m_a++){
						for (int k_a = 0; k_a < K_API; k_a++){

							matA[i*(single_K/K_API)*M_API*K_API + k*M_API*K_API + m_a*K_API + k_a][xy] = rand()%128;
						}
					}
				}
			}

			// write matB to matB.txt
			for (int i = 0; i < single_M*single_K; i++){

				a_file_array[xy] << int(matA[i][xy]);
				if (i % 8 == 7){
					a_file_array[xy] << "\n";
				}
				else{
					a_file_array[xy] << " ";
				}
			}
		}



		// Perform MatMul to generate the golden data
		for (int ii = 0; ii < mult_X; ii++){
			for (int jj = 0; jj < mult_Z; jj++){
				for (int kk = 0; kk < mult_Y; kk++){


					for (int i = 0; i < single_M/M_API; i++){
                        for (int j = 0; j < single_N/N_API; j++){

                            // initialize Chunk to 0
                            for (int m_a = 0; m_a < M_API; m_a++){
                                for (int n_a = 0; n_a < N_API; n_a++){
                                    chunk_C[i*(single_N/N_API)*M_API*N_API + j*M_API*N_API + m_a*N_API + n_a][kk] = 0;
                                }
                            }

                            for (int k = 0; k < single_K/K_API; k++){

                                for (int m_a = 0; m_a < M_API; m_a++){
                                    for (int n_a = 0; n_a < N_API; n_a++){
                                        for (int k_a = 0; k_a < K_API; k_a++){

                                            chunk_C[i*(single_N/N_API)*M_API*N_API + j*M_API*N_API + m_a*N_API + n_a][kk] += 
											matA[i*(single_K/K_API)*M_API*K_API + k*M_API*K_API + m_a*K_API + k_a][ii*mult_Y+ kk] * 
											matB[k*(single_N/N_API)*K_API*N_API + j*K_API*N_API + k_a*N_API + n_a][jj*mult_Y + kk];
                                        }
                                    }
                                }

                            }

                        }
					}


					// add the partial results (reduction in mult_Y dimension)

					if (kk == 0){
						for (int i = 0; i < single_M*single_N; i++){
							matC[i][ii*mult_Z + jj] = chunk_C[i][0];
						}
					}
					else {
						for (int i = 0; i < single_M*single_N; i++){
							matC[i][ii*mult_Z + jj] += chunk_C[i][kk];

						}
					}

				}
			}
		}



		for (int xz = 0; xz < mult_X * mult_Z; xz++){
			// write to output after elementwise addition
			for (int i = 0; i < single_M*single_N; i++){

				c_file_array[xz] << int(std::max((int16_t)(matC[i][xz]),(int16_t)0));
				if (i % 8 == 7){
					c_file_array[xz] << "\n";
				}
				else{
					c_file_array[xz] << " ";
				}
			}
		}

	}

   	// close files
   	for (int i = 0; i < mult_X * mult_Y; i++){
   		a_file_array[i].close();
   	}

   	for (int i = 0; i < mult_Y * mult_Z; i++){
   		b_file_array[i].close();
   	}

   	for (int i = 0; i < mult_X * mult_Z; i++){
   		c_file_array[i].close();
   	}


	delete[] matA;
	delete[] matB;
	delete[] matC;
	delete[] chunk_C;

	return 0;

}
