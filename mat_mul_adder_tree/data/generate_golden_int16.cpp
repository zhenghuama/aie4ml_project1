#include <iostream>
#include <fstream>
#include "../src/kernels/include.h"

using namespace std;

int main() {
   for (int k = 0; k < T; ++k) {
       std::ofstream B;
       B.open("B_"+to_string(k)+".txt", std::ios::trunc);

       for (int i = 0; i < M; ++i) {
           for (int j = k*(K/T); j < (k+1)*(K/T); ++j) {
	       B << ((j == i) ? 1 : 0) << " ";
	       if (j % 8 == 7) B << "\n";
           }
       }
       B.close();
   }
   std::ofstream A;
   A.open("A_matrix.txt", std::ios::trunc);
   for (int i = 1; i <= K*N; ++i) {
       A << i << " ";
       if (i % 8 == 0) A << "\n";
   }
   A.close();
}
