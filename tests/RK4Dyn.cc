#include <iostream>
#include <fstream>
#include<string>
#include <math.h>
#include <numeric>
#include <random>
#include <algorithm>
#include <complex>
#include <sys/time.h>
#include <omp.h>

#include "../Utils/Utils.h"
#include "../RungeKutta/RK4.h"

std::vector<double> linspace(double a, double b, size_t n)
{
  double h = (b - a) / static_cast<double>(n - 1);
  std::vector<double> vec(n);
  typename std::vector<double>::iterator x;
  double val;
  for (x = vec.begin(), val = a; x != vec.end(); ++x, val += h)
    *x = val;
  return vec;
}

/*******************************************************************************/
// Random Complex 1D array Generator (Could be use for both vectors and matrices)
// It initializes the values of a MZType array with random numbers from a uniform
// distribution with limits scaled by scale.
/*******************************************************************************/
void zrndgen(MZType &arr, double scale)
{
  MKL_INT dim = static_cast<MKL_INT>(arr.size());
  std::random_device rd; 
  std::mt19937 rng(rd());
  std::uniform_real_distribution<double> udist(-1.0 * scale ,1.0 * scale);

  for (MKL_INT i = 0; i < dim; i++){
    arr[i] = {udist(rng), udist(rng)};        
  }
}

/*******************************************************************************/
// writes a double precision vector into a file with a single row with column 
// elements separated by single space.
/*******************************************************************************/
void write_vec(std::vector<double> &vec, std::string filename)
{
  MKL_INT dim = static_cast<MKL_INT>(vec.size());
  std::ofstream vecFile(filename);
  if (vecFile.is_open()){
      for (MKL_INT i = 0; i < dim; i++){
        vecFile << vec[i] << (i < dim - 1 ? " " : "");
      }
      vecFile << std::endl;
  }
  vecFile.close();
}

/*******************************************************************************/
// writes a MZType vector into a file with a single row with column elements 
// separated by single space using the format vec[i] = a + bj (python_like)
/*******************************************************************************/
void write_vec(MZType &vec, std::string filename)
{
  MKL_INT dim = static_cast<MKL_INT>(vec.size());
  std::ofstream vecFile(filename);
  if (vecFile.is_open()){
      for (MKL_INT i = 0; i < dim; i++){
        vecFile << vec[i].real() << (vec[i].imag() >= 0 ? "+" : "") << vec[i].imag() << "j" << (i < dim - 1 ? " " : "");
      }
      vecFile << std::endl;
  }
  vecFile.close();
}

/*******************************************************************************/
// writes MZType vectors into a file without overwritting them, i.e., stacks them
// in rows. The format for the row vector entries is vec[i] = a + bj (python_like)
/*******************************************************************************/
void write_vec_stack(MZType &vec, std::string filename)
{
  MKL_INT dim = static_cast<MKL_INT>(vec.size());
  std::ofstream vecFile(filename, std::ios::app); // ios::app tell ofstream to look for the last line and write from there.
  if(vecFile.is_open()){
    for (MKL_INT j = 0; j < dim; ++j){
      vecFile << vec[j].real() << (vec[j].imag() >= 0.0 ? "+" : "") << vec[j].imag() << "j" << (j < dim - 1 ? " " : "");
    }
    vecFile << std::endl;
  }
  vecFile.close();
}

/*******************************************************************************/
// writes a MZType square matrix into a file with columns separated by single 
// space and using the entry format mat[(i * cols) + j] = a + bj (python_like)
/*******************************************************************************/
void write_mat(MZType &mat, std::string filename)
{
  MKL_INT dim = static_cast<MKL_INT>(std::sqrt(mat.size()));
  std::ofstream matrixFile(filename);

  if (matrixFile.is_open()){
    for (MKL_INT i = 0; i < dim; i++){
      for (MKL_INT j = 0; j < dim; j++){
               matrixFile << mat[i*dim + j].real() << (mat[i*dim + j].imag() >= 0 ? "+" : "") << mat[i*dim + j].imag() << "j" << (j < dim - 1 ? " " : "");
      }
      matrixFile << std::endl;
    }
  }
  else{
      std::cerr << "Error: could not open file." << std::endl;
  }
  matrixFile.close();
}
/*******************************************************************************/
// Erases the existing data of a given file
/*******************************************************************************/
void cleanFile(std::string filename)
{
  //assertm(std::filesystem::exists(filename), "File was not found. Cleaning could not be done.");
  std::ofstream file(filename, std::ios::trunc);
  file.close();
}

int main(int argc, char **argv)
{
  MKL_INT dim = 777;
  double tmax = 0.777;

  if(argc != 5){ 
    std::cerr << "Usage: " << argv[0] << " --N [size of states] --tt [max value of time]" << std::endl;
    exit(1);
  }
  for(int i = 0; i < argc; ++i){
    std::string str = argv[i];
    if(str == "--N") dim = atoi(argv[i + 1]);
    if(str == "--tt") tmax = atof(argv[i + 1]);
    else continue;
  }
  if(dim == 777 || tmax == 0.777){
    std::cerr << "Error setting parameters" << std::endl;
    std::cerr << "Usage: " << argv[0] << " --N [size of states] --tt [max value of time]" << std::endl;
    exit(1);
  }  
  /*****************************************************************************/
  // Memory allocation
  /*****************************************************************************/
  MZType Q(dim * dim, 0.0);
  MZType R(dim * dim, 0.0);
  MZType w0(dim, 0.0); 
  MZType v0(dim, 0.0);

  /*****************************************************************************/
  // uniform random initialization
  /*****************************************************************************/
  zrndgen(Q, 0.1);
  zrndgen(R, 0.1);
  zrndgen(w0, 0.1);
  zrndgen(v0, 0.1);

  /*****************************************************************************/
  // Copy constructor for the v states
  /*****************************************************************************/
  MZType v0h2(v0);
  MZType v0h(v0);

  /*****************************************************************************/
  // Writting of initial parameters
  /*****************************************************************************/
  write_mat(Q, "tests/data/Q.csv");
  write_mat(R, "tests/data/R.csv");
  write_vec(w0, "tests/data/w0.csv");
  write_vec(v0, "tests/data/v0.csv");

  cleanFile("tests/data/sol.csv"); //the function write_vec() stacks under the written data so the file must be wiped out before.
  
  /*****************************************************************************/
  // Dynamic evolution
  /*****************************************************************************/
 
  RK4 rk4(dim);

  std::vector<double> times = linspace(0.0, tmax, 10001);
  double delta_t = times[1] - times[0];
  MKL_INT tsteps = times.size();
  
  //Pre-steping  
  rk4.rk4_redfield(0.5 * delta_t, v0h2, Q);
  rk4.rk4_redfield(delta_t, v0h, Q);
  
  for (MKL_INT tt = 0; tt < tsteps; ++tt){
    write_vec_stack(w0, "tests/data/sol.csv"); //stacks the evolved states into a file stacked by rows
    rk4.rk4_redfield_twoops(delta_t, w0, Q, v0, v0h2, v0h, R); //Solves for w(t + h)
    v0 = v0h; //v from the past now is the future one
    rk4.rk4_redfield(delta_t, v0h2, Q); //Solves for v(t + h/2)
    rk4.rk4_redfield(delta_t, v0h, Q); //Solves for v(t + h)
  } 

  write_vec(times, "tests/data/time_eval.csv"); //writes the evaluation times of the dynamics.
  return 0;
}
