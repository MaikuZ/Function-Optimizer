#ifndef FO_FUNCTIONS_HEADER
#define FO_FUNCTIONS_HEADER
/*!
  Author: MaikuZ
*/
#include <vector>

#include "utilities.h"

inline HD float pow(float a, int b) {
  float ret = 1;
  while (b--) 
    ret *= a;
  return ret;
}

///https://www.sfu.ca/~ssurjano/rosen.html
template<int Dim>
HD float rosenbrock_function(float *Arg) {
  float out = 0;
  for (int i = 0; i < Dim - 1;i++) {
    out += 100 * pow((Arg[i + 1]) - pow(Arg[i], 2), 2) + pow(1 - Arg[i], 2);
  }
  return out;
}

template<int Dim>
std::vector<Bound> rosenbrock_function_bounds() {
  std::vector<Bound> bounds;
  for (int i = 0;i < Dim;i++) {
    bounds.push_back({-5.0, 10.0});
  }
  return bounds;
}

///https://www.sfu.ca/~ssurjano/ackley.html
HD float ackley_function_2(float *Arg) {
  return (-20) * exp(-0.2 * sqrtf(0.5 * (pow(Arg[0], 2) + pow(Arg[1], 2)))) -
          expf(0.5 * (cosf(2 * M_PI * Arg[0]) + cosf(2 * M_PI * Arg[1]))) + expf(1) + 20;
}

std::vector<Bound> ackley_function_2_bounds() {
  int Dim = 2;
  std::vector<Bound> bounds;
  for (int i = 0;i < Dim;i++) {
    bounds.push_back({-32.768, 32.768});
  }
  return bounds;
}

///https://www.sfu.ca/~ssurjano/booth.html
HD float booth_function_2(float *Arg) {
  return pow(Arg[0] + 2 * Arg[1] - 7, 2) + pow(2 * Arg[0] + Arg[1] - 5, 2);
}

std::vector<Bound> booth_function_2_bounds() {
  int Dim = 2;
  std::vector<Bound> bounds;
  for (int i = 0;i < Dim;i++) {
    bounds.push_back({-10, 10});
  }
  return bounds;
}

///https://www.sfu.ca/~ssurjano/easom.html
HD float easom_function_2(float *Arg) {
  return -cosf(Arg[0]) * cosf(Arg[1]) 
  * expf(-pow(Arg[0] - M_PI, 2) - pow(Arg[1] - M_PI, 2));
}

std::vector<Bound> easom_function_2_bounds() {
  int Dim = 2;
  std::vector<Bound> bounds;
  for (int i = 0;i < Dim;i++) {
    bounds.push_back({-100, 100});
  }
  return bounds;
}

///https://www.sfu.ca/~ssurjano/rastr.html
template<int Dim>
HD float rastrigin_function(float *Arg) {
  float out = 10 * Dim;
  for (int i = 0;i < Dim;i++) {
    out += Arg[i] * Arg[i] - 10 * cosf(2 * M_PI * Arg[0]);
  }
  return out;
}

template<int Dim>
std::vector<Bound> rastrigin_function_bounds() {
  std::vector<Bound> bounds;
  for (int i = 0;i < Dim;i++) {
    bounds.push_back({-5.12, 5.12});
  }
  return bounds;
}

#endif