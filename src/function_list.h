#ifndef FO_FUNCTION_LIST_HEADER
#define FO_FUNCTION_LIST_HEADER
/*!
  Author: MaikuZ
*/

/*!
  For convenience sake and because of CUDA limitations all the
  functions that are being tested are included in this macro.
  This way it is easy to go for each function that is used in the
  template of the particle swarm and simulated annealing implementations.

  There is also the information about the optimal value, and the
  dimension of the function.

  FORMAT:
  MACRO(FUNCTION_NAME, DIMENSION, OPTIMAL_VALUE, BOUNDS_FUNCTION_NAME)
*/
#define FOR_EACH_TEST_FUNCTION(MACRO)                                   \
MACRO(rosenbrock_function<2>, 2, 0, rosenbrock_function_bounds<2>)      \
MACRO(rosenbrock_function<3>, 3, 0, rosenbrock_function_bounds<3>)      \
MACRO(rastrigin_function<2>, 2, 0, rastrigin_function_bounds<2>)        \
MACRO(ackley_function_2, 2, 0, ackley_function_2_bounds)                \
MACRO(rastrigin_function<7>, 7, 0, rastrigin_function_bounds<7>)        \
MACRO(booth_function_2, 2, 0, booth_function_2_bounds)                  \
MACRO(easom_function_2, 2, -1, easom_function_2_bounds)                 \

#endif