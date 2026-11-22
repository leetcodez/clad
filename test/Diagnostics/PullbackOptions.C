// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"

double fn_test(double x) {
  return x * x;
}

struct FunctorTest {
  double operator()(double x) const { return x * x; }
};

#include <initializer_list>

double fn_init_list(std::initializer_list<double> l) { return 0.0; }

// 6. Dependent non-const pointer parameter in partial pullback
double fn_mut_ptr(double* tmp, double x) { // expected-error {{dependent non-const pointer and array parameters are not supported; differentiate w.r.t. 'tmp' or mark it const}}
  tmp[0] = x * x;
  return tmp[0] * 3.0;
}

void f_void_mut(double* a, double* b) { // expected-error {{dependent non-const pointer and array parameters are not supported; differentiate w.r.t. 'a' or mark it const}}
  *a += 2.0;
  *b = *b * 3.0;
}

double* f_ptr_multi(double* a, double* b) { // expected-error {{dependent non-const pointer and array parameters are not supported; differentiate w.r.t. 'a' or mark it const}}
  *a += 2.0;
  *b += 3.0;
  return b;
}

struct Point {
  double x;
  double y;
};

double fn_struct(Point s) {
  return s.x * s.y;
}

double fn_arr(const double* p) {
  return p[0] + p[1];
}

// 7. Custom pullback adaptation requires scratch of unknown size
double fn_custom_unknown(const double* arr, double y) {
  return arr[0] * y;
}
namespace clad {
namespace custom_derivatives {
void fn_custom_unknown_pullback(const double* arr, double y, double seed, double* d_arr, double* d_y);
}
}

int main() {
  // 1. Enzyme option is not supported for pullback mode
  auto pb_enz = clad::pullback<clad::opts::use_enzyme>(fn_test); // expected-error {{enzyme is not supported for pullback mode}}

  // 2. Vector mode option is not supported for pullback mode
  auto pb_vec = clad::pullback<clad::opts::vector_mode>(fn_test); // expected-error {{reverse vector mode is not yet supported}}

  // 3. std::initializer_list is not supported for pullback mode
  auto pb_init = clad::pullback(fn_init_list); // expected-error {{pullback does not support functions with std::initializer_list parameters}}

  // 4. Lambda expressions are not supported for pullback mode (use named function objects)
  auto l_stateless = [](double x) { return x * x; };
  auto pb_l_stateless = clad::pullback(l_stateless); // expected-error {{pullback does not support lambda expressions; use named function objects instead}}
  auto pb_l_stateless_part = clad::pullback(l_stateless, "x"); // expected-error {{pullback does not support lambda expressions; use named function objects instead}}

  double a = 3.0;
  auto l_cap = [a](double x) { return a * x; };
  auto pb_l_cap = clad::pullback(l_cap); // expected-error {{pullback does not support lambda expressions; use named function objects instead}}
  auto pb_l_cap_part = clad::pullback(l_cap, "x"); // expected-error {{pullback does not support lambda expressions; use named function objects instead}}

  // 5. Invalid / malformed argument selection diagnostics
  auto pb_empty = clad::pullback(fn_test, ""); // expected-error {{no parameters were provided}}
  auto pb_invalid = clad::pullback(fn_test, "nonexistent"); // expected-error {{requested parameter name 'nonexistent' was not found among function parameters}}

  // 6. Dependent mutable pointer error
  auto pb_mut_ptr = clad::pullback(fn_mut_ptr, "x");
  auto pb_void_b = clad::pullback(f_void_mut, "b");
  auto pb_ptr_b = clad::pullback(f_ptr_multi, "b");

  // 7. Unsafe pointer scratch for custom pullback adaptation error
  auto pb_custom_unknown = clad::pullback(fn_custom_unknown, "y"); // expected-error {{cannot adapt custom pullback for partial selection; parameter 'arr' requires scratch storage of unknown size}}

  // 8. Unsupported index ranges and fields
  auto pb_idx = clad::pullback(fn_arr, "p[1]"); // expected-error {{clad::pullback does not support index ranges or fields yet}}
  auto pb_field = clad::pullback(fn_struct, "s.x"); // expected-error {{clad::pullback does not support index ranges or fields yet}}

  // 9. Subsequent valid differentiation succeeds cleanly in the same Sema instance
  auto pb_ok = clad::pullback(fn_test);
  double dx = 0.0;
  pb_ok.execute(3.0, 1.0, &dx);

  return 0;
}
