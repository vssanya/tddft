#pragma once


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  double a00; // top-left element
  double a[3];
  int N;
} tdm_t;

/*!
 * \brief Inverse of a tridiagonal matrix (a)
 * */
void linalg_tdm_inv(double a00, double const a[3], int N, double* b);

/*!
 * \brief Multiply matrix (a) on tridiagonal matrix (b)
 * */
void linalg_m_dot_tdm(int N, double* a, double b00, double const b[3]);

#ifdef __cplusplus
}
#endif
