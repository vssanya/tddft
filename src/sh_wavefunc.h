#pragma once

#include <stdbool.h>
#include <math.h>
#include <functional>

#include "types.h"
#include "grid.h"
#include "sphere_harmonics.h"

/*!
 * \brief Волновая функция представленная в виде разложения по сферическим гармоникам
 *
 * \f[\psi(\vec{r}) = \frac{1}{r} \sum_{l=0}^{\infty} \Theta_{lm}(r) Y_{lm}(\theta, \phi) \simeq \frac{1}{r} \sum_{l=0}^{N_l - 1} \Theta_{lm}(r) Y_{lm}(\theta, \phi)\f]
 *
 * */
class ShWavefunc {
public:
  ShGrid const* grid;

  cdouble* data; //!< data[i + l*grid->Nr] = \f$\Theta_{lm}(r_i)\f$
  bool data_own; //!< кто выделил данные

  int m;         //!< is magnetic quantum number

  ShWavefunc(cdouble* data, ShGrid const* grid, int const m);
  ShWavefunc(ShGrid const* grid, int const m): ShWavefunc(nullptr, grid, m) {}
  ~ShWavefunc();


  inline cdouble& operator() (int ir, int il) {
	  assert(ir < grid->n[iR] && il < grid->n[iL]);
	  return data[ir + il*grid->n[iR]];
  }

  inline cdouble const& operator() (int ir, int il) const {
	  assert(ir < grid->n[iR] && il < grid->n[iL]);
	  return data[ir + il*grid->n[iR]];
  }

  inline cdouble d_dr(int ir, int il) const {
	  return (-(*this)(ir+2, il) + 8*(*this)(ir+1, il) - 8*(*this)(ir-1, il) + (*this)(ir-2, il))/(12*grid->d[iR]);
  }

  inline double abs_2(int ir, int il) const {
	  cdouble const value = (*this)(ir, il);
	  return pow(creal(value), 2) + pow(cimag(value), 2);
  }

  void copy(ShWavefunc* wf_dest) const;

// \return \f$<\psi_1|\psi_2>\f$
  cdouble operator*(ShWavefunc const& other) const;
  void exclude(ShWavefunc const& other);

// <psi|U(r)cos(\theta)|psi>
  double cos(sh_f func) const;
  void   cos_r(sh_f U, double* res) const;
  double cos_r2(sh_f U, int Z) const;

  double norm(sh_f mask = nullptr) const;
  void normalize();

  double z() const;

  void random_l(int l);

  template<class T>
  inline T integrate(std::function<T(ShWavefunc const*, int, int)> func, int l_max) const {
      T res = 0.0;
#pragma omp parallel for reduction(+:res) collapse(2)
      for (int il = 0; il < l_max; ++il) {
          for (int ir = 0; ir < grid->n[iR]; ++ir) {
              res += func(this, ir, il);
          }
      }
      return res*grid->d[iR];
  }

  static void ort_l(int l, int n, ShWavefunc** wfs);

  /*!
 * \return \f$\psi(r, \Omega)\f$
 * */
  cdouble get_sp(SpGrid const* grid, int i[3], YlmCache const* ylm_cache) const;

  void n_sp(SpGrid const* grid, double* n, YlmCache const* ylm_cache) const;

  void print() const;
};
