#pragma once

#include "utils.h"
#include "orbitals.h"

#include <optional>

#include "xc.h"

/*! \file
 * Разложение кулоновского потенциала по сферическим функциям
 * \f[ \frac{1}{\left|r - r'\right|} = \sum_{l=0}^{\infty} \frac{4\pi}{2l + 1} \frac{r_<^l}{r_>^{l+1}} \sum_{m=-l}^{l} Y_l^{m*}(\Omega') Y_l^{m}(\Omega) \f]
 * Here \f$r_< = \min(r', r)\f$ \f$r_> = \max(r', r)\f$
 *
 * Hantree potential
 * \f[U(\vec{r}) = \int \frac{n(\vec{r}')}{\left|\vec{r} - \vec{r}'\right|} d^3 r'\f]
 *
 * Expansion of Hantree potential by spheric functions Ylm:
 * \f[U = U_0 + U_1 Y_1^0 + U_2 Y_2^0 + ...\f]
 */

/*! 
 * U0(r,t) = 2*\sum_{i,l} \int |\theta_{ilm}(r', t)|^2 / r> dr'
 * */
template<typename Grid>
class HartreePotential {
	public:
		static void calc(
				Orbitals<Grid> const* orbs,
				int l, double* U, double* U_local,
				double* f, int order,
				std::optional<Range> rRange = std::nullopt);

		static void calc_int_func(
				Orbitals<Grid> const* orbs, int l, double* f,
				std::optional<Range> rRange = std::nullopt);

		static void calc_wf_l0(
				Wavefunc<Grid> const* wf, double* U, double* f, int order,
				std::optional<Range> rRange = std::nullopt);
};


template <typename Grid>
double mod_dndr(Grid const& grid, double* n, int ir);
double mod_grad_n(SpGrid const& grid, double* n, int ir, int ic);
double ux_lda_func(double n);
double uc_lda_func(double n);

double uxc_lb(double n, double x);
double uxc_lda(double n, double x);
double uxc_lda_x(double n, double x);

typedef double (*potential_xc_f)(double n, double x);
/*!
 * Обменное взаимодействие приближение локальной плотности
 * \param l[in]
 * \param ne[in] is count Kohn's orbitals
 * \param wf[in] is wavefunction of Kohn's orbitals
 * \param Ux[out] is amplitude \f$Y_l^0\f$ component of \f$U_{x} = - \left(\frac{3}{\pi}\right)^{1/3} n(\vec{r})^{1/3}\f$
 * */
template<typename Grid>
class XCPotential {
	public:
		static void calc_l(
				potential_xc_f uxc,
				int l, Orbitals<Grid> const* orbs,
				double* U,
				SpGrid const* grid,
				double* n, // for calc using mpi
				double* n_tmp, // for calc using mpi
				YlmCache const* ylm_cache,
				std::optional<Range> rRange = std::nullopt
				);

		static void calc_l0(
				potential_xc_f uxc,
				int l, Orbitals<Grid> const* orbs,
				double* U,
				SpGrid const* grid,
				double* n, // for calc using mpi
				double* n_tmp, // for calc using mpi
				YlmCache const* ylm_cache,
				std::optional<Range> rRange = std::nullopt
				);
};
