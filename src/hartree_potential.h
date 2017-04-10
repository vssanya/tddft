#pragma once

#include "utils.h"
#include "orbitals.h"

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
void hartree_potential_l0(orbitals_t const* orbs, double U[orbs->grid->n[iR]], double U_local[orbs->grid->n[iR]], double f[orbs->grid->n[iR]]);

void hartree_potential_wf_l0(sh_wavefunc_t const* wf, double U[wf->grid->n[iR]], double f[wf->grid->n[iR]]);

/*!
 * U1(r,t) = \int L1(r',t) r< / r>^2 dr'
 * */
void hartree_potential_l1(orbitals_t const* orbs, double U[orbs->grid->n[iR]], double U_local[orbs->grid->n[iR]], double f[orbs->grid->n[iR]]);

/*!
 * U2(r,t) = \int L2(r',t) r<^2 / r>^3 dr'
 * */
void hartree_potential_l2(orbitals_t const* orbs, double U[orbs->grid->n[iR]], double U_local[orbs->grid->n[iR]], double f[orbs->grid->n[iR]]);

double mod_grad_n(sp_grid_t const* grid, double n[grid->n[iR]*grid->n[iC]], int ir, int ic);
double ux_lda_func(double n);
double uc_lda_func(double n);
double uxc_lb_func(double n, double x);
/*!
 * Обменное взаимодействие приближение локальной плотности
 * \param l[in]
 * \param ne[in] is count Kohn's orbitals
 * \param wf[in] is wavefunction of Kohn's orbitals
 * \param Ux[out] is amplitude \f$Y_l^0\f$ component of \f$U_{x} = - \left(\frac{3}{\pi}\right)^{1/3} n(\vec{r})^{1/3}\f$
 * */
void uxc_lb(
		int l, orbitals_t const* orbs,
		double U[orbs->grid->n[iR]],
		sp_grid_t const* grid,
		double n[grid->n[iR]*grid->n[iC]], // for calc using mpi
		double n_tmp[grid->n[iR]*grid->n[iC]], // for calc using mpi
		ylm_cache_t const* ylm_cache
);
