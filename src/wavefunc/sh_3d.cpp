#include <cmath>

#include "sh_3d.h"
#include "../grid.h"


// <lm|11|l'm'> = √(3/8π) δ_{m,m'+1} (δ_{l,l'+1} a_{lm} - δ_{l,l'-1} b_{l+1,m-1})
// <lm|1,-1|l'm'> = √(3/8π) δ_{m,m'-1} (δ_{l,l'+1} b_{lm} - δ_{l,l'-1} a_{l+1,m+1})

double alm(int l, int m) {
	return std::sqrt((l+m-1)*(l+m)/((2*l-1)*(2*l+1)));
}

double blm(int l, int m) {
	return std::sqrt((l-m-1)*(l-m)/((2*l-1)*(2*l+1)));
}

template <typename Grid>
double ShWavefunc3D<Grid>::cos(sh_f func) const {
	return 2*this->grid.template integrate<double>([this, func](int ir, int il, int im) -> double {
			return clm(il, im)*creal((*this)(ir, il, im)*conj((*this)(ir, il+1, im)))*func(ir, il, im);
			}, this->grid.n[iL]-1);
}

template <typename Grid>
cdouble ShWavefunc3D<Grid>::sin_sin(sh_f func) const {
	return -this->grid.template integrate<double>([this, func](int ir, int il, int im) -> double {
			return (cimag(conj((*this)(ir, il, im))*(*this)(ir, il-1, im-1))*alm(il, im) + cimag(conj((*this)(ir, il, im-1))*(*this)(ir, il-1, im))*blm(il, im-1))*func(ir, il, im);
			}, this->grid.n[iL], 1, -1);
}

template <typename Grid>
cdouble ShWavefunc3D<Grid>::sin_cos(sh_f func) const {
	return this->grid.template integrate<double>([this, func](int ir, int il, int im) -> double {
			return (-creal(conj((*this)(ir, il, im))*(*this)(ir, il-1, im-1))*alm(il, im) + creal(conj((*this)(ir, il, im-1))*(*this)(ir, il-1, im))*blm(il, im-1))*func(ir, il, im);
			}, this->grid.n[iL], 1, -1);
}

template class ShWavefunc3D<ShNotEqudistantGrid3D>;
