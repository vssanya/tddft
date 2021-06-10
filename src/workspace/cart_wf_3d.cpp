#include "cart_wf_3d.h"
#include <fftw3.h>


workspace::CartWavefuncWS::CartWavefuncWS(CartWavefunc3D* wf, Atom const& atom, Array3D<double> const* uabs):
	wf(wf),
	grid(wf->grid),
	atom(atom),
	uabs(uabs) {
		fp_forward = fftw_plan_dft_3d(grid.n[2], grid.n[1], grid.n[0], (fftw_complex*) wf->data, (fftw_complex*) wf->data, FFTW_FORWARD, FFTW_MEASURE);
		fp_backward = fftw_plan_dft_3d(grid.n[2], grid.n[1], grid.n[0], (fftw_complex*) wf->data, (fftw_complex*) wf->data, FFTW_BACKWARD, FFTW_MEASURE);
}

void workspace::CartWavefuncWS::prop_r(double E[3], cdouble dt) {
#pragma omp parallel for collapse(3)
	for (int iz=0; iz<grid.n[2]; iz++) {
		for (int iy=0; iy<grid.n[1]; iy++) {
			for (int ix=0; ix<grid.n[0]; ix++) {
				double z = grid.z(iz);
				double y = grid.y(iy);
				double x = grid.x(ix);

				double r = sqrt(x*x + y*y + z*z);

				double Hr = x*E[0] + y*E[1] + z*E[2] + atom.u(r);
				(*wf)(ix, iy, iz) *= cexp(-0.5*I*dt*Hr);
			}
		}
	}
}

void workspace::CartWavefuncWS::prop_abs(double dt) {
#pragma omp parallel for collapse(3)
	for (int iz=0; iz<grid.n[2]; iz++) {
		for (int iy=0; iy<grid.n[1]; iy++) {
			for (int ix=0; ix<grid.n[0]; ix++) {
				(*wf)(ix, iy, iz) *= exp(-(*uabs)(ix, iy, iz)*dt);
			}
		}
	}
}

void workspace::CartWavefuncWS::prop_r_norm_abs(double E[3], double norm, double dt) {
#pragma omp parallel for collapse(3)
	for (int iz=0; iz<grid.n[2]; iz++) {
		for (int iy=0; iy<grid.n[1]; iy++) {
			for (int ix=0; ix<grid.n[0]; ix++) {
				double z = grid.z(iz);
				double y = grid.y(iy);
				double x = grid.x(ix);

				double r = sqrt(x*x + y*y + z*z);

				double Hr = x*E[0] + y*E[1] + z*E[2] + atom.u(r);
				(*wf)(ix, iy, iz) *= cexp(-0.5*I*dt*Hr)*exp(-(*uabs)(ix, iy, iz)*dt)/norm;
			}
		}
	}
}

void workspace::CartWavefuncWS::prop_r_norm(double E[3], double norm, cdouble dt) {
#pragma omp parallel for collapse(3)
	for (int iz=0; iz<grid.n[2]; iz++) {
		for (int iy=0; iy<grid.n[1]; iy++) {
			for (int ix=0; ix<grid.n[0]; ix++) {
				double z = grid.z(iz);
				double y = grid.y(iy);
				double x = grid.x(ix);

				double r = sqrt(x*x + y*y + z*z);

				double Hr = x*E[0] + y*E[1] + z*E[2] + atom.u(r);
				(*wf)(ix, iy, iz) *= cexp(-0.5*I*dt*Hr)/norm;
			}
		}
	}
}

void workspace::CartWavefuncWS::prop_norm(double norm) {
#pragma omp parallel for collapse(3)
	for (int iz=0; iz<grid.n[2]; iz++) {
		for (int iy=0; iy<grid.n[1]; iy++) {
			for (int ix=0; ix<grid.n[0]; ix++) {
				(*wf)(ix, iy, iz) /= norm;
			}
		}
	}
}

void workspace::CartWavefuncWS::prop(const field_t *field, double t, double dt) {
	double E[3] = {field_E(field, t+dt/2)};

	prop_r(E, dt);
	
	fftw_execute(fp_forward);

#pragma omp parallel for collapse(3)
	for (int iz=0; iz<grid.n[2]; iz++) {
		for (int iy=0; iy<grid.n[1]; iy++) {
			for (int ix=0; ix<grid.n[0]; ix++) {
				double pz = grid.pz(iz);
				double py = grid.py(iy);
				double px = grid.px(ix);

				double p2 = px*px + py*py + pz*pz;

				double Hp = 0.5*p2;
				(*wf)(ix, iy, iz) *= cexp(-I*dt*Hp);
			}
		}
	}

	fftw_execute(fp_backward);

	if (uabs != nullptr) {
		prop_r_norm_abs(E, grid.size(), dt);
	} else {
		prop_r_norm(E, grid.size(), dt);
	}
}

void workspace::CartWavefuncWS::prop_img(double dt) {
	double E[3] = {0.0, 0.0, 0.0};

	prop_r(E, -I*dt);
	
	fftw_execute(fp_forward);

#pragma omp parallel for collapse(3)
	for (int iz=0; iz<grid.n[2]; iz++) {
		for (int iy=0; iy<grid.n[1]; iy++) {
			for (int ix=0; ix<grid.n[0]; ix++) {
				double pz = grid.pz(iz);
				double py = grid.py(iy);
				double px = grid.px(ix);

				double p2 = px*px + py*py + pz*pz;

				double Hp = 0.5*p2;
				(*wf)(ix, iy, iz) *= cexp(-dt*Hp);
			}
		}
	}

	fftw_execute(fp_backward);

	prop_r_norm(E, grid.size(), -I*dt);
}
