#include "atom.h"
#include "fields.h"
#include <array.h>
#include <grid.h>
#include <wavefunc/cart_3d.h>

#include "fftw3.h"


namespace workspace {
	class CartWavefuncWS {
		public:
		CartWavefuncWS(CartWavefunc3D* wf, Atom const& atom, Array3D<double> const* uabs);

		void prop_abs(double dt);
		void prop(field_t const* field, double t, double dt);
		void prop_img(double dt);
		void prop_r(double E[3], cdouble dt);
		void prop_norm(double norm);
		void prop_r_norm(double E[3], double norm, cdouble dt);
		void prop_r_norm_abs(double E[3], double norm, double dt);

		CartWavefunc3D* wf;
		Grid3d const& grid;

		Atom const& atom;
		Array3D<double> const* uabs;

		fftw_plan fp_forward;
		fftw_plan fp_backward;
	};
}
