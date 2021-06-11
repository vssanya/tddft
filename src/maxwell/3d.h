#pragma once

#include "mpi_utils.h"

#include "../grid.h"
#include "../array.h"

#include "field_3d.h"


namespace maxwell {
	// dx / \lambda \ll 1 , for numerical calc use dx / \lambda = 20
	class Workspace3D {
		public:
			Workspace3D(Grid3d const& grid, MPI_Comm mpi_comm);
			~Workspace3D();

			void prop(Field3D field, double dt) {
				prop_H(field, dt);
				prop_E(field, dt);
			}

			void prop(Field3D field, double dt, Array3D<double> j[3]) {
				prop_H(field, dt);
				prop_E(field, dt, j);
			}

			void prop(Field3D field, double dt, Array3D<double>& sigma) {
				prop_H(field, dt);
				prop_E(field, dt, sigma);
			}

			Grid3d const& grid;
			Grid2d bound_grid;

			Array2D<double> rb_Ex;
			Array2D<double> rb_Ey;

			Array2D<double> lb_Hx;
			Array2D<double> lb_Hy;

		private:
			void prop_E(Field3D& field, double dt);
			void prop_E(Field3D& field, double dt, Array3D<double> j[3]);
			void prop_E(Field3D& field, double dt, Array3D<double>& sigma);
			void prop_H(Field3D& field, double dt);

#ifdef _MPI
			MPI_Comm mpi_comm;

			int mpi_size;
			int mpi_rank;
#endif
	};
}
