#pragma once

#include "../grid.h"
#include "../array.h"


namespace maxwell {
	// dx / \lambda \ll 1 , for numerical calc use dx / \lambda = 20
	class Workspace1D {
		public:
			Workspace1D(Grid1d const& grid);
			~Workspace1D();

			void prop(double dt);
			void prop(double dt, double eps[]);

			Grid1d const& grid;

			double* E;
			double* D;
			double* H;
	};

	class Workspace2D {
		public:
			Workspace2D(Grid2d const& grid);
			~Workspace2D();

			Grid2d const& grid;

			Array2D Ez;
			Array2D Dz;
			Array2D Hx;
			Array2D Hy;

			void prop(double dt);
			void prop(double dt, double eps[]);

		private:
			static void prop_Hx(Array2D& Hx, Array2D const& Ez, double ksi);
			static void prop_Hy(Array2D& Hy, Array2D const& Ez, double ksi);
			static void prop_Dz(Array2D& Dz, Array2D const& Hx, Array2D const& Hy, double ksi);
			static void prop_Ez(Array2D& Ez, Array2D const& Dz, Array2D const& eps);
	};
}
