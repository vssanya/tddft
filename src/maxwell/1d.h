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
		typedef Array2D<double, Grid2d> Array2;

		public:
			Workspace2D(Grid2d const& grid);
			~Workspace2D();

			Grid2d const& grid;

			Array2 Ez;
			Array2 Dz;
			Array2 Hx;
			Array2 Hy;

			void prop(double dt);
			void prop(double dt, double eps[]);

		private:
			static void prop_Hx(Array2& Hx, Array2 const& Ez, double ksi);
			static void prop_Hy(Array2& Hy, Array2 const& Ez, double ksi);
			static void prop_Dz(Array2& Dz, Array2 const& Hx, Array2 const& Hy, double ksi);
			static void prop_Ez(Array2& Ez, Array2 const& Dz, Array2 const& eps);
	};
}
