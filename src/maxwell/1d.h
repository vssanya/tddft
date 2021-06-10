#pragma once

#include "../grid.h"
#include "../array.h"


namespace maxwell {
	// dx / \lambda \ll 1 , for numerical calc use dx / \lambda = 20
	class Workspace1D {
		typedef Array1D<double> Arr1;

		public:
			Workspace1D(Grid1d const& grid);
			~Workspace1D();

			void prop(double dt);
			void prop(double dt, Arr1 const& eps);
			void prop(double dt, double* eps) {
				auto arr = Arr1(eps, grid);
				prop(dt, arr);
			};

			void prop_pol(double dt, Arr1 const& P);
			void prop_pol(double dt, double* P) {
				auto arr = Arr1(P, grid);
				prop_pol(dt, arr);
			};

			// return shift in count of points
			int move_center_window_to_max_E();

			Grid1d const& grid;

			Arr1 E;
			Arr1 D;
			Arr1 H;

		private:
			void prop_Bz(Arr1& Bz, Arr1 const& Ey, double ksi) const;
			void prop_Dy(Arr1& Dy, Arr1 const& Hz, double ksi) const;
	};

	class WorkspaceCyl1D {
		typedef Array1D<double> Arr1;

		public:
			WorkspaceCyl1D(Grid1d const& grid);
			~WorkspaceCyl1D();

			void prop(double dt);
			// N: r=[0, dr/2, dr, 3dr/2, 2dr, ...]
			void prop(double dt, double* N, double nu);

			Grid1d const& grid;

			Arr1 Er; // r = [dr/2, 3dr/2, ...]
			Arr1 Ephi; // r = [0, dr, 2dr, ...]
			Arr1 Hz; // r = [dr/2, 3dr/2, ...]

			Arr1 jr; // r = [0, dr, 2dr, ...]
			Arr1 jphi; // r = [dr/2, 3dr/2, ...]
	};

	class Workspace2D {
		typedef Array2D<double> Arr2;

		public:
			Workspace2D(Grid2d const& grid);
			~Workspace2D();

			Grid2d const& grid;

			Arr2 Ez;
			Arr2 Dz;
			Arr2 Hx;
			Arr2 Hy;

			void prop(double dt);
			void prop(double dt, Arr2 const& eps);

		private:
			static void prop_Hx(Arr2& Hx, Arr2 const& Ez, double ksi);
			static void prop_Hy(Arr2& Hy, Arr2 const& Ez, double ksi);
			static void prop_Dz(Arr2& Dz, Arr2 const& Hx, Arr2 const& Hy, double ksi);
			static void prop_Ez(Arr2& Ez, Arr2 const& Dz, Arr2 const& eps);
	};
}
