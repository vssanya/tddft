#pragma once

#include "../grid.h"
#include "../array.h"


namespace maxwell {
	// dx / \lambda \ll 1 , for numerical calc use dx / \lambda = 20
	class Workspace3D {
		typedef Array3D<double> Arr3;

		public:
			Workspace3D(Grid3d const& grid);
			~Workspace3D();

			void prop(double dt) {
				prop_H(dt);
				prop_E(dt);
			}

			void prop(double dt, Arr3 j[3]) {
				prop_H(dt);
				prop_E(dt, j);
			}

			Grid3d const& grid;

			Arr3 Ex;
			Arr3 Ey;
			Arr3 Ez;

			Arr3 Hx;
			Arr3 Hy;
			Arr3 Hz;

		private:
			void prop_E(double dt);
			void prop_E(double dt, Arr3 j[3]);
			void prop_H(double dt);
	};
}
