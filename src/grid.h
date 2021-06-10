#pragma once

#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <memory>
#include <functional>

#define check_index(space, index) assert(index >= 0 || index < n[space])

/*!
 * \brief Indexes for coordinate
 * */
enum {
    iX = 0, //!< is index for x
    iY = 1, //!< is index for y
    iZ = 2, //!< is index for z
};

/*!
 * \brief Indexes for spherical coordinate
 * */
enum {
    iR = 0, //!< is index for r
    iT = 1, //!< is index for \f$\theta\f$
    iP = 2, //!< is index for \f$\phi\f$
};

/*!
 * \brief Indexes for spherical harmonics
 * */
enum {
    iL = 1, //!< is index for l
    iM = 2, //!< is index for m
};


class Grid1d {
	public:
		int n;
		double d;

		double x(int ix) const {
			return ix*d;
		}

		size_t size() const { return n; }
		size_t index(int i) const { return i; }

		double dr(int i) const { return d; };

		Grid1d(int n, double d): n(n), d(d) {}
		Grid1d(int n): n(n), d(1.0) {}
		Grid1d(): n(0), d(1.0) {}
};

class GridNotEq1d {
	public:
		int n;
		double* d;

		size_t size() const { return n; }
		size_t index(int i) const { return i; }

		double dr(int i) const { return d[i]; };

		GridNotEq1d(int n, double* d): n(n), d(d) {}
		GridNotEq1d(int n): n(n), d(nullptr) {}
		GridNotEq1d(): n(0), d(nullptr) {}
};

/*!
 * \brief Grid for 2D space
 * */
class Grid2d {
public:
    int    n[2]; //!< is counts of points
    double d[2]; //!< is steps

	Grid2d(int nx, int ny, double dx, double dy): n{nx, ny}, d{dx, dy} {}
	Grid2d(int nx, int ny): Grid2d(nx, ny, 1.0, 1.0) {}
	Grid2d(): n{0}, d{1.0} {}

	size_t size() const { return n[0]*n[1]; }

	size_t index(int ix, int iy) const {
		check_index(0, ix);
		check_index(1, iy);

		return ix + iy*n[0];
	}
};

/*!
 * \brief Grid for 3D space
 * */
class Grid3d {
public:
    int    n[3]; //!< is counts of points
    double d[3]; //!< is steps
		double dp[3];

		Grid3d(): n{0}, d{0.0}, dp{0.0} {}
		Grid3d(int nx, int ny, int nz, double dx, double dy, double dz): n{nx, ny, nz}, d{dx, dy, dz}, dp{2*M_PI/(nx*dx), 2*M_PI/(ny*dy), 2*M_PI/(nz*dz)} {}

		Grid2d getSliceZ() const {
			return Grid2d(n[0], n[1], d[0], d[1]);
		}

		double x(int ix) const {
			return (ix - n[iX]/2 + 0.5)*d[0];
		}

		double y(int iy) const {
			return (iy - n[iY]/2 + 0.5)*d[1];
		}

		double z(int iz) const {
			return (iz - n[iZ]/2 + 0.5)*d[2];
		}

		template <int index>
		double p(int i) const {
			if (i < n[index]/2) {
				return i*dp[index];
			} else {
				return -(n[index] - i)*dp[index];
			}
		}

		double px(int ix) const {
			return p<0>(ix);
		}

		double py(int iy) const {
			return p<1>(iy);
		}

		double pz(int iz) const {
			return p<2>(iz);
		}

    size_t size() const { return n[0]*n[1]*n[2]; }

    size_t index(int ix, int iy, int iz) const {
        check_index(0, ix);
        check_index(1, iy);
        check_index(2, iz);

        return ix + iy*n[0] + iz*n[0]*n[1];
    }

		class Range {
			public:
				Range() {}
				Range(Grid3d const& g1, Grid3d const& g2) {}
		};

		template<class T>
			inline T integrate(std::function<T(int, int, int)> func, Range range = Range()) const {
				T res = 0.0;
#pragma omp parallel for reduction(+:res) collapse(3)
				for (int iz=0; iz<n[2]; iz++) {
					for (int iy=0; iy<n[1]; iy++) {
						for (int ix=0; ix<n[0]; ix++) {
							res += func(ix, iy, iz);
						}
					}
				}

				return res*d[iX]*d[iY]*d[iZ];
			}
};

class SpGrid2d: public Grid2d {
public:
	SpGrid2d(): Grid2d() {}
    SpGrid2d(int const n[2], double Rmax) {
        for (int i = 0; i < 2; ++i) {
            this->n[i] = n[i];
        }

        d[iR] = Rmax/n[iR];
        d[iT] = M_PI/(n[iT]-1.0);
    }

    double r(int ir) const {
        check_index(iR, ir);
        return d[iR]*(ir+1);
    }

    int l(int il) const {
        check_index(iL, il);
        return il;
    }

    double theta(int it) const {
        check_index(iT, it);
        return d[iT]*it;
    }

		double J(int ir, int il) const {
			return 1.0;
		}

		class Range {
			public:
			int r_min;
			int r_max;

			int m;

			Range(SpGrid2d const& g1, SpGrid2d const& g2): Range() {}
			Range(int r_min, int r_max, int m): r_min(r_min), r_max(r_max), m(m) {}
			Range(int r_min, int r_max): r_min(r_min), r_max(r_max), m(0) {}
			Range(): r_min(0), r_max(-1) {}

			int getRmax(SpGrid2d const& grid) {
				if (r_max < 0) {
					return grid.n[iR] + 1 - r_max;
				} else {
					return r_max;
				}
			}
		};


		template<class T>
			inline T integrate(std::function<T(int, int)> func, Range range = Range()) const {
				T res = 0.0;
#pragma omp parallel for reduction(+:res) collapse(2)
				for (int it = 0; it < n[iT]; ++it) {
					for (int ir = 0; ir < n[iR]; ++ir) {
						double r = this->r(ir);
						double theta = this->theta(it);
						res += func(ir, it)*r*r*sin(theta);
					}
				}
				return res*d[iR]*2*M_PI*d[iT];
			}
};

class SpGrid: public Grid3d {
public:
	SpGrid(): Grid3d() {}

    SpGrid(int n[3], double Rmax) {
        for (int i=0; i<3; ++i) {
            this->n[i] = n[i];
        }

        d[iR] = Rmax/n[iR];
        d[iT] = M_PI/(n[iT]-1.0);
        d[iP] = 2.0*M_PI/n[iP];
    }

	SpGrid2d getGrid2d() const {
		return SpGrid2d(n, d[iR]*n[iR]);
	}

    double r(int ir) const {
        check_index(iR, ir);
        return d[iR]*(ir+1);
    }

    double Rmax() const {
        return r(n[iR]-1);
    }

    double theta(int it) const {
        check_index(iT, it);
        return d[iT]*it;
    }

    double phi(int ip) const {
        check_index(iP, ip);
        return d[iP]*ip;
    }

    int ir(double r) const {
        return (int) (r/d[iR]) - 1;
    }

    int it(double theta) const {
        return (int) (theta / d[iT]);
    }
};

class ShGrid: public Grid2d {
public:
	ShGrid(): Grid2d() {}

	ShGrid(int n[2], double Rmax) {
		for (int i=0; i<2; ++i) {
			this->n[i] = n[i];
		}

		d[iR] = Rmax/n[iR];
		d[iL] = 1;

		double dr2 = d[iR]*d[iR];

		m_d2[0] = 1.0/dr2;
		m_d2[1] = -2.0/dr2;
		m_d2[2] = m_d2[0];
	}

	virtual double r(int ir) const {
		check_index(iR, ir);
		return d[iR]*(ir+1);
	}

	double Rmax() const {
		return r(n[iR]-1);
	}

	int l(int il) const {
		check_index(iL, il);
		return il;
	}

	int m(int im) const {
		assert(im >= -n[iL] && im <= n[iL]);
		return im;
	}

	/*
	 * Коэффициенты трехдиагональной матрицы оператора d^2/dr^2 
	 */
	double d2(int ir, int i) const {
		return m_d2[i];
	}

	double J(int ir, int il) const {
		return 1.0;
	}

	template <typename T>
	T d_dr(T* f, int ir) const {
		if (ir == 0) {
			return (f[ir+1] - f[ir])/d[iR];
		} else if (ir == n[iR] - 1) {
			return (f[ir] - f[ir-1])/d[iR];
		} else {
			return (f[ir-1] - f[ir+1])/(2*d[iR]);
		}
	}

	template <typename T>
	T d_dr_zero(T* f, int ir) const {
		if (ir == 0) {
			return (f[ir+1] - f[ir])/d[iR];
		} else if (ir == n[iR] - 1) {
			return (f[ir] - f[ir-1])/d[iR];
		} else {
			return (f[ir-1] - f[ir+1])/(2*d[iR]);
		}
	}

	class RangeR {
		public:
			int r_min;
			int r_max;

			RangeR(int r_min, int r_max): r_min(r_min), r_max(r_max) {};
			RangeR(): RangeR(0, -1) {};

			int getRmax(ShGrid const& grid) const {
				if (r_max < 0) {
					return grid.n[iR] + 1 - r_max;
				} else {
					return r_max;
				}
			}
	};

	RangeR getRange(double rmax) const {
		return RangeR(0, (int) rmax/d[iR]);
	}

	class Range {
		public:
			int l_min;
			int l_max;

			int m;

			Range(int l_min, int l_max, int m): l_min(l_min), l_max(l_max), m(m) {}
			Range(int l_min, int l_max): l_min(l_min), l_max(l_max), m(0) {}
			Range(): Range(0, -1, 0) {}
			Range(ShGrid const& g1, ShGrid const& g2): l_min(0), l_max(std::min(g1.n[iL], g2.n[iL])), m(0) {}
	};

	template<class T>
		inline T integrate(std::function<T(int, int)> func, Range range = Range()) const {
			if (range.l_max < 0) {
				range.l_max = n[iL] + 1 - range.l_max;
			}

			range.l_min = std::max(range.l_min, range.m);

			T res = 0.0;
#pragma omp parallel for reduction(+:res) collapse(2)
			for (int il = range.l_min; il < range.l_max; ++il) {
				for (int ir = 0; ir < n[iR]; ++ir) {
					res += func(ir, il);
				}
			}
			return res*d[iR];
		}

private:
	double m_d2[3];
};

class ShNotEqudistantGrid: public ShGrid {
	typedef std::function<double(double)> func_t;

	public:
	ShNotEqudistantGrid(double Rmin, double Rmax, double Ra, double dr_max, int Nl) {
		d[iL] = 1;
		n[iL] = Nl;

		d[iR] = dr_max;

		double A = Rmin/d[iR] - 1.0;
		double xi_max = Rmax - A*Ra;

		n[iR] = static_cast<int>(xi_max / d[iR]);

		init(
				// r = f(xi)
				[=](double xi) -> double {
					return xi + A*Ra*std::tanh(xi/Ra);
				},
				// drdxi
				[=](double xi) -> double {
					return 1.0 + A/std::pow(cosh(xi/Ra), 2);
				},
				// d2rdxi2
				[=](double xi) -> double {
					return -2*A*std::tanh(xi/Ra)/std::pow(cosh(xi/Ra), 2) / Ra;
				}
			);
	}

	void init(func_t f, func_t df, func_t d2f) {
		double dr = d[iR];
		double dr2 = dr*dr;

		m_d2[0] = 1.0/dr2;
		m_d2[1] = -2.0/dr2;
		m_d2[2] = m_d2[0];

		m_d1[0] = -0.5/dr;
		m_d1[1] = 0.0;
		m_d1[2] = 0.5/dr;

		dfdxi = std::shared_ptr<double>(new double[n[iR]]);
		h     = std::shared_ptr<double>(new double[n[iR]]);
		g     = std::shared_ptr<double>(new double[n[iR]]);
		m_r   = std::shared_ptr<double>(new double[n[iR]]);
		m_dr  = std::shared_ptr<double>(new double[n[iR]]);

		for (int ir=0; ir<n[iR]; ir++) {
			double xi = d[iR]*(ir+1);

			m_r.get()[ir] = f(xi);

			dfdxi.get()[ir] = df(xi);

			h.get()[ir] = std::pow(df(xi), -2);
			g.get()[ir] = -std::pow(df(xi), -3)*d2f(xi);
		}

		m_dr.get()[0] = m_r.get()[0];
		for (int ir=1; ir<n[iR]; ir++) {
			m_dr.get()[ir] = m_r.get()[ir] - m_r.get()[ir-1];
		}
	}

	~ShNotEqudistantGrid() {}

	double dr(int ir) const {
		return m_dr.get()[ir];
	}

	double r(int ir) const {
		return m_r.get()[ir];
	}

	double Rmax() const {
		return r(n[iR]-1);
	}

	template <typename T>
	T d_dr(T* f, int ir) const {
		std::function<double(double, double)> const d1[3] = {
			[](double d1, double d2) -> double {
				return - d2*d2*(2*d1 + d2) / (d1*(d1 + d2)*(d1*d1 + d1*d2 + d2*d2));
			},
			[](double d1, double d2) -> double {
				return (d2 - d1)*pow(d1 + d2, 2) / (d1*d2*(d1*d1 + d1*d2 + d2*d2));
			},
			[](double d1, double d2) -> double {
				return d1*d1*(d1 + 2*d2) / (d2*(d1 + d2)*(d1*d1 + d1*d2 + d2*d2));
			}
		};

		double dr1 = this->dr(ir);

		if (ir == 0) {
			double dr2 = this->dr(ir+1);

			return (f[ir+1] - f[ir])/dr2;
		} else if (ir == n[iR] - 1) {
			double dr2 = dr1;

			return d1[0](dr1, dr2)*f[ir-1] + d1[1](dr1, dr2)*f[ir];
		} else {
			double dr2 = this->dr(ir+1);

			return d1[0](dr1, dr2)*f[ir-1] + d1[1](dr1, dr2)*f[ir] + d1[2](dr1, dr2)*f[ir+1];
		}
	}

	template <typename T>
	T d_dr_zero(T* f, int ir) const {
		std::function<double(double, double)> const d1[3] = {
			[](double d1, double d2) -> double {
				return - d2*d2*(2*d1 + d2) / (d1*(d1 + d2)*(d1*d1 + d1*d2 + d2*d2));
			},
			[](double d1, double d2) -> double {
				return (d2 - d1)*pow(d1 + d2, 2) / (d1*d2*(d1*d1 + d1*d2 + d2*d2));
			},
			[](double d1, double d2) -> double {
				return d1*d1*(d1 + 2*d2) / (d2*(d1 + d2)*(d1*d1 + d1*d2 + d2*d2));
			}
		};

		double dr1 = this->dr(ir);

		if (ir == 0) {
			double dr2 = this->dr(ir+1);

			return (d1[0](dr1, dr2) + d1[1](dr1, dr2))*f[ir] + d1[2](dr1, dr2)*f[ir+1];
		} else if (ir == n[iR] - 1) {
			double dr2 = dr1;

			return d1[0](dr1, dr2)*f[ir-1] + d1[1](dr1, dr2)*f[ir];
		} else {
			double dr2 = this->dr(ir+1);

			return d1[0](dr1, dr2)*f[ir-1] + d1[1](dr1, dr2)*f[ir] + d1[2](dr1, dr2)*f[ir+1];
		}
	}

	/*
	 * Коэффициенты трехдиагональной матрицы оператора d^2/dr^2 
	 */
	double d2(int ir, int i) const {
		return h.get()[ir]*m_d2[i] + g.get()[ir]*m_d1[i];
	}

	double J(int ir, int il) const {
		return dfdxi.get()[ir];
	}


	template<class T>
	inline T integrate(std::function<T(int, int)> func, Range range = Range()) const {
		if (range.l_max < 0) {
			range.l_max = n[iL] + 1 - range.l_max;
		}

		range.l_min = std::max(range.l_min, range.m);

		T res = 0.0;
#pragma omp parallel for reduction(+:res) collapse(2)
		for (int il = range.l_min; il < range.l_max; ++il) {
			for (int ir = 0; ir < n[iR]; ++ir) {
				res += func(ir, il)*J(ir, il);
			}
		}
		return res*d[iR];
	}

	std::shared_ptr<double> m_dr;

	private:
	double m_d2[3];
	double m_d1[3];

	// d^2/dr^2 = h(\xi) d2/d\xi^2 + g(\xi) d/d\xi
	std::shared_ptr<double> dfdxi;

	std::shared_ptr<double> h;
	std::shared_ptr<double> g;

	std::shared_ptr<double> m_r;
};

class ShNotEqudistantGrid3D: public ShNotEqudistantGrid {
	public:
		ShNotEqudistantGrid3D(double Rmin, double Rmax, double Ra, double dr_max, int Nl): ShNotEqudistantGrid(Rmin, Rmax, Ra, dr_max, Nl) {}

		int size() const {
			return n[iR]*n[iL]*n[iL];
		}

		int index(int ir, int il, int im) const {
			return ir + (im - il)*n[iR] + il*il*n[iR];
		}

		class Range {
			public:
				Range(int l_min, int l_max, int m_shift): l_min(l_min), l_max(l_max), m_shift(m_shift) {}
				Range(int l_min, int l_max): l_min(l_min), l_max(l_max), m_shift(0) {}
				Range(ShNotEqudistantGrid3D const& g1, ShNotEqudistantGrid3D const& g2): l_min(0), l_max(std::min(g1.n[iL], g2.n[iL])), m_shift(0) {}
				Range(): Range(0, -1, 0) {}

				int l_min;
				int l_max;

				int m_shift;
		};

		template<class T>
			inline T integrate(std::function<T(int, int, int)> func, Range range = Range()) const {
				T res = 0.0;

				int m_min = 0;
				int m_max = 0;

				if (range.m_shift > 0) {
					m_max =  range.m_shift;
				} else if (range.m_shift < 0) {
					m_min = -range.m_shift;
				}

				if (range.l_max < 0) {
					range.l_max = n[iL];
				}

//#pragma omp parallel for reduction(+:res) collapse(3)
				for (int il = range.l_min; il < range.l_max; ++il) {
					for (int im = -il+m_min; im < il-m_max; im++) {
						for (int ir = 0; ir < n[iR]; ++ir) {
							res += func(ir, il, im)*J(ir, il);
						}
					}
				}

				return res*d[iR];
			}
};


class CtGrid: public Grid2d {
	public:
		CtGrid(int n[2], double x_max, double y_max) {
			for (int i = 0; i < 2; ++i) {
				this->n[i] = n[i];
			}

			d[iX] = (2*x_max)/(n[iX]-1);
			d[iY] = y_max/n[iY];
		}

		double x(int ix) const {
			return 0.5*d[iX]*(2*ix - n[iX] + 1);
		}

		double y(int iy) const {
			return iy*d[iY];
		}
};
