#pragma once

#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <functional>


#define check_index(space, index) assert(index >= 0 || index < n[space])

class Range {
	public:
		int start;
		int end;

		Range(int start, int end): start(start), end(end) {}
		Range() = default;
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

	Grid2d(): n{0}, d{1.0} {}
	Grid2d(int nx, int ny): n{nx, ny}, d{1.0,1.0} {}

	Range getFullRange(int index) const {
		return Range(0, n[index]);
	}

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

    size_t size() const { return n[0]*n[1]*n[2]; }

	Range getFullRange(int index) const {
		return Range(0, n[index]);
	}

    size_t index(int i[3]) const {
        check_index(0, i[0]);
        check_index(1, i[1]);
        check_index(2, i[2]);

        return i[0] + i[1]*n[0] + i[2]*n[1]*n[0];
    }
};

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
    iC = 1, //!< is index for \f$\cos\theta\f$
    iP = 2, //!< is index for \f$\phi\f$
};

/*!
 * \brief Indexes for spherical harmonics
 * */
enum {
    iL = 1, //!< is index for l
    iM = 2, //!< is index for m
};

class SpGrid2d: public Grid2d {
public:
	SpGrid2d(): Grid2d() {}
    SpGrid2d(int const n[2], double Rmax) {
        for (int i = 0; i < 2; ++i) {
            this->n[i] = n[i];
        }

        d[iR] = Rmax/n[iR];
        d[iC] = M_PI/(n[iC]-1.0);
    }

    double r(int ir) const {
        check_index(iR, ir);
        return d[iR]*(ir+1);
    }

    int l(int il) const {
        check_index(iL, il);
        return il;
    }

    double theta(int ic) const {
        check_index(iC, ic);
        return d[iC]*ic;
    }

	double J(int ir, int il) const {
		return 1.0;
	}
};

class SpGrid: public Grid3d {
public:
    double dtheta;

    SpGrid(int n[3], double Rmax) {
        for (int i=0; i<3; ++i) {
            this->n[i] = n[i];
        }

        d[iR] = Rmax/n[iR];
        d[iC] = 2.0/(n[iC]-1.0);
        d[iP] = 2.0*M_PI/n[iP];

        dtheta = M_PI/(n[iC]-1.0);
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

    double c(int ic) const {
        check_index(iC, ic);
        return d[iC]*ic - 1.0;
    }

    double theta(int ic) const {
        check_index(iC, ic);
        return dtheta*ic;
    }

    double phi(int ip) const {
        check_index(iP, ip);
        return d[iP]*ip;
    }

    int ir(double r) const {
        return (int) (r/d[iR]) - 1;
    }

    int ic(double c) const {
        return (int) ((c + 1.0) / d[iC]);
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

	Range getRange(double rmax) const {
		return Range(0, (int) (rmax/d[iR]));
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

	ShNotEqudistantGrid(const ShNotEqudistantGrid& grid) = delete;

	void init(func_t f, func_t df, func_t d2f) {
		double dr = d[iR];
		double dr2 = dr*dr;

		m_d2[0] = 1.0/dr2;
		m_d2[1] = -2.0/dr2;
		m_d2[2] = m_d2[0];

		m_d1[0] = -0.5/dr;
		m_d1[1] = 0.0;
		m_d1[2] = 0.5/dr;

		dfdxi = new double[n[iR]];

		h = new double[n[iR]];
		g = new double[n[iR]];

		m_r = new double[n[iR]];
		m_dr = new double[n[iR]];

		for (int ir=0; ir<n[iR]; ir++) {
			double xi = d[iR]*(ir+1);

			m_r[ir] = f(xi);

			dfdxi[ir] = df(xi);

			h[ir] = std::pow(df(xi), -2);
			g[ir] = -std::pow(df(xi), -3)*d2f(xi);
		}

		m_dr[0] = m_r[0];
		for (int ir=1; ir<n[iR]; ir++) {
			m_dr[ir] = m_r[ir] - m_r[ir-1];
		}
	}

	~ShNotEqudistantGrid() {
		delete[] dfdxi;
		delete[] h;
		delete[] g;
		delete[] m_r;
		delete[] m_dr;
	}

	double dr(int ir) const {
		return m_dr[ir];
	}

	double r(int ir) const {
		return m_r[ir];
	}

	double Rmax() const {
		return r(n[iR]-1);
	}

	Range getRange(double rmax) const {
		for (int i=0; i<n[iR]; i++) {
			if (r(i) > rmax) {
				return Range(0, i);
			}
		}

		return getFullRange(iR);
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

			return d1[0](dr1, dr2)*f[ir] + d1[1](dr1, dr2)*f[ir] + d1[2](dr1, dr2)*f[ir+1];
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
		return h[ir]*m_d2[i] + g[ir]*m_d1[i];
	}

	double J(int ir, int il) const {
		return dfdxi[ir];
	}

	double* m_dr;

	private:
	double m_d2[3];
	double m_d1[3];

	// d^2/dr^2 = h(\xi) d2/d\xi^2 + g(\xi) d/d\xi
	double* dfdxi;

	double* h;
	double* g;

	double* m_r;
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

		template<class T>
			inline T integrate(std::function<T(int, int, int)> func, int l_max, int l_min = 0, int m_shift = 0) const {
				T res = 0.0;

				int m_min = 0;
				int m_max = 0;

				if (m_shift > 0) {
					m_max =  m_shift;
				} else if (m_shift < 0) {
					m_min = -m_shift;
				}

//#pragma omp parallel for reduction(+:res) collapse(3)
				for (int il = l_min; il < l_max; ++il) {
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
