#pragma once

#include "grid.h"
#include "sh_wavefunc.h"

#include "types.h"

#include <vector>

class Atom {
	public:
		enum PotentialType {
			POTENTIAL_SMOOTH,
			POTENTIAL_COULOMB
		};

		struct State {
			int n;
			int l;
			int m;
			int s;
			int countElectrons;

            State(const char* id="1s", int m=0, int countElectrons = 0, int s=0): m(m), s(s) {
				switch (id[1]) {
					case 's':
						l = 0;
						break;
					case 'p':
						l = 1;
						break;
					case 'd':
						l = 2;
					default:
						l = 0;
				}

				switch (id[0]) {
					case '1':
						n = 0;
						break;
					case '2':
						n = 1 - l;
						break;
					case '3':
						n = 2 - l;
						break;
					default:
						l = 0;
				}

                if (countElectrons == 0) {
					if (s == 0) {
						this->countElectrons = 2;
					} else {
						this->countElectrons = 1;
					}
				} else {
					this->countElectrons = countElectrons;
				}
			}
		};

		int Z;

		std::vector<State> orbs;
		int countOrbs;

		State groundState;

		PotentialType potentialType;
		int countElectrons;

        Atom(int Z, std::vector<State> orbs, State groundState, PotentialType type):
			Z(Z),
			orbs(orbs),
			countOrbs(orbs.size()),
            groundState(groundState),
			potentialType(type),
			countElectrons(0)
		{
			for (State s: orbs) {
				countElectrons += s.countElectrons;
			}
		}

        Atom(int Z, std::vector<State> orbs, int groundStateId, PotentialType type): Atom(Z, orbs, orbs[groundStateId], type) {}
        virtual ~Atom() {}

        virtual double u(double r) const = 0;
        virtual double dudz(double r) const = 0;

        int getNumberOrt(int ie) const {
			if (ie == countOrbs) {
				return 1;
			}

			auto next_state = orbs[ie];

			for (int i=ie+1; i<countOrbs; ++i) {
				auto state = orbs[i];
				if (state.l != next_state.l || state.m != next_state.m || state.s != next_state.s) {
					return i - ie;
				}
			}

			return countOrbs - ie;
		}

        bool isSpinPolarized() const {
            return countOrbs > 1 && orbs[0].s != 0;
        }
};

class AtomCache {
	public:
        AtomCache(Atom const& atom, ShGrid const* grid, double* u): atom(atom), grid(grid) {
            const int Nr = grid->n[iR];
            data_u = new double[Nr];
            data_dudz = new double[Nr];

            if (u == nullptr) {
#pragma omp parallel for
                for (int ir=0; ir<Nr; ir++) {
                    double r = grid->r(ir);
                    data_u[ir] = atom.u(r);
                    data_dudz[ir] = atom.dudz(r);
                }
            } else {
                for (int ir=0; ir<Nr; ir++) {
                    data_u[ir] = u[ir] + atom.u(grid->r(ir));
                }

                { int ir = 0;
                    data_dudz[ir] = atom.dudz(grid->r(ir));
                }

                for (int ir=1; ir<Nr-1; ir++) {
                    data_dudz[ir] = (u[ir+1] - u[ir-1])/(2*grid->d[iR]) + atom.dudz(grid->r(ir));
                }

                { int ir = Nr-1;
                    data_dudz[ir] = atom.dudz(grid->r(ir)) / atom.Z;
                }
            }
        }

        AtomCache(Atom const& atom, ShGrid const* grid): AtomCache(atom, grid, nullptr) {}

        ~AtomCache() {
            delete[] data_u;
            delete[] data_dudz;
        }

        double u(int ir) const {
            return data_u[ir];
		}

        double dudz(int ir) const {
            return data_dudz[ir];
		}

        Atom const& atom;
        ShGrid const* grid;

		double* data_u;
		double* data_dudz;
};

class AtomCoulomb: public Atom {
	public:
        AtomCoulomb(int Z, std::vector<State> orbs, int groundStateId): Atom(Z, orbs, groundStateId, POTENTIAL_COULOMB) {}

        double u(double r) const {
			return -Z/r;
		}

        double dudz(double r) const {
			return Z/pow(r, 2);
		}
};

template<int S, int np, std::array<double, S*np> const& C, std::array<double, S*np> const& B>
class AtomSGB: public Atom {
	public:
        AtomSGB(int Z, std::vector<State> orbs, int groundStateId): Atom(Z, orbs, groundStateId, POTENTIAL_COULOMB) {}

        double u(double r) const {
			double res = 0.0;

#pragma unroll
			for (int p=0; p<S; p++) {
#pragma unroll
				for (int k=0; k<np; k++) {
					res += C[k + p*np]*pow(r, p)*exp(-B[k + p*np]*r);
				}
			}

			return - (Z - countElectrons + 1 + (countElectrons-1)*res) / r;
		}

        double dudz(double r) const {
			double res1 = 0.0;
			double res2 = 0.0;

#pragma unroll
			for (int p=0; p<S; p++) {
#pragma unroll
				for (int k=0; k<np; k++) {
					double tmp = C[k + p*np]*exp(-B[k + p*np]*r);
					res1 += tmp*pow(r, p);
					res2 += tmp*(p*pow(r, p-1) - B[k + p*np]*pow(r, p));
				}
			}

			return (Z - countElectrons + 1 + (countElectrons-1)*res1) / (r*r) - (countElectrons-1)*res2/r;
		}
};

class MgAtom: public AtomCoulomb {
public:
    static const std::vector<State> GroundStateOrbs;
    MgAtom(): AtomCoulomb(12, GroundStateOrbs, 2) {}
};

class NaAtom: public AtomCoulomb {
	public:
		static const std::vector<State> GroundStateOrbs;

		NaAtom(): AtomCoulomb(11, GroundStateOrbs, 2) {}
};

extern const std::array<double, 2*3> Na_B;
extern const std::array<double, 2*3> Na_C;

class NaAtomSGB: public AtomSGB<3, 2, Na_C, Na_B> {
	public:
		NaAtomSGB(): AtomSGB(11, NaAtom::GroundStateOrbs, 2) {}
};


class HAtom: public AtomCoulomb {
	public:
		static const std::vector<State> GroundStateOrbs;
		HAtom(): AtomCoulomb(1, GroundStateOrbs, 0) {}
};

class HSmothAtom: public Atom {
	public:
		static constexpr double A = 0.3;
		static constexpr double ALPHA = 2.17;

        HSmothAtom(): Atom(1, HAtom::GroundStateOrbs, 0, POTENTIAL_SMOOTH) {}

        double u(double r) const {
			return -ALPHA*pow(cosh(r/A), -2) - tanh(r/A)/r;
		}

        double dudz(double r) const {
			double const t = tanh(r/A);
			double const s = pow(cosh(r/A), -2);

			return 2.0*ALPHA*t*s/A + t/pow(r, 2) - s/(A*r);
		}
};

class NeAtom: public AtomCoulomb {
	public:
		static const std::vector<State> GroundStateOrbs;
		NeAtom(): AtomCoulomb(10, GroundStateOrbs, 2) {}
};

class ArAtom: public AtomCoulomb {
	public:
		static const std::vector<State> GroundStateOrbs;
		ArAtom(): AtomCoulomb(18, GroundStateOrbs, 4) {}
};

class ArSaeAtom: public Atom {
	public:
		static constexpr double A = 5.4;
		static constexpr double B = 1;
		static constexpr double C = 3.682;

        ArSaeAtom(): Atom(18, ArAtom::GroundStateOrbs, 4, POTENTIAL_COULOMB) {}

        double u(double r) const {
			return - (1.0 + (A*exp(-B*r) + (17 - A)*exp(-C*r)))/r;
		}

        double dudz(double r) const {
			return (1.0 + (  A*exp(-B*r) +   (17 - A)*exp(-C*r)))/(r*r) +
				(B*A*exp(-B*r) + C*(17 - A)*exp(-C*r))/r;
		}
};

class ArSaeSmoothAtom: public Atom {
public:
    static constexpr double A = 0.3;
    static constexpr double ALPHA = 3.88;

    ArSaeSmoothAtom(): Atom(18, ArAtom::GroundStateOrbs, 4, POTENTIAL_SMOOTH) {}

    double u(double r) const {
        return -ALPHA*pow(cosh(r/A), -2) - tanh(r/A)/r;
    }

    double dudz(double r) const {
        double const t = tanh(r/A);
        double const s = pow(cosh(r/A), -2);

        return 2.0*ALPHA*t*s/A + t/pow(r, 2) - s/(A*r);
    }
};


class NoneAtom: public Atom {
public:
    NoneAtom(): Atom(0, {}, State("1s"), POTENTIAL_SMOOTH) {}

    double u(double r) const { return 0.0; }
    double dudz(double r) const { return 0.0; }
};
