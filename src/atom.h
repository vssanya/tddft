#pragma once

#include "grid.h"
#include "sh_wavefunc.h"

#include "types.h"

#include <vector>
#include <algorithm>
#include <cstdlib>


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
			int shell;

			State(int n, int l, int m, int s, int countElectrons, int shell):
				n(n), l(l), m(m), s(0), 
				countElectrons(countElectrons), shell(shell)
			{}

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
						break;
					default:
						l = 0;
				}

				char tmp[2] = {id[0], '\0'};
				shell = atoi(tmp) - 1;
				n = shell - l;

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

		Atom():
			Z(0),
			orbs(),
			countOrbs(0),
            groundState(),
			potentialType(POTENTIAL_SMOOTH),
			countElectrons(0)
		{
		}

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

		int getNumberShell() const {
			return std::max_element(orbs.begin(), orbs.end(),
					[](const State& a, const State& b) {
						return a.shell < b.shell;
					})->shell + 1;
		}

		void getActiveOrbs(int shell, bool activeOrbs[]) const {
			for (int i=0; i<countOrbs; i++) {
				activeOrbs[i] = orbs[i].shell <= shell;
			}
		}

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

		int getMmax() const {
			return std::max_element(orbs.begin(), orbs.end(), [](const State& s1, const State& s2) {
					return s1.m < s2.m;
					})->m;
		}

        bool isSpinPolarized() const {
            return countOrbs > 1 && orbs[0].s != 0;
        }
};

template<class Grid>
class AtomCache {
	public:
        AtomCache(Atom const& atom, Grid const& grid, double* u, int N = -1);
        AtomCache(Atom const& atom, Grid const& grid): AtomCache(atom, grid, nullptr) {}
		~AtomCache();

        double u(int ir) const { return data_u[ir]; }
        double dudz(int ir) const { return data_dudz[ir]; }

		double* getGPUDataU();
		double* getGPUDatadUdz();

        Atom const& atom;
        Grid const& grid;

		double* data_u;
		double* data_dudz;


	private:
		double* gpu_data_u;
		double* gpu_data_dudz;
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

class Fulleren: public Atom {
	const double Ri = 5.3;
	const double R0 = 8.1;
	const double V0 = 0.68;
	const double k = 250.0/(pow(R0, 3) - pow(Ri, 3));

	public:
		Fulleren() {
			Z = 250;

			countOrbs = 70;
			orbs.resize(countOrbs);

			int i = 0;
			for (int l = 0; l < 10; ++l) {
				for (int m=0; m<=l; m++) {
					for (int n = 0; n < (l < 5 ? 2 : 1); n++) {
						orbs[i] = State(n, l, m, 0, m==0 ? 2 : 4, n);
						i++;
					}
				}
			}


			groundState = orbs[0];
		}
	
		double u(double r) const {
			if (r <= Ri) {
				return - 1.5*k*(R0*R0 - Ri*Ri);
			} else if (r >= R0) {
				return -Z/r;
			} else {
				return -k*(1.5*R0*R0 - (0.5*r*r + pow(Ri, 3)/r)) - V0;
			}
		}

		double dudz(double r) const {
			if (r <= Ri) {
				return 0.0;
			} else if (r >= R0) {
				return Z/pow(r, 2);
			} else {
				return k*(r - pow(Ri, 3)/pow(r,2));
			}
		}
};

template<int S, int np, std::array<double, S*np> const& C, std::array<double, S*np> const& B>
class AtomSGB: public Atom {
	public:
        AtomSGB(int Z, std::vector<State> orbs, int groundStateId): Atom(Z, orbs, groundStateId, POTENTIAL_COULOMB) {}

        double u(double r) const {
			double res = 0.0;

			for (int p=0; p<S; p++) {
				for (int k=0; k<np; k++) {
					res += C[k + p*np]*pow(r, p)*exp(-B[k + p*np]*r);
				}
			}

			return - (Z - countElectrons + 1 + (countElectrons-1)*res) / r;
		}

        double dudz(double r) const {
			double res1 = 0.0;
			double res2 = 0.0;

			for (int p=0; p<S; p++) {
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

class RbAtom: public AtomCoulomb {
	public:
		static const std::vector<State> GroundStateOrbs;

		RbAtom(): AtomCoulomb(37, GroundStateOrbs, 4) {}
};

extern const std::array<double, 2*5> Rb_B;
extern const std::array<double, 2*5> Rb_C;
class RbAtomSGB: public AtomSGB<5, 2, Rb_C, Rb_B> {
	public:
		RbAtomSGB(): AtomSGB(37, RbAtom::GroundStateOrbs, 4) {}
};

class LiAtom: public AtomCoulomb {
	public:
		static const std::vector<State> GroundStateOrbs;

		LiAtom(): AtomCoulomb(3, GroundStateOrbs, 2) {}
};

extern const std::array<double, 2*3> Li_B;
extern const std::array<double, 2*3> Li_C;

class LiAtomSGB: public AtomSGB<3, 2, Li_C, Li_B> {
	public:
		LiAtomSGB(): AtomSGB(3, LiAtom::GroundStateOrbs, 2) {}
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

class HeAtom: public AtomCoulomb {
	public:
		static const std::vector<State> GroundStateOrbs;
		HeAtom(): AtomCoulomb(2, GroundStateOrbs, 0) {}
};

extern const std::array<double, 2*3> He_B;
extern const std::array<double, 2*3> He_C;

class HeAtomSGB: public AtomSGB<3, 2, He_C, He_B> {
	public:
		HeAtomSGB(): AtomSGB(2, HeAtom::GroundStateOrbs, 0) {}
};


class NeAtom: public AtomCoulomb {
	public:
		static const std::vector<State> GroundStateOrbs;
		NeAtom(): AtomCoulomb(10, GroundStateOrbs, 2) {}
};

extern const std::array<double, 2*3> Ne_B;
extern const std::array<double, 2*3> Ne_C;

class NeAtomSGB: public AtomSGB<3, 2, Ne_C, Ne_B> {
	public:
		NeAtomSGB(): AtomSGB(10, NeAtom::GroundStateOrbs, 2) {}
};

class FNegativeIon: public AtomCoulomb {
	public:
		FNegativeIon(): AtomCoulomb(9, NeAtom::GroundStateOrbs, 2) {}
};

class FNegativeSaeIon: public Atom {
	public:
		static constexpr double a1 = 5.137;
		static constexpr double a2 = 3.863;
		static constexpr double alpha1 = 1.288;
		static constexpr double alpha2 = 3.545;

		FNegativeSaeIon(): Atom(9, NeAtom::GroundStateOrbs, 2, POTENTIAL_COULOMB) {}

        double u(double r) const {
			return - (a1*exp(-alpha1*r) + a2*exp(-alpha2*r))/r;
		}

        double dudz(double r) const {
			return (alpha1*a1*exp(-alpha1*r) + alpha2*a2*exp(-alpha2*r))/r - u(r)/r;
		}
};

class ArAtom: public AtomCoulomb {
	public:
		static const std::vector<State> GroundStateOrbs;
		ArAtom(): AtomCoulomb(18, GroundStateOrbs, 4) {}
};

class KrAtom: public AtomCoulomb {
	public:
		static const std::vector<State> GroundStateOrbs;
		KrAtom(): AtomCoulomb(36, GroundStateOrbs, 6) {}
};

class XeAtom: public AtomCoulomb {
	public:
		static const std::vector<State> GroundStateOrbs;
		XeAtom(): AtomCoulomb(54, GroundStateOrbs, 8) {}
};

class CsAtom: public AtomCoulomb {
	public:
		static const std::vector<State> GroundStateOrbs;
		CsAtom(): AtomCoulomb(55, GroundStateOrbs, 5) {}
};

class CsPAtom: public AtomCoulomb {
	public:
		CsPAtom(): AtomCoulomb(55, XeAtom::GroundStateOrbs, 8) {}
};

class Ba2PAtom: public AtomCoulomb {
	public:
		Ba2PAtom(): AtomCoulomb(56, XeAtom::GroundStateOrbs, 8) {}
};

class BaPAtom: public AtomCoulomb {
	public:
		static const std::vector<State> GroundStateOrbs;
		BaPAtom(): AtomCoulomb(56, GroundStateOrbs, 8) {}
};

class BaAtom: public AtomCoulomb {
	public:
		static const std::vector<State> GroundStateOrbs;
		BaAtom(): AtomCoulomb(56, GroundStateOrbs, 5) {}
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


extern const std::array<double, 2*3> Ar_B;
extern const std::array<double, 2*3> Ar_C;

class ArAtomSGB: public AtomSGB<3, 2, Ar_C, Ar_B> {
	public:
		ArAtomSGB(): AtomSGB(18, ArAtom::GroundStateOrbs, 2) {}
};


class NoneAtom: public Atom {
public:
    NoneAtom(): Atom(0, {}, State("1s"), POTENTIAL_SMOOTH) {}

    double u(double r) const { return 0.0; }
    double dudz(double r) const { return 0.0; }
};


class ShortAtom: public Atom {
	double c;
	double n;

	public:
		ShortAtom(double c, double n): Atom(0, {}, State("1s"), POTENTIAL_SMOOTH), c(c), n(n) {}

		double u(double r) const {
			//return - c * 0.5 * n / sqrt(M_PI) * exp(-pow(n*r, 2));
			return - c / pow(cosh(n*r), 2);
		}

		double dudz(double r) const {
			//return  -2*n*n*r*u(r);
			return -2*n*tanh(n*r)*u(r);
		}
};
