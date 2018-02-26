#include "atom.h"
#include "orbitals.h"

#include <array>


const std::vector<Atom::State> HAtom::GroundStateOrbs = {
	State("1s")
}; 

const std::vector<Atom::State> MgAtom::GroundStateOrbs = {
    State("1s"), State("2s"), State("3s"),
    State("2p"), State("2p", 1, 4)
};

const std::vector<Atom::State> NaAtom::GroundStateOrbs = {
	State("1s", 0, 1, -1), State("2s", 0, 1, -1), State("3s",  0, 1, -1),
	State("2p", 0, 1, -1), State("2p", 1, 1, -1), State("2p", -1, 1, -1),
	State("1s", 0, 1,  1), State("2s", 0, 1,  1),
	State("2p", 0, 1,  1), State("2p", 1, 1,  1), State("2p",  1, 1,  1)
};

const std::vector<Atom::State> ArAtom::GroundStateOrbs = {
	State("1s"), State("2s"), State("3s"), // m = 0
	State("2p"), State("3p"),              // m = 0
	State("2p", 1, 4), State("3p", 1, 4)   // m = +- 1
};

constexpr std::array<double, 2*3> Na_B {{
    6.46644991, 2.03040457,
    9.07195947, 1.22049052,
    3.66561470, 3.88900584
}};

constexpr std::array<double, 2*3> Na_C {{
    0.35071677, 0.64928323,
    1.00486813, -0.05093639,
    1.06629058, 0.70089565,
}};

/*
extern constexpr std::array<double, 2*5> rb_B {{
	7.83077875,  2.75163799,
	4.30010258,  0.0,
	43.31975597, 0.0,
	2.93818679,  0.0,
	4.97097146,  0.0
}};

extern constexpr std::array<double, 2*5> rb_C {{
	0.81691787,   0.18308213,
	2.53670563,   0.0,
	-19.56508990, 0.0,
	1.06320272,   0.0,
	-0.99934358,  0.0
}};

double atom_u_rb_sae(Atom const* atom, ShGrid const* grid, int ir) {
	return potential_sgb_u<37, 37, 5, 2, rb_C, rb_B>(grid->r(ir));
}

double atom_dudz_rb_sae(Atom const* atom, ShGrid const* grid, int ir) {
	return potential_sgb_dudz<37, 37, 5, 2, rb_C, rb_B>(grid->r(ir));
}
*/


void atom_hydrogen_ground(ShWavefunc* wf) {
	assert(wf->m == 0);
	// l = 0
	{
		int const il = 0;
		for (int ir = 0; ir < wf->grid->n[iR]; ++ir) {
            double r = wf->grid->r(ir);
			(*wf)(ir, il) = 2*r*exp(-r);
		}
	}

	for (int il = 1; il < wf->grid->n[iL]; ++il) {
		for (int ir = 0; ir < wf->grid->n[iR]; ++ir) {
			(*wf)(ir, il) = 0.0;
		}
	}
}
