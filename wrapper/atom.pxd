from grid cimport cShGrid, cShNeGrid, ShGrid, ShNeGrid

from libcpp cimport bool as bool_t
from libcpp.vector cimport vector


cdef extern from "atom.h":
    cdef cppclass cAtom "Atom":
        enum PotentialType:
            POTENTIAL_SMOOTH,
            POTENTIAL_COULOMB

        cppclass cState "State":
            int n
            int l
            int m
            int s
            int countElectrons

            cState(char* id, int m, int countElectrons, int s)
            cState()

        int Z
        vector[cState] orbs
        int countOrbs

        cState groundState

        PotentialType potentialType
        int countElectrons

        double u(double r)
        double dudz(double r)

        int getNumberOrt(int ie)
        int getNumberShell()
        void getActiveOrbs(int shell, bool_t* activeOrbs)

    cdef cppclass AtomCache[Grid]:
        AtomCache(cAtom& atom, Grid& grid)
        AtomCache(cAtom& atom, Grid& grid, double* u, int N)
        double u(int ir)
        double dudz(int ir)

        cAtom& atom;
        cShGrid& grid;

        double* data_u;
        double* data_dudz;

    cdef cppclass ShortAtom:
        ShortAtom(double c, double n)

    cdef cppclass MgAtom:
        MgAtom()

    cdef cppclass NaAtom:
        NaAtom()

    cdef cppclass NeAtomSGB:
        NeAtomSGB()

    cdef cppclass ArAtomSGB:
        ArAtomSGB()

    cdef cppclass NaAtomSGB:
        NaAtomSGB()

    cdef cppclass RbAtomSGB:
        RbAtomSGB()

    cdef cppclass HAtom:
        HAtom()

    cdef cppclass HSmothAtom:
        HSmothAtom()

    cdef cppclass HeAtom:
        HeAtom()

    cdef cppclass HeAtomSGB:
        HeAtomSGB()

    cdef cppclass LiAtom:
        LiAtom()

    cdef cppclass LiAtomSGB:
        LiAtomSGB()

    cdef cppclass NeAtom:
        NeAtom()

    cdef cppclass FNegativeIon:
        FNegativeIon()

    cdef cppclass FNegativeSaeIon:
        FNegativeSaeIon()

    cdef cppclass ArAtom:
        ArAtom()

    cdef cppclass KrAtom:
        KrAtom()

    cdef cppclass XeAtom:
        XeAtom()

    cdef cppclass CsPAtom:
        CsPAtom()

    cdef cppclass Ba2PAtom:
        Ba2PAtom()

    cdef cppclass BaPAtom:
        BaPAtom()

    cdef cppclass BaAtom:
        BaAtom()

    cdef cppclass ArSaeAtom:
        ArSaeAtom()

    cdef cppclass ArSaeSmoothAtom:
        ArSaeSmoothAtom()

    cdef cppclass NoneAtom:
        NoneAtom()

    cdef cppclass Fulleren:
        Fulleren()

cdef class State:
    cdef cAtom.cState cdata
    @staticmethod
    cdef State from_c(cAtom.cState state)

cdef class Atom:
    cdef cAtom* cdata
    cdef public State ground_state
    cdef str name

    @staticmethod
    cdef Atom from_c(cAtom* atom, str name)

cdef class ShAtomCache:
    cdef AtomCache[cShGrid]* cdata
    cdef public ShGrid grid
    cdef public Atom atom

cdef class ShNeAtomCache:
    cdef AtomCache[cShNeGrid]* cdata
    cdef public ShNeGrid grid
    cdef public Atom atom
