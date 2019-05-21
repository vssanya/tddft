from grid cimport cShGrid, cShNeGrid, ShGrid, ShNeGrid

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

    cdef cppclass AtomCache[Grid]:
        AtomCache(cAtom& atom, Grid& grid)
        AtomCache(cAtom& atom, Grid& grid, double* u)
        double u(int ir)
        double dudz(int ir)

        cAtom& atom;
        cShGrid& grid;

        double* data_u;
        double* data_dudz;

    cdef cppclass MgAtom:
        MgAtom()

    cdef cppclass NaAtom:
        NaAtom()

    cdef cppclass NaAtomSGB:
        NaAtomSGB()

    cdef cppclass HAtom:
        HAtom()

    cdef cppclass HSmothAtom:
        HSmothAtom()

    cdef cppclass HeAtom:
        HeAtom()

    cdef cppclass NeAtom:
        NeAtom()

    cdef cppclass FNegativeIon:
        FNegativeIon()

    cdef cppclass ArAtom:
        ArAtom()

    cdef cppclass KrAtom:
        KrAtom()

    cdef cppclass XeAtom:
        XeAtom()

    cdef cppclass ArSaeAtom:
        ArSaeAtom()

    cdef cppclass ArSaeSmoothAtom:
        ArSaeSmoothAtom()

    cdef cppclass NoneAtom:
        NoneAtom()

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
