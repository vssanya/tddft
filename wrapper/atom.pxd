from grid cimport cShGrid, ShGrid
from libcpp.vector cimport vector

cdef extern from "atom.h":
    cdef cppclass cAtom "Atom":
        enum PotentialType:
            POTENTIAL_SMOOTH,
            POTENTIAL_COULOMB

        cppclass State:
            int n
            int l
            int m
            int s
            int countElectrons

        int Z
        vector[State] orbs
        int countOrbs

        State groundState

        PotentialType potentialType
        int countElectrons

        double u(double r)
        double dudz(double r)

    cdef cppclass cAtomCache "AtomCache":
        cAtomCache(cAtom& atom, cShGrid* grid)
        double u(int ir)
        double dudz(int ir)

        cAtom& atom;
        cShGrid* grid;

        double* data_u;
        double* data_dudz;

    cdef cppclass NaAtom:
        NaAtom()

    cdef cppclass NaAtomSGB:
        NaAtomSGB()

    cdef cppclass HAtom:
        HAtom()
    
    cdef cppclass HSmothAtom:
        HSmothAtom()

    cdef cppclass ArAtom:
        ArAtom()

    cdef cppclass ArSaeAtom:
        ArSaeAtom()

    cdef cppclass ArSaeSmoothAtom:
        ArSaeSmoothAtom()

    cdef cppclass NoneAtom:
        NoneAtom()

cdef class Atom:
    cdef cAtom* cdata
    @staticmethod
    cdef Atom from_c(cAtom* atom)

cdef class AtomCache:
    cdef cAtomCache* cdata
    cdef public Atom atom
    cdef public ShGrid grid
