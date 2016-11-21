cdef extern from "hydrogen.h":
    double hydrogen_U(double r)
    double hydrogen_dUdz(double r)
