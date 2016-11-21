cdef class SGrid:
    def __cinit__(self, int Nr, int Nl, double dr):
        self.data.Nr = Nr
        self.data.Nl = Nl
        self.data.dr = dr
