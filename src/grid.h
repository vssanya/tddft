#pragma once

#include <stdlib.h>
#include <assert.h>
#include <math.h>


#define check_index(space, index) assert(index >= 0 || index < n[space])

/*!
 * \brief Grid for 2D space
 * */
class Grid2d {
public:
    int    n[2]; //!< is counts of points
    double d[2]; //!< is steps

    size_t size() const { return n[0]*n[1]; }

    size_t index(int i[2]) const {
        check_index(0, i[0]);
        check_index(1, i[1]);

        return i[0] + i[1]*n[0];
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
    SpGrid2d(int n[2], double Rmax) {
        for (int i = 0; i < 2; ++i) {
            this->n[i] = n[i];
        }

        d[iR] = Rmax/n[iR];
        d[iC] = 2.0/(n[iC]-1);
    }

    double r(int ir) const {
        check_index(iR, ir);
        return d[iR]*(ir+1);
    }

    double c(int ic) const {
        check_index(iC, ic);
        return d[iC]*ic - 1.0;
    }
};

class SpGrid: public Grid3d {
public:
    SpGrid(int n[3], double Rmax) {
        for (int i=0; i<3; ++i) {
            this->n[i] = n[i];
        }

        d[iR] = Rmax/n[iR];
        d[iC] = 2.0/(n[iC]-1.0);
        d[iP] = 2.0*M_PI/n[iP];
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
    ShGrid(int n[2], double Rmax) {
        for (int i=0; i<2; ++i) {
            this->n[i] = n[i];
        }

        d[iR] = Rmax/n[iR];
        d[iL] = 1;
    }

    double r(int ir) const {
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
