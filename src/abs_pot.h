#pragma once

#include "grid.h"
#include <array>

double uabs(ShGrid const* grid, int ir, int il, int im) __attribute__((pure));

__attribute__((pure))
double mask_core(ShGrid const* grid, int ir, int il, int im);

class Uabs {
public:
    virtual ~Uabs() {}

    virtual double u(ShGrid const& grid, double r) const = 0;
};

class UabsZero: public Uabs {
public:
    double u(ShGrid const& grid, double r) const {
        return 0.0;
    }
};

class UabsMultiHump: public Uabs {
public:
    UabsMultiHump(double l_min, double l_max);
    double u(ShGrid const& grid, double r) const;

    std::array<double, 2> l;
    std::array<double, 2> a;
};

class UabsCache {
public:
    UabsCache(Uabs const& uabs, ShGrid const& grid, double* u = nullptr);
    ~UabsCache();

    inline double u(int ir) const {
        return data[ir];
    }

    Uabs const& uabs;
    ShGrid const& grid;

    double* data;
};
