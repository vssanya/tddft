#pragma once

#include "grid.h"

#include <vector>
#include <functional>


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

class Hump {
public:
	std::function<double(double)> u;

	double e0;
	double e2;

    Hump(): u(nullptr), e0(0.0), e2(0.0) {}
	Hump(std::function<double(double)> u, double e0, double e2):
		u(u), e0(e0), e2(e2) {}

	double e_opt(double k) const {
		return e0 + e2*k*k;
	}

	double a_opt(double l) const {
		return e_opt(2*M_PI) / (l*l);
	}
};

const Hump CosHump([](double x) {
	if (fabs(x) >= 1) {
		return 0.0;
	}

	return pow(cos(M_PI * x / 2.0), 2);
}, 2.15, 0.65);

const Hump PTHump([](double x) {
    double const alpha = 2.0*acosh(sqrt(2.0));
	return pow(cosh(alpha*x), -2);
}, 3.7, 1.14);

class UabsMultiHump: public Uabs {
private:
    void init(double l_min, double l_max, double shift);
public:
    UabsMultiHump(double l_min, double l_max, std::vector<Hump> humps, double shift = 0.0);
    UabsMultiHump(double l_min, double l_max, int n, double shift = 0.0);
    double u(ShGrid const& grid, double r) const;

	double getHumpAmplitude(int i) const;
	void setHumpAmplitude(int i, double value);

    std::vector<Hump> humps;
    std::vector<double> l;
    std::vector<double> a;
    std::vector<double> shifts;
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
