#include "hydrogen.h"

#include "math.h"


double hydrogen_U(double r) {
	return -1.0/r;
}

double hydrogen_dUdz(double r) {
	return 1.0/pow(r, 2);
}
