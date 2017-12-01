#pragma once


#ifdef __cplusplus
extern "C" {
#endif

__attribute__((pure))
double Fn(double n0, double g, double pn, double pz, double a, double phi);

__attribute__((pure))
double Qc(double g, double w);

__attribute__((pure))
double Ip_Up(double g, double a);

__attribute__((pure))
double C1(double w, double g, double a);

__attribute__((pure))
double pn(int n, double n0, double g);

__attribute__((pure))
int n_min(double n0, double g);

__attribute__((pure))
double sfa_djdt(double E, double w, double a, double phi);

__attribute__((pure))
double sfa_dwdt(double E, double w, double a, double phi);

#ifdef __cplusplus
}
#endif
