#pragma once


double Fn(double n0, double g, double pn, double pz, double a, double phi);
double Qc(double g, double w);
double Ip_Up(double g, double a);
double C1(double w, double g, double a);
double pn(int n, double n0, double g);
int n_min(double n0, double g);

double sfa_djdt(double E, double w, double a, double phi);
double sfa_dwdt(double E, double w, double a, double phi);
