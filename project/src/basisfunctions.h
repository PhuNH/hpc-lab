/** This file is generated. Do not edit. */
#ifndef BASISFUNCTIONS_H_
#define BASISFUNCTIONS_H_
#include <cmath>
static double basisFunction0(double xi, double eta) {
  double phi = 1;
  return phi;
}
static double basisFunction1(double xi, double eta) {
  double phi = 2 * xi - 1;
  return phi;
}
static double basisFunction2(double xi, double eta) {
  double phi = 2 * eta - 1;
  return phi;
}
static double basisFunction3(double xi, double eta) {
  double phi = 6 * xi * xi - 6 * xi + 1;
  return phi;
}
static double basisFunction4(double xi, double eta) {
  double phi = 4 * eta * xi - 2 * eta - 2 * xi + 1;
  return phi;
}
static double basisFunction5(double xi, double eta) {
  double phi = 6 * eta * eta - 6 * eta + 1;
  return phi;
}
static double basisFunction6(double xi, double eta) {
  double phi = 0.20e2 * pow(xi, 0.3e1) - 0.30e2 * xi * xi + 0.12e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction7(double xi, double eta) {
  double phi = 12 * eta * xi * xi - 12 * eta * xi - 6 * xi * xi + 2 * eta + 6 * xi - 1;
  return phi;
}
static double basisFunction8(double xi, double eta) {
  double phi = 12 * eta * eta * xi - 6 * eta * eta - 12 * eta * xi + 6 * eta + 2 * xi - 1;
  return phi;
}
static double basisFunction9(double xi, double eta) {
  double phi = 0.20e2 * pow(eta, 0.3e1) - 0.30e2 * eta * eta + 0.12e2 * eta - 0.1e1;
  return phi;
}
static double basisFunction10(double xi, double eta) {
  double phi = 0.70e2 * pow(xi, 0.4e1) - 0.140e3 * pow(xi, 0.3e1) + 0.90e2 * xi * xi - 0.20e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction11(double xi, double eta) {
  double phi = 0.40e2 * eta * pow(xi, 0.3e1) - 0.60e2 * eta * xi * xi - 0.20e2 * pow(xi, 0.3e1) + 0.24e2 * eta * xi + 0.30e2 * xi * xi - 0.2e1 * eta - 0.12e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction12(double xi, double eta) {
  double phi = 36 * eta * eta * xi * xi - 36 * eta * eta * xi - 36 * eta * xi * xi + 6 * eta * eta + 36 * eta * xi + 6 * xi * xi - 6 * eta - 6 * xi + 1;
  return phi;
}
static double basisFunction13(double xi, double eta) {
  double phi = 0.40e2 * pow(eta, 0.3e1) * xi - 0.20e2 * pow(eta, 0.3e1) - 0.60e2 * eta * eta * xi + 0.30e2 * eta * eta + 0.24e2 * eta * xi - 0.12e2 * eta - 0.2e1 * xi + 0.1e1;
  return phi;
}
static double basisFunction14(double xi, double eta) {
  double phi = 0.70e2 * pow(eta, 0.4e1) - 0.140e3 * pow(eta, 0.3e1) + 0.90e2 * eta * eta - 0.20e2 * eta + 0.1e1;
  return phi;
}
static double basisFunction15(double xi, double eta) {
  double phi = 0.252e3 * pow(xi, 0.5e1) - 0.630e3 * pow(xi, 0.4e1) + 0.560e3 * pow(xi, 0.3e1) - 0.210e3 * xi * xi + 0.30e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction16(double xi, double eta) {
  double phi = 0.140e3 * eta * pow(xi, 0.4e1) - 0.280e3 * eta * pow(xi, 0.3e1) - 0.70e2 * pow(xi, 0.4e1) + 0.180e3 * eta * xi * xi + 0.140e3 * pow(xi, 0.3e1) - 0.40e2 * eta * xi - 0.90e2 * xi * xi + 0.2e1 * eta + 0.20e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction17(double xi, double eta) {
  double phi = 0.120e3 * eta * eta * pow(xi, 0.3e1) - 0.180e3 * eta * eta * xi * xi - 0.120e3 * eta * pow(xi, 0.3e1) + 0.72e2 * eta * eta * xi + 0.180e3 * eta * xi * xi + 0.20e2 * pow(xi, 0.3e1) - 0.6e1 * eta * eta - 0.72e2 * eta * xi - 0.30e2 * xi * xi + 0.6e1 * eta + 0.12e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction18(double xi, double eta) {
  double phi = 0.120e3 * pow(eta, 0.3e1) * xi * xi - 0.120e3 * pow(eta, 0.3e1) * xi - 0.180e3 * eta * eta * xi * xi + 0.20e2 * pow(eta, 0.3e1) + 0.180e3 * eta * eta * xi + 0.72e2 * eta * xi * xi - 0.30e2 * eta * eta - 0.72e2 * eta * xi - 0.6e1 * xi * xi + 0.12e2 * eta + 0.6e1 * xi - 0.1e1;
  return phi;
}
static double basisFunction19(double xi, double eta) {
  double phi = 0.140e3 * pow(eta, 0.4e1) * xi - 0.70e2 * pow(eta, 0.4e1) - 0.280e3 * pow(eta, 0.3e1) * xi + 0.140e3 * pow(eta, 0.3e1) + 0.180e3 * eta * eta * xi - 0.90e2 * eta * eta - 0.40e2 * eta * xi + 0.20e2 * eta + 0.2e1 * xi - 0.1e1;
  return phi;
}
static double basisFunction20(double xi, double eta) {
  double phi = 0.252e3 * pow(eta, 0.5e1) - 0.630e3 * pow(eta, 0.4e1) + 0.560e3 * pow(eta, 0.3e1) - 0.210e3 * eta * eta + 0.30e2 * eta - 0.1e1;
  return phi;
}
static double basisFunction21(double xi, double eta) {
  double phi = 0.924e3 * pow(xi, 0.6e1) - 0.2772e4 * pow(xi, 0.5e1) + 0.3150e4 * pow(xi, 0.4e1) - 0.1680e4 * pow(xi, 0.3e1) + 0.420e3 * xi * xi - 0.42e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction22(double xi, double eta) {
  double phi = 0.504e3 * eta * pow(xi, 0.5e1) - 0.1260e4 * eta * pow(xi, 0.4e1) - 0.252e3 * pow(xi, 0.5e1) + 0.1120e4 * eta * pow(xi, 0.3e1) + 0.630e3 * pow(xi, 0.4e1) - 0.420e3 * eta * xi * xi - 0.560e3 * pow(xi, 0.3e1) + 0.60e2 * eta * xi + 0.210e3 * xi * xi - 0.2e1 * eta - 0.30e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction23(double xi, double eta) {
  double phi = 0.420e3 * eta * eta * pow(xi, 0.4e1) - 0.840e3 * eta * eta * pow(xi, 0.3e1) - 0.420e3 * eta * pow(xi, 0.4e1) + 0.540e3 * eta * eta * xi * xi + 0.840e3 * eta * pow(xi, 0.3e1) + 0.70e2 * pow(xi, 0.4e1) - 0.120e3 * eta * eta * xi - 0.540e3 * eta * xi * xi - 0.140e3 * pow(xi, 0.3e1) + 0.6e1 * eta * eta + 0.120e3 * eta * xi + 0.90e2 * xi * xi - 0.6e1 * eta - 0.20e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction24(double xi, double eta) {
  double phi = 0.400e3 * pow(eta, 0.3e1) * pow(xi, 0.3e1) - 0.600e3 * pow(eta, 0.3e1) * xi * xi - 0.600e3 * eta * eta * pow(xi, 0.3e1) + 0.240e3 * pow(eta, 0.3e1) * xi + 0.900e3 * eta * eta * xi * xi + 0.240e3 * eta * pow(xi, 0.3e1) - 0.20e2 * pow(eta, 0.3e1) - 0.360e3 * eta * eta * xi - 0.360e3 * eta * xi * xi - 0.20e2 * pow(xi, 0.3e1) + 0.30e2 * eta * eta + 0.144e3 * eta * xi + 0.30e2 * xi * xi - 0.12e2 * eta - 0.12e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction25(double xi, double eta) {
  double phi = 0.420e3 * pow(eta, 0.4e1) * xi * xi - 0.420e3 * pow(eta, 0.4e1) * xi - 0.840e3 * pow(eta, 0.3e1) * xi * xi + 0.70e2 * pow(eta, 0.4e1) + 0.840e3 * pow(eta, 0.3e1) * xi + 0.540e3 * eta * eta * xi * xi - 0.140e3 * pow(eta, 0.3e1) - 0.540e3 * eta * eta * xi - 0.120e3 * eta * xi * xi + 0.90e2 * eta * eta + 0.120e3 * eta * xi + 0.6e1 * xi * xi - 0.20e2 * eta - 0.6e1 * xi + 0.1e1;
  return phi;
}
static double basisFunction26(double xi, double eta) {
  double phi = 0.504e3 * pow(eta, 0.5e1) * xi - 0.252e3 * pow(eta, 0.5e1) - 0.1260e4 * pow(eta, 0.4e1) * xi + 0.630e3 * pow(eta, 0.4e1) + 0.1120e4 * pow(eta, 0.3e1) * xi - 0.560e3 * pow(eta, 0.3e1) - 0.420e3 * eta * eta * xi + 0.210e3 * eta * eta + 0.60e2 * eta * xi - 0.30e2 * eta - 0.2e1 * xi + 0.1e1;
  return phi;
}
static double basisFunction27(double xi, double eta) {
  double phi = 0.924e3 * pow(eta, 0.6e1) - 0.2772e4 * pow(eta, 0.5e1) + 0.3150e4 * pow(eta, 0.4e1) - 0.1680e4 * pow(eta, 0.3e1) + 0.420e3 * eta * eta - 0.42e2 * eta + 0.1e1;
  return phi;
}
static double basisFunction28(double xi, double eta) {
  double phi = 0.3432e4 * pow(xi, 0.7e1) - 0.12012e5 * pow(xi, 0.6e1) + 0.16632e5 * pow(xi, 0.5e1) - 0.11550e5 * pow(xi, 0.4e1) + 0.4200e4 * pow(xi, 0.3e1) - 0.756e3 * xi * xi + 0.56e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction29(double xi, double eta) {
  double phi = 0.1848e4 * eta * pow(xi, 0.6e1) - 0.5544e4 * eta * pow(xi, 0.5e1) - 0.924e3 * pow(xi, 0.6e1) + 0.6300e4 * eta * pow(xi, 0.4e1) + 0.2772e4 * pow(xi, 0.5e1) - 0.3360e4 * eta * pow(xi, 0.3e1) - 0.3150e4 * pow(xi, 0.4e1) + 0.840e3 * eta * xi * xi + 0.1680e4 * pow(xi, 0.3e1) - 0.84e2 * eta * xi - 0.420e3 * xi * xi + 0.2e1 * eta + 0.42e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction30(double xi, double eta) {
  double phi = 0.1512e4 * eta * eta * pow(xi, 0.5e1) - 0.3780e4 * eta * eta * pow(xi, 0.4e1) - 0.1512e4 * eta * pow(xi, 0.5e1) + 0.3360e4 * eta * eta * pow(xi, 0.3e1) + 0.3780e4 * eta * pow(xi, 0.4e1) + 0.252e3 * pow(xi, 0.5e1) - 0.1260e4 * eta * eta * xi * xi - 0.3360e4 * eta * pow(xi, 0.3e1) - 0.630e3 * pow(xi, 0.4e1) + 0.180e3 * eta * eta * xi + 0.1260e4 * eta * xi * xi + 0.560e3 * pow(xi, 0.3e1) - 0.6e1 * eta * eta - 0.180e3 * eta * xi - 0.210e3 * xi * xi + 0.6e1 * eta + 0.30e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction31(double xi, double eta) {
  double phi = 0.1400e4 * pow(eta, 0.3e1) * pow(xi, 0.4e1) - 0.2800e4 * pow(eta, 0.3e1) * pow(xi, 0.3e1) - 0.2100e4 * eta * eta * pow(xi, 0.4e1) + 0.1800e4 * pow(eta, 0.3e1) * xi * xi + 0.4200e4 * eta * eta * pow(xi, 0.3e1) + 0.840e3 * eta * pow(xi, 0.4e1) - 0.400e3 * pow(eta, 0.3e1) * xi - 0.2700e4 * eta * eta * xi * xi - 0.1680e4 * eta * pow(xi, 0.3e1) - 0.70e2 * pow(xi, 0.4e1) + 0.20e2 * pow(eta, 0.3e1) + 0.600e3 * eta * eta * xi + 0.1080e4 * eta * xi * xi + 0.140e3 * pow(xi, 0.3e1) - 0.30e2 * eta * eta - 0.240e3 * eta * xi - 0.90e2 * xi * xi + 0.12e2 * eta + 0.20e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction32(double xi, double eta) {
  double phi = 0.1400e4 * pow(eta, 0.4e1) * pow(xi, 0.3e1) - 0.2100e4 * pow(eta, 0.4e1) * xi * xi - 0.2800e4 * pow(eta, 0.3e1) * pow(xi, 0.3e1) + 0.840e3 * pow(eta, 0.4e1) * xi + 0.4200e4 * pow(eta, 0.3e1) * xi * xi + 0.1800e4 * eta * eta * pow(xi, 0.3e1) - 0.70e2 * pow(eta, 0.4e1) - 0.1680e4 * pow(eta, 0.3e1) * xi - 0.2700e4 * eta * eta * xi * xi - 0.400e3 * eta * pow(xi, 0.3e1) + 0.140e3 * pow(eta, 0.3e1) + 0.1080e4 * eta * eta * xi + 0.600e3 * eta * xi * xi + 0.20e2 * pow(xi, 0.3e1) - 0.90e2 * eta * eta - 0.240e3 * eta * xi - 0.30e2 * xi * xi + 0.20e2 * eta + 0.12e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction33(double xi, double eta) {
  double phi = 0.1512e4 * pow(eta, 0.5e1) * xi * xi - 0.1512e4 * pow(eta, 0.5e1) * xi - 0.3780e4 * pow(eta, 0.4e1) * xi * xi + 0.252e3 * pow(eta, 0.5e1) + 0.3780e4 * pow(eta, 0.4e1) * xi + 0.3360e4 * pow(eta, 0.3e1) * xi * xi - 0.630e3 * pow(eta, 0.4e1) - 0.3360e4 * pow(eta, 0.3e1) * xi - 0.1260e4 * eta * eta * xi * xi + 0.560e3 * pow(eta, 0.3e1) + 0.1260e4 * eta * eta * xi + 0.180e3 * eta * xi * xi - 0.210e3 * eta * eta - 0.180e3 * eta * xi - 0.6e1 * xi * xi + 0.30e2 * eta + 0.6e1 * xi - 0.1e1;
  return phi;
}
static double basisFunction34(double xi, double eta) {
  double phi = 0.1848e4 * pow(eta, 0.6e1) * xi - 0.924e3 * pow(eta, 0.6e1) - 0.5544e4 * pow(eta, 0.5e1) * xi + 0.2772e4 * pow(eta, 0.5e1) + 0.6300e4 * pow(eta, 0.4e1) * xi - 0.3150e4 * pow(eta, 0.4e1) - 0.3360e4 * pow(eta, 0.3e1) * xi + 0.1680e4 * pow(eta, 0.3e1) + 0.840e3 * eta * eta * xi - 0.420e3 * eta * eta - 0.84e2 * eta * xi + 0.42e2 * eta + 0.2e1 * xi - 0.1e1;
  return phi;
}
static double basisFunction35(double xi, double eta) {
  double phi = 0.3432e4 * pow(eta, 0.7e1) - 0.12012e5 * pow(eta, 0.6e1) + 0.16632e5 * pow(eta, 0.5e1) - 0.11550e5 * pow(eta, 0.4e1) + 0.4200e4 * pow(eta, 0.3e1) - 0.756e3 * eta * eta + 0.56e2 * eta - 0.1e1;
  return phi;
}
static double basisFunction36(double xi, double eta) {
  double phi = 0.12870e5 * pow(xi, 0.8e1) - 0.51480e5 * pow(xi, 0.7e1) + 0.84084e5 * pow(xi, 0.6e1) - 0.72072e5 * pow(xi, 0.5e1) + 0.34650e5 * pow(xi, 0.4e1) - 0.9240e4 * pow(xi, 0.3e1) + 0.1260e4 * xi * xi - 0.72e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction37(double xi, double eta) {
  double phi = 0.6864e4 * eta * pow(xi, 0.7e1) - 0.24024e5 * eta * pow(xi, 0.6e1) - 0.3432e4 * pow(xi, 0.7e1) + 0.33264e5 * eta * pow(xi, 0.5e1) + 0.12012e5 * pow(xi, 0.6e1) - 0.23100e5 * eta * pow(xi, 0.4e1) - 0.16632e5 * pow(xi, 0.5e1) + 0.8400e4 * eta * pow(xi, 0.3e1) + 0.11550e5 * pow(xi, 0.4e1) - 0.1512e4 * eta * xi * xi - 0.4200e4 * pow(xi, 0.3e1) + 0.112e3 * eta * xi + 0.756e3 * xi * xi - 0.2e1 * eta - 0.56e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction38(double xi, double eta) {
  double phi = 0.5544e4 * eta * eta * pow(xi, 0.6e1) - 0.16632e5 * eta * eta * pow(xi, 0.5e1) - 0.5544e4 * eta * pow(xi, 0.6e1) + 0.18900e5 * eta * eta * pow(xi, 0.4e1) + 0.16632e5 * eta * pow(xi, 0.5e1) + 0.924e3 * pow(xi, 0.6e1) - 0.10080e5 * eta * eta * pow(xi, 0.3e1) - 0.18900e5 * eta * pow(xi, 0.4e1) - 0.2772e4 * pow(xi, 0.5e1) + 0.2520e4 * eta * eta * xi * xi + 0.10080e5 * eta * pow(xi, 0.3e1) + 0.3150e4 * pow(xi, 0.4e1) - 0.252e3 * eta * eta * xi - 0.2520e4 * eta * xi * xi - 0.1680e4 * pow(xi, 0.3e1) + 0.6e1 * eta * eta + 0.252e3 * eta * xi + 0.420e3 * xi * xi - 0.6e1 * eta - 0.42e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction39(double xi, double eta) {
  double phi = 0.5040e4 * pow(eta, 0.3e1) * pow(xi, 0.5e1) - 0.12600e5 * pow(eta, 0.3e1) * pow(xi, 0.4e1) - 0.7560e4 * eta * eta * pow(xi, 0.5e1) + 0.11200e5 * pow(eta, 0.3e1) * pow(xi, 0.3e1) + 0.18900e5 * eta * eta * pow(xi, 0.4e1) + 0.3024e4 * eta * pow(xi, 0.5e1) - 0.4200e4 * pow(eta, 0.3e1) * xi * xi - 0.16800e5 * eta * eta * pow(xi, 0.3e1) - 0.7560e4 * eta * pow(xi, 0.4e1) - 0.252e3 * pow(xi, 0.5e1) + 0.600e3 * pow(eta, 0.3e1) * xi + 0.6300e4 * eta * eta * xi * xi + 0.6720e4 * eta * pow(xi, 0.3e1) + 0.630e3 * pow(xi, 0.4e1) - 0.20e2 * pow(eta, 0.3e1) - 0.900e3 * eta * eta * xi - 0.2520e4 * eta * xi * xi - 0.560e3 * pow(xi, 0.3e1) + 0.30e2 * eta * eta + 0.360e3 * eta * xi + 0.210e3 * xi * xi - 0.12e2 * eta - 0.30e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction40(double xi, double eta) {
  double phi = 0.4900e4 * pow(eta, 0.4e1) * pow(xi, 0.4e1) - 0.9800e4 * pow(eta, 0.4e1) * pow(xi, 0.3e1) - 0.9800e4 * pow(eta, 0.3e1) * pow(xi, 0.4e1) + 0.6300e4 * pow(eta, 0.4e1) * xi * xi + 0.19600e5 * pow(eta, 0.3e1) * pow(xi, 0.3e1) + 0.6300e4 * eta * eta * pow(xi, 0.4e1) - 0.1400e4 * pow(eta, 0.4e1) * xi - 0.12600e5 * pow(eta, 0.3e1) * xi * xi - 0.12600e5 * eta * eta * pow(xi, 0.3e1) - 0.1400e4 * eta * pow(xi, 0.4e1) + 0.70e2 * pow(eta, 0.4e1) + 0.2800e4 * pow(eta, 0.3e1) * xi + 0.8100e4 * eta * eta * xi * xi + 0.2800e4 * eta * pow(xi, 0.3e1) + 0.70e2 * pow(xi, 0.4e1) - 0.140e3 * pow(eta, 0.3e1) - 0.1800e4 * eta * eta * xi - 0.1800e4 * eta * xi * xi - 0.140e3 * pow(xi, 0.3e1) + 0.90e2 * eta * eta + 0.400e3 * eta * xi + 0.90e2 * xi * xi - 0.20e2 * eta - 0.20e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction41(double xi, double eta) {
  double phi = 0.5040e4 * pow(eta, 0.5e1) * pow(xi, 0.3e1) - 0.7560e4 * pow(eta, 0.5e1) * xi * xi - 0.12600e5 * pow(eta, 0.4e1) * pow(xi, 0.3e1) + 0.3024e4 * pow(eta, 0.5e1) * xi + 0.18900e5 * pow(eta, 0.4e1) * xi * xi + 0.11200e5 * pow(eta, 0.3e1) * pow(xi, 0.3e1) - 0.252e3 * pow(eta, 0.5e1) - 0.7560e4 * pow(eta, 0.4e1) * xi - 0.16800e5 * pow(eta, 0.3e1) * xi * xi - 0.4200e4 * eta * eta * pow(xi, 0.3e1) + 0.630e3 * pow(eta, 0.4e1) + 0.6720e4 * pow(eta, 0.3e1) * xi + 0.6300e4 * eta * eta * xi * xi + 0.600e3 * eta * pow(xi, 0.3e1) - 0.560e3 * pow(eta, 0.3e1) - 0.2520e4 * eta * eta * xi - 0.900e3 * eta * xi * xi - 0.20e2 * pow(xi, 0.3e1) + 0.210e3 * eta * eta + 0.360e3 * eta * xi + 0.30e2 * xi * xi - 0.30e2 * eta - 0.12e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction42(double xi, double eta) {
  double phi = 0.5544e4 * pow(eta, 0.6e1) * xi * xi - 0.5544e4 * pow(eta, 0.6e1) * xi - 0.16632e5 * pow(eta, 0.5e1) * xi * xi + 0.924e3 * pow(eta, 0.6e1) + 0.16632e5 * pow(eta, 0.5e1) * xi + 0.18900e5 * pow(eta, 0.4e1) * xi * xi - 0.2772e4 * pow(eta, 0.5e1) - 0.18900e5 * pow(eta, 0.4e1) * xi - 0.10080e5 * pow(eta, 0.3e1) * xi * xi + 0.3150e4 * pow(eta, 0.4e1) + 0.10080e5 * pow(eta, 0.3e1) * xi + 0.2520e4 * eta * eta * xi * xi - 0.1680e4 * pow(eta, 0.3e1) - 0.2520e4 * eta * eta * xi - 0.252e3 * eta * xi * xi + 0.420e3 * eta * eta + 0.252e3 * eta * xi + 0.6e1 * xi * xi - 0.42e2 * eta - 0.6e1 * xi + 0.1e1;
  return phi;
}
static double basisFunction43(double xi, double eta) {
  double phi = 0.6864e4 * pow(eta, 0.7e1) * xi - 0.3432e4 * pow(eta, 0.7e1) - 0.24024e5 * pow(eta, 0.6e1) * xi + 0.12012e5 * pow(eta, 0.6e1) + 0.33264e5 * pow(eta, 0.5e1) * xi - 0.16632e5 * pow(eta, 0.5e1) - 0.23100e5 * pow(eta, 0.4e1) * xi + 0.11550e5 * pow(eta, 0.4e1) + 0.8400e4 * pow(eta, 0.3e1) * xi - 0.4200e4 * pow(eta, 0.3e1) - 0.1512e4 * eta * eta * xi + 0.756e3 * eta * eta + 0.112e3 * eta * xi - 0.56e2 * eta - 0.2e1 * xi + 0.1e1;
  return phi;
}
static double basisFunction44(double xi, double eta) {
  double phi = 0.12870e5 * pow(eta, 0.8e1) - 0.51480e5 * pow(eta, 0.7e1) + 0.84084e5 * pow(eta, 0.6e1) - 0.72072e5 * pow(eta, 0.5e1) + 0.34650e5 * pow(eta, 0.4e1) - 0.9240e4 * pow(eta, 0.3e1) + 0.1260e4 * eta * eta - 0.72e2 * eta + 0.1e1;
  return phi;
}
static double basisFunction45(double xi, double eta) {
  double phi = 0.48620e5 * pow(xi, 0.9e1) - 0.218790e6 * pow(xi, 0.8e1) + 0.411840e6 * pow(xi, 0.7e1) - 0.420420e6 * pow(xi, 0.6e1) + 0.252252e6 * pow(xi, 0.5e1) - 0.90090e5 * pow(xi, 0.4e1) + 0.18480e5 * pow(xi, 0.3e1) - 0.1980e4 * xi * xi + 0.90e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction46(double xi, double eta) {
  double phi = 0.25740e5 * eta * pow(xi, 0.8e1) - 0.102960e6 * eta * pow(xi, 0.7e1) - 0.12870e5 * pow(xi, 0.8e1) + 0.168168e6 * eta * pow(xi, 0.6e1) + 0.51480e5 * pow(xi, 0.7e1) - 0.144144e6 * eta * pow(xi, 0.5e1) - 0.84084e5 * pow(xi, 0.6e1) + 0.69300e5 * eta * pow(xi, 0.4e1) + 0.72072e5 * pow(xi, 0.5e1) - 0.18480e5 * eta * pow(xi, 0.3e1) - 0.34650e5 * pow(xi, 0.4e1) + 0.2520e4 * eta * xi * xi + 0.9240e4 * pow(xi, 0.3e1) - 0.144e3 * eta * xi - 0.1260e4 * xi * xi + 0.2e1 * eta + 0.72e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction47(double xi, double eta) {
  double phi = 0.20592e5 * eta * eta * pow(xi, 0.7e1) - 0.72072e5 * eta * eta * pow(xi, 0.6e1) - 0.20592e5 * eta * pow(xi, 0.7e1) + 0.99792e5 * eta * eta * pow(xi, 0.5e1) + 0.72072e5 * eta * pow(xi, 0.6e1) + 0.3432e4 * pow(xi, 0.7e1) - 0.69300e5 * eta * eta * pow(xi, 0.4e1) - 0.99792e5 * eta * pow(xi, 0.5e1) - 0.12012e5 * pow(xi, 0.6e1) + 0.25200e5 * eta * eta * pow(xi, 0.3e1) + 0.69300e5 * eta * pow(xi, 0.4e1) + 0.16632e5 * pow(xi, 0.5e1) - 0.4536e4 * eta * eta * xi * xi - 0.25200e5 * eta * pow(xi, 0.3e1) - 0.11550e5 * pow(xi, 0.4e1) + 0.336e3 * eta * eta * xi + 0.4536e4 * eta * xi * xi + 0.4200e4 * pow(xi, 0.3e1) - 0.6e1 * eta * eta - 0.336e3 * eta * xi - 0.756e3 * xi * xi + 0.6e1 * eta + 0.56e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction48(double xi, double eta) {
  double phi = 0.18480e5 * pow(eta, 0.3e1) * pow(xi, 0.6e1) - 0.55440e5 * pow(eta, 0.3e1) * pow(xi, 0.5e1) - 0.27720e5 * eta * eta * pow(xi, 0.6e1) + 0.63000e5 * pow(eta, 0.3e1) * pow(xi, 0.4e1) + 0.83160e5 * eta * eta * pow(xi, 0.5e1) + 0.11088e5 * eta * pow(xi, 0.6e1) - 0.33600e5 * pow(eta, 0.3e1) * pow(xi, 0.3e1) - 0.94500e5 * eta * eta * pow(xi, 0.4e1) - 0.33264e5 * eta * pow(xi, 0.5e1) - 0.924e3 * pow(xi, 0.6e1) + 0.8400e4 * pow(eta, 0.3e1) * xi * xi + 0.50400e5 * eta * eta * pow(xi, 0.3e1) + 0.37800e5 * eta * pow(xi, 0.4e1) + 0.2772e4 * pow(xi, 0.5e1) - 0.840e3 * pow(eta, 0.3e1) * xi - 0.12600e5 * eta * eta * xi * xi - 0.20160e5 * eta * pow(xi, 0.3e1) - 0.3150e4 * pow(xi, 0.4e1) + 0.20e2 * pow(eta, 0.3e1) + 0.1260e4 * eta * eta * xi + 0.5040e4 * eta * xi * xi + 0.1680e4 * pow(xi, 0.3e1) - 0.30e2 * eta * eta - 0.504e3 * eta * xi - 0.420e3 * xi * xi + 0.12e2 * eta + 0.42e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction49(double xi, double eta) {
  double phi = 0.17640e5 * pow(eta, 0.4e1) * pow(xi, 0.5e1) - 0.44100e5 * pow(eta, 0.4e1) * pow(xi, 0.4e1) - 0.35280e5 * pow(eta, 0.3e1) * pow(xi, 0.5e1) + 0.39200e5 * pow(eta, 0.4e1) * pow(xi, 0.3e1) + 0.88200e5 * pow(eta, 0.3e1) * pow(xi, 0.4e1) + 0.22680e5 * eta * eta * pow(xi, 0.5e1) - 0.14700e5 * pow(eta, 0.4e1) * xi * xi - 0.78400e5 * pow(eta, 0.3e1) * pow(xi, 0.3e1) - 0.56700e5 * eta * eta * pow(xi, 0.4e1) - 0.5040e4 * eta * pow(xi, 0.5e1) + 0.2100e4 * pow(eta, 0.4e1) * xi + 0.29400e5 * pow(eta, 0.3e1) * xi * xi + 0.50400e5 * eta * eta * pow(xi, 0.3e1) + 0.12600e5 * eta * pow(xi, 0.4e1) + 0.252e3 * pow(xi, 0.5e1) - 0.70e2 * pow(eta, 0.4e1) - 0.4200e4 * pow(eta, 0.3e1) * xi - 0.18900e5 * eta * eta * xi * xi - 0.11200e5 * eta * pow(xi, 0.3e1) - 0.630e3 * pow(xi, 0.4e1) + 0.140e3 * pow(eta, 0.3e1) + 0.2700e4 * eta * eta * xi + 0.4200e4 * eta * xi * xi + 0.560e3 * pow(xi, 0.3e1) - 0.90e2 * eta * eta - 0.600e3 * eta * xi - 0.210e3 * xi * xi + 0.20e2 * eta + 0.30e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction50(double xi, double eta) {
  double phi = 0.17640e5 * pow(eta, 0.5e1) * pow(xi, 0.4e1) - 0.35280e5 * pow(eta, 0.5e1) * pow(xi, 0.3e1) - 0.44100e5 * pow(eta, 0.4e1) * pow(xi, 0.4e1) + 0.22680e5 * pow(eta, 0.5e1) * xi * xi + 0.88200e5 * pow(eta, 0.4e1) * pow(xi, 0.3e1) + 0.39200e5 * pow(eta, 0.3e1) * pow(xi, 0.4e1) - 0.5040e4 * pow(eta, 0.5e1) * xi - 0.56700e5 * pow(eta, 0.4e1) * xi * xi - 0.78400e5 * pow(eta, 0.3e1) * pow(xi, 0.3e1) - 0.14700e5 * eta * eta * pow(xi, 0.4e1) + 0.252e3 * pow(eta, 0.5e1) + 0.12600e5 * pow(eta, 0.4e1) * xi + 0.50400e5 * pow(eta, 0.3e1) * xi * xi + 0.29400e5 * eta * eta * pow(xi, 0.3e1) + 0.2100e4 * eta * pow(xi, 0.4e1) - 0.630e3 * pow(eta, 0.4e1) - 0.11200e5 * pow(eta, 0.3e1) * xi - 0.18900e5 * eta * eta * xi * xi - 0.4200e4 * eta * pow(xi, 0.3e1) - 0.70e2 * pow(xi, 0.4e1) + 0.560e3 * pow(eta, 0.3e1) + 0.4200e4 * eta * eta * xi + 0.2700e4 * eta * xi * xi + 0.140e3 * pow(xi, 0.3e1) - 0.210e3 * eta * eta - 0.600e3 * eta * xi - 0.90e2 * xi * xi + 0.30e2 * eta + 0.20e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction51(double xi, double eta) {
  double phi = 0.18480e5 * pow(eta, 0.6e1) * pow(xi, 0.3e1) - 0.27720e5 * pow(eta, 0.6e1) * xi * xi - 0.55440e5 * pow(eta, 0.5e1) * pow(xi, 0.3e1) + 0.11088e5 * pow(eta, 0.6e1) * xi + 0.83160e5 * pow(eta, 0.5e1) * xi * xi + 0.63000e5 * pow(eta, 0.4e1) * pow(xi, 0.3e1) - 0.924e3 * pow(eta, 0.6e1) - 0.33264e5 * pow(eta, 0.5e1) * xi - 0.94500e5 * pow(eta, 0.4e1) * xi * xi - 0.33600e5 * pow(eta, 0.3e1) * pow(xi, 0.3e1) + 0.2772e4 * pow(eta, 0.5e1) + 0.37800e5 * pow(eta, 0.4e1) * xi + 0.50400e5 * pow(eta, 0.3e1) * xi * xi + 0.8400e4 * eta * eta * pow(xi, 0.3e1) - 0.3150e4 * pow(eta, 0.4e1) - 0.20160e5 * pow(eta, 0.3e1) * xi - 0.12600e5 * eta * eta * xi * xi - 0.840e3 * eta * pow(xi, 0.3e1) + 0.1680e4 * pow(eta, 0.3e1) + 0.5040e4 * eta * eta * xi + 0.1260e4 * eta * xi * xi + 0.20e2 * pow(xi, 0.3e1) - 0.420e3 * eta * eta - 0.504e3 * eta * xi - 0.30e2 * xi * xi + 0.42e2 * eta + 0.12e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction52(double xi, double eta) {
  double phi = 0.20592e5 * pow(eta, 0.7e1) * xi * xi - 0.20592e5 * pow(eta, 0.7e1) * xi - 0.72072e5 * pow(eta, 0.6e1) * xi * xi + 0.3432e4 * pow(eta, 0.7e1) + 0.72072e5 * pow(eta, 0.6e1) * xi + 0.99792e5 * pow(eta, 0.5e1) * xi * xi - 0.12012e5 * pow(eta, 0.6e1) - 0.99792e5 * pow(eta, 0.5e1) * xi - 0.69300e5 * pow(eta, 0.4e1) * xi * xi + 0.16632e5 * pow(eta, 0.5e1) + 0.69300e5 * pow(eta, 0.4e1) * xi + 0.25200e5 * pow(eta, 0.3e1) * xi * xi - 0.11550e5 * pow(eta, 0.4e1) - 0.25200e5 * pow(eta, 0.3e1) * xi - 0.4536e4 * eta * eta * xi * xi + 0.4200e4 * pow(eta, 0.3e1) + 0.4536e4 * eta * eta * xi + 0.336e3 * eta * xi * xi - 0.756e3 * eta * eta - 0.336e3 * eta * xi - 0.6e1 * xi * xi + 0.56e2 * eta + 0.6e1 * xi - 0.1e1;
  return phi;
}
static double basisFunction53(double xi, double eta) {
  double phi = 0.25740e5 * pow(eta, 0.8e1) * xi - 0.12870e5 * pow(eta, 0.8e1) - 0.102960e6 * pow(eta, 0.7e1) * xi + 0.51480e5 * pow(eta, 0.7e1) + 0.168168e6 * pow(eta, 0.6e1) * xi - 0.84084e5 * pow(eta, 0.6e1) - 0.144144e6 * pow(eta, 0.5e1) * xi + 0.72072e5 * pow(eta, 0.5e1) + 0.69300e5 * pow(eta, 0.4e1) * xi - 0.34650e5 * pow(eta, 0.4e1) - 0.18480e5 * pow(eta, 0.3e1) * xi + 0.9240e4 * pow(eta, 0.3e1) + 0.2520e4 * eta * eta * xi - 0.1260e4 * eta * eta - 0.144e3 * eta * xi + 0.72e2 * eta + 0.2e1 * xi - 0.1e1;
  return phi;
}
static double basisFunction54(double xi, double eta) {
  double phi = 0.48620e5 * pow(eta, 0.9e1) - 0.218790e6 * pow(eta, 0.8e1) + 0.411840e6 * pow(eta, 0.7e1) - 0.420420e6 * pow(eta, 0.6e1) + 0.252252e6 * pow(eta, 0.5e1) - 0.90090e5 * pow(eta, 0.4e1) + 0.18480e5 * pow(eta, 0.3e1) - 0.1980e4 * eta * eta + 0.90e2 * eta - 0.1e1;
  return phi;
}
static double basisFunction55(double xi, double eta) {
  double phi = 0.184756e6 * pow(xi, 0.10e2) - 0.923780e6 * pow(xi, 0.9e1) + 0.1969110e7 * pow(xi, 0.8e1) - 0.2333760e7 * pow(xi, 0.7e1) + 0.1681680e7 * pow(xi, 0.6e1) - 0.756756e6 * pow(xi, 0.5e1) + 0.210210e6 * pow(xi, 0.4e1) - 0.34320e5 * pow(xi, 0.3e1) + 0.2970e4 * xi * xi - 0.110e3 * xi + 0.1e1;
  return phi;
}
static double basisFunction56(double xi, double eta) {
  double phi = 0.97240e5 * eta * pow(xi, 0.9e1) - 0.437580e6 * eta * pow(xi, 0.8e1) - 0.48620e5 * pow(xi, 0.9e1) + 0.823680e6 * eta * pow(xi, 0.7e1) + 0.218790e6 * pow(xi, 0.8e1) - 0.840840e6 * eta * pow(xi, 0.6e1) - 0.411840e6 * pow(xi, 0.7e1) + 0.504504e6 * eta * pow(xi, 0.5e1) + 0.420420e6 * pow(xi, 0.6e1) - 0.180180e6 * eta * pow(xi, 0.4e1) - 0.252252e6 * pow(xi, 0.5e1) + 0.36960e5 * eta * pow(xi, 0.3e1) + 0.90090e5 * pow(xi, 0.4e1) - 0.3960e4 * eta * xi * xi - 0.18480e5 * pow(xi, 0.3e1) + 0.180e3 * eta * xi + 0.1980e4 * xi * xi - 0.2e1 * eta - 0.90e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction57(double xi, double eta) {
  double phi = 0.77220e5 * eta * eta * pow(xi, 0.8e1) - 0.308880e6 * eta * eta * pow(xi, 0.7e1) - 0.77220e5 * eta * pow(xi, 0.8e1) + 0.504504e6 * eta * eta * pow(xi, 0.6e1) + 0.308880e6 * eta * pow(xi, 0.7e1) + 0.12870e5 * pow(xi, 0.8e1) - 0.432432e6 * eta * eta * pow(xi, 0.5e1) - 0.504504e6 * eta * pow(xi, 0.6e1) - 0.51480e5 * pow(xi, 0.7e1) + 0.207900e6 * eta * eta * pow(xi, 0.4e1) + 0.432432e6 * eta * pow(xi, 0.5e1) + 0.84084e5 * pow(xi, 0.6e1) - 0.55440e5 * eta * eta * pow(xi, 0.3e1) - 0.207900e6 * eta * pow(xi, 0.4e1) - 0.72072e5 * pow(xi, 0.5e1) + 0.7560e4 * eta * eta * xi * xi + 0.55440e5 * eta * pow(xi, 0.3e1) + 0.34650e5 * pow(xi, 0.4e1) - 0.432e3 * eta * eta * xi - 0.7560e4 * eta * xi * xi - 0.9240e4 * pow(xi, 0.3e1) + 0.6e1 * eta * eta + 0.432e3 * eta * xi + 0.1260e4 * xi * xi - 0.6e1 * eta - 0.72e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction58(double xi, double eta) {
  double phi = 0.68640e5 * pow(eta, 0.3e1) * pow(xi, 0.7e1) - 0.240240e6 * pow(eta, 0.3e1) * pow(xi, 0.6e1) - 0.102960e6 * eta * eta * pow(xi, 0.7e1) + 0.332640e6 * pow(eta, 0.3e1) * pow(xi, 0.5e1) + 0.360360e6 * eta * eta * pow(xi, 0.6e1) + 0.41184e5 * eta * pow(xi, 0.7e1) - 0.231000e6 * pow(eta, 0.3e1) * pow(xi, 0.4e1) - 0.498960e6 * eta * eta * pow(xi, 0.5e1) - 0.144144e6 * eta * pow(xi, 0.6e1) - 0.3432e4 * pow(xi, 0.7e1) + 0.84000e5 * pow(eta, 0.3e1) * pow(xi, 0.3e1) + 0.346500e6 * eta * eta * pow(xi, 0.4e1) + 0.199584e6 * eta * pow(xi, 0.5e1) + 0.12012e5 * pow(xi, 0.6e1) - 0.15120e5 * pow(eta, 0.3e1) * xi * xi - 0.126000e6 * eta * eta * pow(xi, 0.3e1) - 0.138600e6 * eta * pow(xi, 0.4e1) - 0.16632e5 * pow(xi, 0.5e1) + 0.1120e4 * pow(eta, 0.3e1) * xi + 0.22680e5 * eta * eta * xi * xi + 0.50400e5 * eta * pow(xi, 0.3e1) + 0.11550e5 * pow(xi, 0.4e1) - 0.20e2 * pow(eta, 0.3e1) - 0.1680e4 * eta * eta * xi - 0.9072e4 * eta * xi * xi - 0.4200e4 * pow(xi, 0.3e1) + 0.30e2 * eta * eta + 0.672e3 * eta * xi + 0.756e3 * xi * xi - 0.12e2 * eta - 0.56e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction59(double xi, double eta) {
  double phi = 0.64680e5 * pow(eta, 0.4e1) * pow(xi, 0.6e1) - 0.194040e6 * pow(eta, 0.4e1) * pow(xi, 0.5e1) - 0.129360e6 * pow(eta, 0.3e1) * pow(xi, 0.6e1) + 0.220500e6 * pow(eta, 0.4e1) * pow(xi, 0.4e1) + 0.388080e6 * pow(eta, 0.3e1) * pow(xi, 0.5e1) + 0.83160e5 * eta * eta * pow(xi, 0.6e1) - 0.117600e6 * pow(eta, 0.4e1) * pow(xi, 0.3e1) - 0.441000e6 * pow(eta, 0.3e1) * pow(xi, 0.4e1) - 0.249480e6 * eta * eta * pow(xi, 0.5e1) - 0.18480e5 * eta * pow(xi, 0.6e1) + 0.29400e5 * pow(eta, 0.4e1) * xi * xi + 0.235200e6 * pow(eta, 0.3e1) * pow(xi, 0.3e1) + 0.283500e6 * eta * eta * pow(xi, 0.4e1) + 0.55440e5 * eta * pow(xi, 0.5e1) + 0.924e3 * pow(xi, 0.6e1) - 0.2940e4 * pow(eta, 0.4e1) * xi - 0.58800e5 * pow(eta, 0.3e1) * xi * xi - 0.151200e6 * eta * eta * pow(xi, 0.3e1) - 0.63000e5 * eta * pow(xi, 0.4e1) - 0.2772e4 * pow(xi, 0.5e1) + 0.70e2 * pow(eta, 0.4e1) + 0.5880e4 * pow(eta, 0.3e1) * xi + 0.37800e5 * eta * eta * xi * xi + 0.33600e5 * eta * pow(xi, 0.3e1) + 0.3150e4 * pow(xi, 0.4e1) - 0.140e3 * pow(eta, 0.3e1) - 0.3780e4 * eta * eta * xi - 0.8400e4 * eta * xi * xi - 0.1680e4 * pow(xi, 0.3e1) + 0.90e2 * eta * eta + 0.840e3 * eta * xi + 0.420e3 * xi * xi - 0.20e2 * eta - 0.42e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction60(double xi, double eta) {
  double phi = 0.63504e5 * pow(eta, 0.5e1) * pow(xi, 0.5e1) - 0.158760e6 * pow(eta, 0.5e1) * pow(xi, 0.4e1) - 0.158760e6 * pow(eta, 0.4e1) * pow(xi, 0.5e1) + 0.141120e6 * pow(eta, 0.5e1) * pow(xi, 0.3e1) + 0.396900e6 * pow(eta, 0.4e1) * pow(xi, 0.4e1) + 0.141120e6 * pow(eta, 0.3e1) * pow(xi, 0.5e1) - 0.52920e5 * pow(eta, 0.5e1) * xi * xi - 0.352800e6 * pow(eta, 0.4e1) * pow(xi, 0.3e1) - 0.352800e6 * pow(eta, 0.3e1) * pow(xi, 0.4e1) - 0.52920e5 * eta * eta * pow(xi, 0.5e1) + 0.7560e4 * pow(eta, 0.5e1) * xi + 0.132300e6 * pow(eta, 0.4e1) * xi * xi + 0.313600e6 * pow(eta, 0.3e1) * pow(xi, 0.3e1) + 0.132300e6 * eta * eta * pow(xi, 0.4e1) + 0.7560e4 * eta * pow(xi, 0.5e1) - 0.252e3 * pow(eta, 0.5e1) - 0.18900e5 * pow(eta, 0.4e1) * xi - 0.117600e6 * pow(eta, 0.3e1) * xi * xi - 0.117600e6 * eta * eta * pow(xi, 0.3e1) - 0.18900e5 * eta * pow(xi, 0.4e1) - 0.252e3 * pow(xi, 0.5e1) + 0.630e3 * pow(eta, 0.4e1) + 0.16800e5 * pow(eta, 0.3e1) * xi + 0.44100e5 * eta * eta * xi * xi + 0.16800e5 * eta * pow(xi, 0.3e1) + 0.630e3 * pow(xi, 0.4e1) - 0.560e3 * pow(eta, 0.3e1) - 0.6300e4 * eta * eta * xi - 0.6300e4 * eta * xi * xi - 0.560e3 * pow(xi, 0.3e1) + 0.210e3 * eta * eta + 0.900e3 * eta * xi + 0.210e3 * xi * xi - 0.30e2 * eta - 0.30e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction61(double xi, double eta) {
  double phi = 0.64680e5 * pow(eta, 0.6e1) * pow(xi, 0.4e1) - 0.129360e6 * pow(eta, 0.6e1) * pow(xi, 0.3e1) - 0.194040e6 * pow(eta, 0.5e1) * pow(xi, 0.4e1) + 0.83160e5 * pow(eta, 0.6e1) * xi * xi + 0.388080e6 * pow(eta, 0.5e1) * pow(xi, 0.3e1) + 0.220500e6 * pow(eta, 0.4e1) * pow(xi, 0.4e1) - 0.18480e5 * pow(eta, 0.6e1) * xi - 0.249480e6 * pow(eta, 0.5e1) * xi * xi - 0.441000e6 * pow(eta, 0.4e1) * pow(xi, 0.3e1) - 0.117600e6 * pow(eta, 0.3e1) * pow(xi, 0.4e1) + 0.924e3 * pow(eta, 0.6e1) + 0.55440e5 * pow(eta, 0.5e1) * xi + 0.283500e6 * pow(eta, 0.4e1) * xi * xi + 0.235200e6 * pow(eta, 0.3e1) * pow(xi, 0.3e1) + 0.29400e5 * eta * eta * pow(xi, 0.4e1) - 0.2772e4 * pow(eta, 0.5e1) - 0.63000e5 * pow(eta, 0.4e1) * xi - 0.151200e6 * pow(eta, 0.3e1) * xi * xi - 0.58800e5 * eta * eta * pow(xi, 0.3e1) - 0.2940e4 * eta * pow(xi, 0.4e1) + 0.3150e4 * pow(eta, 0.4e1) + 0.33600e5 * pow(eta, 0.3e1) * xi + 0.37800e5 * eta * eta * xi * xi + 0.5880e4 * eta * pow(xi, 0.3e1) + 0.70e2 * pow(xi, 0.4e1) - 0.1680e4 * pow(eta, 0.3e1) - 0.8400e4 * eta * eta * xi - 0.3780e4 * eta * xi * xi - 0.140e3 * pow(xi, 0.3e1) + 0.420e3 * eta * eta + 0.840e3 * eta * xi + 0.90e2 * xi * xi - 0.42e2 * eta - 0.20e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction62(double xi, double eta) {
  double phi = 0.68640e5 * pow(eta, 0.7e1) * pow(xi, 0.3e1) - 0.102960e6 * pow(eta, 0.7e1) * xi * xi - 0.240240e6 * pow(eta, 0.6e1) * pow(xi, 0.3e1) + 0.41184e5 * pow(eta, 0.7e1) * xi + 0.360360e6 * pow(eta, 0.6e1) * xi * xi + 0.332640e6 * pow(eta, 0.5e1) * pow(xi, 0.3e1) - 0.3432e4 * pow(eta, 0.7e1) - 0.144144e6 * pow(eta, 0.6e1) * xi - 0.498960e6 * pow(eta, 0.5e1) * xi * xi - 0.231000e6 * pow(eta, 0.4e1) * pow(xi, 0.3e1) + 0.12012e5 * pow(eta, 0.6e1) + 0.199584e6 * pow(eta, 0.5e1) * xi + 0.346500e6 * pow(eta, 0.4e1) * xi * xi + 0.84000e5 * pow(eta, 0.3e1) * pow(xi, 0.3e1) - 0.16632e5 * pow(eta, 0.5e1) - 0.138600e6 * pow(eta, 0.4e1) * xi - 0.126000e6 * pow(eta, 0.3e1) * xi * xi - 0.15120e5 * eta * eta * pow(xi, 0.3e1) + 0.11550e5 * pow(eta, 0.4e1) + 0.50400e5 * pow(eta, 0.3e1) * xi + 0.22680e5 * eta * eta * xi * xi + 0.1120e4 * eta * pow(xi, 0.3e1) - 0.4200e4 * pow(eta, 0.3e1) - 0.9072e4 * eta * eta * xi - 0.1680e4 * eta * xi * xi - 0.20e2 * pow(xi, 0.3e1) + 0.756e3 * eta * eta + 0.672e3 * eta * xi + 0.30e2 * xi * xi - 0.56e2 * eta - 0.12e2 * xi + 0.1e1;
  return phi;
}
static double basisFunction63(double xi, double eta) {
  double phi = 0.77220e5 * pow(eta, 0.8e1) * xi * xi - 0.77220e5 * pow(eta, 0.8e1) * xi - 0.308880e6 * pow(eta, 0.7e1) * xi * xi + 0.12870e5 * pow(eta, 0.8e1) + 0.308880e6 * pow(eta, 0.7e1) * xi + 0.504504e6 * pow(eta, 0.6e1) * xi * xi - 0.51480e5 * pow(eta, 0.7e1) - 0.504504e6 * pow(eta, 0.6e1) * xi - 0.432432e6 * pow(eta, 0.5e1) * xi * xi + 0.84084e5 * pow(eta, 0.6e1) + 0.432432e6 * pow(eta, 0.5e1) * xi + 0.207900e6 * pow(eta, 0.4e1) * xi * xi - 0.72072e5 * pow(eta, 0.5e1) - 0.207900e6 * pow(eta, 0.4e1) * xi - 0.55440e5 * pow(eta, 0.3e1) * xi * xi + 0.34650e5 * pow(eta, 0.4e1) + 0.55440e5 * pow(eta, 0.3e1) * xi + 0.7560e4 * eta * eta * xi * xi - 0.9240e4 * pow(eta, 0.3e1) - 0.7560e4 * eta * eta * xi - 0.432e3 * eta * xi * xi + 0.1260e4 * eta * eta + 0.432e3 * eta * xi + 0.6e1 * xi * xi - 0.72e2 * eta - 0.6e1 * xi + 0.1e1;
  return phi;
}
static double basisFunction64(double xi, double eta) {
  double phi = 0.97240e5 * pow(eta, 0.9e1) * xi - 0.48620e5 * pow(eta, 0.9e1) - 0.437580e6 * pow(eta, 0.8e1) * xi + 0.218790e6 * pow(eta, 0.8e1) + 0.823680e6 * pow(eta, 0.7e1) * xi - 0.411840e6 * pow(eta, 0.7e1) - 0.840840e6 * pow(eta, 0.6e1) * xi + 0.420420e6 * pow(eta, 0.6e1) + 0.504504e6 * pow(eta, 0.5e1) * xi - 0.252252e6 * pow(eta, 0.5e1) - 0.180180e6 * pow(eta, 0.4e1) * xi + 0.90090e5 * pow(eta, 0.4e1) + 0.36960e5 * pow(eta, 0.3e1) * xi - 0.18480e5 * pow(eta, 0.3e1) - 0.3960e4 * eta * eta * xi + 0.1980e4 * eta * eta + 0.180e3 * eta * xi - 0.90e2 * eta - 0.2e1 * xi + 0.1e1;
  return phi;
}
static double basisFunction65(double xi, double eta) {
  double phi = 0.184756e6 * pow(eta, 0.10e2) - 0.923780e6 * pow(eta, 0.9e1) + 0.1969110e7 * pow(eta, 0.8e1) - 0.2333760e7 * pow(eta, 0.7e1) + 0.1681680e7 * pow(eta, 0.6e1) - 0.756756e6 * pow(eta, 0.5e1) + 0.210210e6 * pow(eta, 0.4e1) - 0.34320e5 * pow(eta, 0.3e1) + 0.2970e4 * eta * eta - 0.110e3 * eta + 0.1e1;
  return phi;
}
static double basisFunction66(double xi, double eta) {
  double phi = 0.705432e6 * pow(xi, 0.11e2) - 0.3879876e7 * pow(xi, 0.10e2) + 0.9237800e7 * pow(xi, 0.9e1) - 0.12471030e8 * pow(xi, 0.8e1) + 0.10501920e8 * pow(xi, 0.7e1) - 0.5717712e7 * pow(xi, 0.6e1) + 0.2018016e7 * pow(xi, 0.5e1) - 0.450450e6 * pow(xi, 0.4e1) + 0.60060e5 * pow(xi, 0.3e1) - 0.4290e4 * xi * xi + 0.132e3 * xi - 0.1e1;
  return phi;
}
static double basisFunction67(double xi, double eta) {
  double phi = 0.369512e6 * eta * pow(xi, 0.10e2) - 0.1847560e7 * eta * pow(xi, 0.9e1) - 0.184756e6 * pow(xi, 0.10e2) + 0.3938220e7 * eta * pow(xi, 0.8e1) + 0.923780e6 * pow(xi, 0.9e1) - 0.4667520e7 * eta * pow(xi, 0.7e1) - 0.1969110e7 * pow(xi, 0.8e1) + 0.3363360e7 * eta * pow(xi, 0.6e1) + 0.2333760e7 * pow(xi, 0.7e1) - 0.1513512e7 * eta * pow(xi, 0.5e1) - 0.1681680e7 * pow(xi, 0.6e1) + 0.420420e6 * eta * pow(xi, 0.4e1) + 0.756756e6 * pow(xi, 0.5e1) - 0.68640e5 * eta * pow(xi, 0.3e1) - 0.210210e6 * pow(xi, 0.4e1) + 0.5940e4 * eta * xi * xi + 0.34320e5 * pow(xi, 0.3e1) - 0.220e3 * eta * xi - 0.2970e4 * xi * xi + 0.2e1 * eta + 0.110e3 * xi - 0.1e1;
  return phi;
}
static double basisFunction68(double xi, double eta) {
  double phi = 0.291720e6 * eta * eta * pow(xi, 0.9e1) - 0.1312740e7 * eta * eta * pow(xi, 0.8e1) - 0.291720e6 * eta * pow(xi, 0.9e1) + 0.2471040e7 * eta * eta * pow(xi, 0.7e1) + 0.1312740e7 * eta * pow(xi, 0.8e1) + 0.48620e5 * pow(xi, 0.9e1) - 0.2522520e7 * eta * eta * pow(xi, 0.6e1) - 0.2471040e7 * eta * pow(xi, 0.7e1) - 0.218790e6 * pow(xi, 0.8e1) + 0.1513512e7 * eta * eta * pow(xi, 0.5e1) + 0.2522520e7 * eta * pow(xi, 0.6e1) + 0.411840e6 * pow(xi, 0.7e1) - 0.540540e6 * eta * eta * pow(xi, 0.4e1) - 0.1513512e7 * eta * pow(xi, 0.5e1) - 0.420420e6 * pow(xi, 0.6e1) + 0.110880e6 * eta * eta * pow(xi, 0.3e1) + 0.540540e6 * eta * pow(xi, 0.4e1) + 0.252252e6 * pow(xi, 0.5e1) - 0.11880e5 * eta * eta * xi * xi - 0.110880e6 * eta * pow(xi, 0.3e1) - 0.90090e5 * pow(xi, 0.4e1) + 0.540e3 * eta * eta * xi + 0.11880e5 * eta * xi * xi + 0.18480e5 * pow(xi, 0.3e1) - 0.6e1 * eta * eta - 0.540e3 * eta * xi - 0.1980e4 * xi * xi + 0.6e1 * eta + 0.90e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction69(double xi, double eta) {
  double phi = 0.257400e6 * pow(eta, 0.3e1) * pow(xi, 0.8e1) - 0.1029600e7 * pow(eta, 0.3e1) * pow(xi, 0.7e1) - 0.386100e6 * eta * eta * pow(xi, 0.8e1) + 0.1681680e7 * pow(eta, 0.3e1) * pow(xi, 0.6e1) + 0.1544400e7 * eta * eta * pow(xi, 0.7e1) + 0.154440e6 * eta * pow(xi, 0.8e1) - 0.1441440e7 * pow(eta, 0.3e1) * pow(xi, 0.5e1) - 0.2522520e7 * eta * eta * pow(xi, 0.6e1) - 0.617760e6 * eta * pow(xi, 0.7e1) - 0.12870e5 * pow(xi, 0.8e1) + 0.693000e6 * pow(eta, 0.3e1) * pow(xi, 0.4e1) + 0.2162160e7 * eta * eta * pow(xi, 0.5e1) + 0.1009008e7 * eta * pow(xi, 0.6e1) + 0.51480e5 * pow(xi, 0.7e1) - 0.184800e6 * pow(eta, 0.3e1) * pow(xi, 0.3e1) - 0.1039500e7 * eta * eta * pow(xi, 0.4e1) - 0.864864e6 * eta * pow(xi, 0.5e1) - 0.84084e5 * pow(xi, 0.6e1) + 0.25200e5 * pow(eta, 0.3e1) * xi * xi + 0.277200e6 * eta * eta * pow(xi, 0.3e1) + 0.415800e6 * eta * pow(xi, 0.4e1) + 0.72072e5 * pow(xi, 0.5e1) - 0.1440e4 * pow(eta, 0.3e1) * xi - 0.37800e5 * eta * eta * xi * xi - 0.110880e6 * eta * pow(xi, 0.3e1) - 0.34650e5 * pow(xi, 0.4e1) + 0.20e2 * pow(eta, 0.3e1) + 0.2160e4 * eta * eta * xi + 0.15120e5 * eta * xi * xi + 0.9240e4 * pow(xi, 0.3e1) - 0.30e2 * eta * eta - 0.864e3 * eta * xi - 0.1260e4 * xi * xi + 0.12e2 * eta + 0.72e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction70(double xi, double eta) {
  double phi = 0.240240e6 * pow(eta, 0.4e1) * pow(xi, 0.7e1) - 0.840840e6 * pow(eta, 0.4e1) * pow(xi, 0.6e1) - 0.480480e6 * pow(eta, 0.3e1) * pow(xi, 0.7e1) + 0.1164240e7 * pow(eta, 0.4e1) * pow(xi, 0.5e1) + 0.1681680e7 * pow(eta, 0.3e1) * pow(xi, 0.6e1) + 0.308880e6 * eta * eta * pow(xi, 0.7e1) - 0.808500e6 * pow(eta, 0.4e1) * pow(xi, 0.4e1) - 0.2328480e7 * pow(eta, 0.3e1) * pow(xi, 0.5e1) - 0.1081080e7 * eta * eta * pow(xi, 0.6e1) - 0.68640e5 * eta * pow(xi, 0.7e1) + 0.294000e6 * pow(eta, 0.4e1) * pow(xi, 0.3e1) + 0.1617000e7 * pow(eta, 0.3e1) * pow(xi, 0.4e1) + 0.1496880e7 * eta * eta * pow(xi, 0.5e1) + 0.240240e6 * eta * pow(xi, 0.6e1) + 0.3432e4 * pow(xi, 0.7e1) - 0.52920e5 * pow(eta, 0.4e1) * xi * xi - 0.588000e6 * pow(eta, 0.3e1) * pow(xi, 0.3e1) - 0.1039500e7 * eta * eta * pow(xi, 0.4e1) - 0.332640e6 * eta * pow(xi, 0.5e1) - 0.12012e5 * pow(xi, 0.6e1) + 0.3920e4 * pow(eta, 0.4e1) * xi + 0.105840e6 * pow(eta, 0.3e1) * xi * xi + 0.378000e6 * eta * eta * pow(xi, 0.3e1) + 0.231000e6 * eta * pow(xi, 0.4e1) + 0.16632e5 * pow(xi, 0.5e1) - 0.70e2 * pow(eta, 0.4e1) - 0.7840e4 * pow(eta, 0.3e1) * xi - 0.68040e5 * eta * eta * xi * xi - 0.84000e5 * eta * pow(xi, 0.3e1) - 0.11550e5 * pow(xi, 0.4e1) + 0.140e3 * pow(eta, 0.3e1) + 0.5040e4 * eta * eta * xi + 0.15120e5 * eta * xi * xi + 0.4200e4 * pow(xi, 0.3e1) - 0.90e2 * eta * eta - 0.1120e4 * eta * xi - 0.756e3 * xi * xi + 0.20e2 * eta + 0.56e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction71(double xi, double eta) {
  double phi = 0.232848e6 * pow(eta, 0.5e1) * pow(xi, 0.6e1) - 0.698544e6 * pow(eta, 0.5e1) * pow(xi, 0.5e1) - 0.582120e6 * pow(eta, 0.4e1) * pow(xi, 0.6e1) + 0.793800e6 * pow(eta, 0.5e1) * pow(xi, 0.4e1) + 0.1746360e7 * pow(eta, 0.4e1) * pow(xi, 0.5e1) + 0.517440e6 * pow(eta, 0.3e1) * pow(xi, 0.6e1) - 0.423360e6 * pow(eta, 0.5e1) * pow(xi, 0.3e1) - 0.1984500e7 * pow(eta, 0.4e1) * pow(xi, 0.4e1) - 0.1552320e7 * pow(eta, 0.3e1) * pow(xi, 0.5e1) - 0.194040e6 * eta * eta * pow(xi, 0.6e1) + 0.105840e6 * pow(eta, 0.5e1) * xi * xi + 0.1058400e7 * pow(eta, 0.4e1) * pow(xi, 0.3e1) + 0.1764000e7 * pow(eta, 0.3e1) * pow(xi, 0.4e1) + 0.582120e6 * eta * eta * pow(xi, 0.5e1) + 0.27720e5 * eta * pow(xi, 0.6e1) - 0.10584e5 * pow(eta, 0.5e1) * xi - 0.264600e6 * pow(eta, 0.4e1) * xi * xi - 0.940800e6 * pow(eta, 0.3e1) * pow(xi, 0.3e1) - 0.661500e6 * eta * eta * pow(xi, 0.4e1) - 0.83160e5 * eta * pow(xi, 0.5e1) - 0.924e3 * pow(xi, 0.6e1) + 0.252e3 * pow(eta, 0.5e1) + 0.26460e5 * pow(eta, 0.4e1) * xi + 0.235200e6 * pow(eta, 0.3e1) * xi * xi + 0.352800e6 * eta * eta * pow(xi, 0.3e1) + 0.94500e5 * eta * pow(xi, 0.4e1) + 0.2772e4 * pow(xi, 0.5e1) - 0.630e3 * pow(eta, 0.4e1) - 0.23520e5 * pow(eta, 0.3e1) * xi - 0.88200e5 * eta * eta * xi * xi - 0.50400e5 * eta * pow(xi, 0.3e1) - 0.3150e4 * pow(xi, 0.4e1) + 0.560e3 * pow(eta, 0.3e1) + 0.8820e4 * eta * eta * xi + 0.12600e5 * eta * xi * xi + 0.1680e4 * pow(xi, 0.3e1) - 0.210e3 * eta * eta - 0.1260e4 * eta * xi - 0.420e3 * xi * xi + 0.30e2 * eta + 0.42e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction72(double xi, double eta) {
  double phi = 0.232848e6 * pow(eta, 0.6e1) * pow(xi, 0.5e1) - 0.582120e6 * pow(eta, 0.6e1) * pow(xi, 0.4e1) - 0.698544e6 * pow(eta, 0.5e1) * pow(xi, 0.5e1) + 0.517440e6 * pow(eta, 0.6e1) * pow(xi, 0.3e1) + 0.1746360e7 * pow(eta, 0.5e1) * pow(xi, 0.4e1) + 0.793800e6 * pow(eta, 0.4e1) * pow(xi, 0.5e1) - 0.194040e6 * pow(eta, 0.6e1) * xi * xi - 0.1552320e7 * pow(eta, 0.5e1) * pow(xi, 0.3e1) - 0.1984500e7 * pow(eta, 0.4e1) * pow(xi, 0.4e1) - 0.423360e6 * pow(eta, 0.3e1) * pow(xi, 0.5e1) + 0.27720e5 * pow(eta, 0.6e1) * xi + 0.582120e6 * pow(eta, 0.5e1) * xi * xi + 0.1764000e7 * pow(eta, 0.4e1) * pow(xi, 0.3e1) + 0.1058400e7 * pow(eta, 0.3e1) * pow(xi, 0.4e1) + 0.105840e6 * eta * eta * pow(xi, 0.5e1) - 0.924e3 * pow(eta, 0.6e1) - 0.83160e5 * pow(eta, 0.5e1) * xi - 0.661500e6 * pow(eta, 0.4e1) * xi * xi - 0.940800e6 * pow(eta, 0.3e1) * pow(xi, 0.3e1) - 0.264600e6 * eta * eta * pow(xi, 0.4e1) - 0.10584e5 * eta * pow(xi, 0.5e1) + 0.2772e4 * pow(eta, 0.5e1) + 0.94500e5 * pow(eta, 0.4e1) * xi + 0.352800e6 * pow(eta, 0.3e1) * xi * xi + 0.235200e6 * eta * eta * pow(xi, 0.3e1) + 0.26460e5 * eta * pow(xi, 0.4e1) + 0.252e3 * pow(xi, 0.5e1) - 0.3150e4 * pow(eta, 0.4e1) - 0.50400e5 * pow(eta, 0.3e1) * xi - 0.88200e5 * eta * eta * xi * xi - 0.23520e5 * eta * pow(xi, 0.3e1) - 0.630e3 * pow(xi, 0.4e1) + 0.1680e4 * pow(eta, 0.3e1) + 0.12600e5 * eta * eta * xi + 0.8820e4 * eta * xi * xi + 0.560e3 * pow(xi, 0.3e1) - 0.420e3 * eta * eta - 0.1260e4 * eta * xi - 0.210e3 * xi * xi + 0.42e2 * eta + 0.30e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction73(double xi, double eta) {
  double phi = 0.240240e6 * pow(eta, 0.7e1) * pow(xi, 0.4e1) - 0.480480e6 * pow(eta, 0.7e1) * pow(xi, 0.3e1) - 0.840840e6 * pow(eta, 0.6e1) * pow(xi, 0.4e1) + 0.308880e6 * pow(eta, 0.7e1) * xi * xi + 0.1681680e7 * pow(eta, 0.6e1) * pow(xi, 0.3e1) + 0.1164240e7 * pow(eta, 0.5e1) * pow(xi, 0.4e1) - 0.68640e5 * pow(eta, 0.7e1) * xi - 0.1081080e7 * pow(eta, 0.6e1) * xi * xi - 0.2328480e7 * pow(eta, 0.5e1) * pow(xi, 0.3e1) - 0.808500e6 * pow(eta, 0.4e1) * pow(xi, 0.4e1) + 0.3432e4 * pow(eta, 0.7e1) + 0.240240e6 * pow(eta, 0.6e1) * xi + 0.1496880e7 * pow(eta, 0.5e1) * xi * xi + 0.1617000e7 * pow(eta, 0.4e1) * pow(xi, 0.3e1) + 0.294000e6 * pow(eta, 0.3e1) * pow(xi, 0.4e1) - 0.12012e5 * pow(eta, 0.6e1) - 0.332640e6 * pow(eta, 0.5e1) * xi - 0.1039500e7 * pow(eta, 0.4e1) * xi * xi - 0.588000e6 * pow(eta, 0.3e1) * pow(xi, 0.3e1) - 0.52920e5 * eta * eta * pow(xi, 0.4e1) + 0.16632e5 * pow(eta, 0.5e1) + 0.231000e6 * pow(eta, 0.4e1) * xi + 0.378000e6 * pow(eta, 0.3e1) * xi * xi + 0.105840e6 * eta * eta * pow(xi, 0.3e1) + 0.3920e4 * eta * pow(xi, 0.4e1) - 0.11550e5 * pow(eta, 0.4e1) - 0.84000e5 * pow(eta, 0.3e1) * xi - 0.68040e5 * eta * eta * xi * xi - 0.7840e4 * eta * pow(xi, 0.3e1) - 0.70e2 * pow(xi, 0.4e1) + 0.4200e4 * pow(eta, 0.3e1) + 0.15120e5 * eta * eta * xi + 0.5040e4 * eta * xi * xi + 0.140e3 * pow(xi, 0.3e1) - 0.756e3 * eta * eta - 0.1120e4 * eta * xi - 0.90e2 * xi * xi + 0.56e2 * eta + 0.20e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction74(double xi, double eta) {
  double phi = 0.257400e6 * pow(eta, 0.8e1) * pow(xi, 0.3e1) - 0.386100e6 * pow(eta, 0.8e1) * xi * xi - 0.1029600e7 * pow(eta, 0.7e1) * pow(xi, 0.3e1) + 0.154440e6 * pow(eta, 0.8e1) * xi + 0.1544400e7 * pow(eta, 0.7e1) * xi * xi + 0.1681680e7 * pow(eta, 0.6e1) * pow(xi, 0.3e1) - 0.12870e5 * pow(eta, 0.8e1) - 0.617760e6 * pow(eta, 0.7e1) * xi - 0.2522520e7 * pow(eta, 0.6e1) * xi * xi - 0.1441440e7 * pow(eta, 0.5e1) * pow(xi, 0.3e1) + 0.51480e5 * pow(eta, 0.7e1) + 0.1009008e7 * pow(eta, 0.6e1) * xi + 0.2162160e7 * pow(eta, 0.5e1) * xi * xi + 0.693000e6 * pow(eta, 0.4e1) * pow(xi, 0.3e1) - 0.84084e5 * pow(eta, 0.6e1) - 0.864864e6 * pow(eta, 0.5e1) * xi - 0.1039500e7 * pow(eta, 0.4e1) * xi * xi - 0.184800e6 * pow(eta, 0.3e1) * pow(xi, 0.3e1) + 0.72072e5 * pow(eta, 0.5e1) + 0.415800e6 * pow(eta, 0.4e1) * xi + 0.277200e6 * pow(eta, 0.3e1) * xi * xi + 0.25200e5 * eta * eta * pow(xi, 0.3e1) - 0.34650e5 * pow(eta, 0.4e1) - 0.110880e6 * pow(eta, 0.3e1) * xi - 0.37800e5 * eta * eta * xi * xi - 0.1440e4 * eta * pow(xi, 0.3e1) + 0.9240e4 * pow(eta, 0.3e1) + 0.15120e5 * eta * eta * xi + 0.2160e4 * eta * xi * xi + 0.20e2 * pow(xi, 0.3e1) - 0.1260e4 * eta * eta - 0.864e3 * eta * xi - 0.30e2 * xi * xi + 0.72e2 * eta + 0.12e2 * xi - 0.1e1;
  return phi;
}
static double basisFunction75(double xi, double eta) {
  double phi = 0.291720e6 * pow(eta, 0.9e1) * xi * xi - 0.291720e6 * pow(eta, 0.9e1) * xi - 0.1312740e7 * pow(eta, 0.8e1) * xi * xi + 0.48620e5 * pow(eta, 0.9e1) + 0.1312740e7 * pow(eta, 0.8e1) * xi + 0.2471040e7 * pow(eta, 0.7e1) * xi * xi - 0.218790e6 * pow(eta, 0.8e1) - 0.2471040e7 * pow(eta, 0.7e1) * xi - 0.2522520e7 * pow(eta, 0.6e1) * xi * xi + 0.411840e6 * pow(eta, 0.7e1) + 0.2522520e7 * pow(eta, 0.6e1) * xi + 0.1513512e7 * pow(eta, 0.5e1) * xi * xi - 0.420420e6 * pow(eta, 0.6e1) - 0.1513512e7 * pow(eta, 0.5e1) * xi - 0.540540e6 * pow(eta, 0.4e1) * xi * xi + 0.252252e6 * pow(eta, 0.5e1) + 0.540540e6 * pow(eta, 0.4e1) * xi + 0.110880e6 * pow(eta, 0.3e1) * xi * xi - 0.90090e5 * pow(eta, 0.4e1) - 0.110880e6 * pow(eta, 0.3e1) * xi - 0.11880e5 * eta * eta * xi * xi + 0.18480e5 * pow(eta, 0.3e1) + 0.11880e5 * eta * eta * xi + 0.540e3 * eta * xi * xi - 0.1980e4 * eta * eta - 0.540e3 * eta * xi - 0.6e1 * xi * xi + 0.90e2 * eta + 0.6e1 * xi - 0.1e1;
  return phi;
}
static double basisFunction76(double xi, double eta) {
  double phi = 0.369512e6 * pow(eta, 0.10e2) * xi - 0.184756e6 * pow(eta, 0.10e2) - 0.1847560e7 * pow(eta, 0.9e1) * xi + 0.923780e6 * pow(eta, 0.9e1) + 0.3938220e7 * pow(eta, 0.8e1) * xi - 0.1969110e7 * pow(eta, 0.8e1) - 0.4667520e7 * pow(eta, 0.7e1) * xi + 0.2333760e7 * pow(eta, 0.7e1) + 0.3363360e7 * pow(eta, 0.6e1) * xi - 0.1681680e7 * pow(eta, 0.6e1) - 0.1513512e7 * pow(eta, 0.5e1) * xi + 0.756756e6 * pow(eta, 0.5e1) + 0.420420e6 * pow(eta, 0.4e1) * xi - 0.210210e6 * pow(eta, 0.4e1) - 0.68640e5 * pow(eta, 0.3e1) * xi + 0.34320e5 * pow(eta, 0.3e1) + 0.5940e4 * eta * eta * xi - 0.2970e4 * eta * eta - 0.220e3 * eta * xi + 0.110e3 * eta + 0.2e1 * xi - 0.1e1;
  return phi;
}
static double basisFunction77(double xi, double eta) {
  double phi = 0.705432e6 * pow(eta, 0.11e2) - 0.3879876e7 * pow(eta, 0.10e2) + 0.9237800e7 * pow(eta, 0.9e1) - 0.12471030e8 * pow(eta, 0.8e1) + 0.10501920e8 * pow(eta, 0.7e1) - 0.5717712e7 * pow(eta, 0.6e1) + 0.2018016e7 * pow(eta, 0.5e1) - 0.450450e6 * pow(eta, 0.4e1) + 0.60060e5 * pow(eta, 0.3e1) - 0.4290e4 * eta * eta + 0.132e3 * eta - 0.1e1;
  return phi;
}
static double (* const basisFunctions[])(double, double) = { basisFunction0,basisFunction1,basisFunction2,basisFunction3,basisFunction4,basisFunction5,basisFunction6,basisFunction7,basisFunction8,basisFunction9,basisFunction10,basisFunction11,basisFunction12,basisFunction13,basisFunction14,basisFunction15,basisFunction16,basisFunction17,basisFunction18,basisFunction19,basisFunction20,basisFunction21,basisFunction22,basisFunction23,basisFunction24,basisFunction25,basisFunction26,basisFunction27,basisFunction28,basisFunction29,basisFunction30,basisFunction31,basisFunction32,basisFunction33,basisFunction34,basisFunction35,basisFunction36,basisFunction37,basisFunction38,basisFunction39,basisFunction40,basisFunction41,basisFunction42,basisFunction43,basisFunction44,basisFunction45,basisFunction46,basisFunction47,basisFunction48,basisFunction49,basisFunction50,basisFunction51,basisFunction52,basisFunction53,basisFunction54,basisFunction55,basisFunction56,basisFunction57,basisFunction58,basisFunction59,basisFunction60,basisFunction61,basisFunction62,basisFunction63,basisFunction64,basisFunction65,basisFunction66,basisFunction67,basisFunction68,basisFunction69,basisFunction70,basisFunction71,basisFunction72,basisFunction73,basisFunction74,basisFunction75,basisFunction76,basisFunction77 };
#endif // BASISFUNCTIONS_H_
