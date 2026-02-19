#include <algorithm>
#include <cmath>
#include <iostream>

// Standard Normal CDF
double norm_cdf(double x) { return 0.5 * std::erfc(-x * M_SQRT1_2); }

#ifndef M_1_SQRT2PI
#define M_1_SQRT2PI 0.39894228040143267794
#endif

// Standard Normal PDF
double norm_pdf(double x) { return M_1_SQRT2PI * std::exp(-0.5 * x * x); }

extern "C" {

double black_scholes_price(double S, double K, double T, double r, double sigma,
                           int is_call) {
  if (T <= 0) {
    return is_call ? std::max(0.0, S - K) : std::max(0.0, K - S);
  }

  double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) /
              (sigma * std::sqrt(T));
  double d2 = d1 - sigma * std::sqrt(T);

  if (is_call) {
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
  } else {
    return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
  }
}

double black_scholes_vega(double S, double K, double T, double r,
                          double sigma) {
  if (T <= 0)
    return 0.0;
  double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) /
              (sigma * std::sqrt(T));
  return S * std::sqrt(T) * norm_pdf(d1);
}

double black_scholes_gamma(double S, double K, double T, double r,
                           double sigma) {
  if (T <= 0)
    return 0.0;
  double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) /
              (sigma * std::sqrt(T));
  return norm_pdf(d1) / (S * sigma * std::sqrt(T));
}

double black_scholes_delta(double S, double K, double T, double r, double sigma,
                           int is_call) {
  if (T <= 0) {
    if (is_call)
      return S > K ? 1.0 : 0.0;
    else
      return S < K ? -1.0 : 0.0;
  }
  double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) /
              (sigma * std::sqrt(T));
  return is_call ? norm_cdf(d1) : (norm_cdf(d1) - 1.0);
}

double black_scholes_theta(double S, double K, double T, double r, double sigma,
                           int is_call) {
  if (T <= 0)
    return 0.0;
  double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) /
              (sigma * std::sqrt(T));
  double d2 = d1 - sigma * std::sqrt(T);

  double term1 = -(S * norm_pdf(d1) * sigma) / (2 * std::sqrt(T));
  double term2 = r * K * std::exp(-r * T) * norm_cdf(d2);
  double term3 = r * K * std::exp(-r * T) * norm_cdf(-d2);

  if (is_call) {
    return (term1 - term2) / 365.0; // Daily theta
  } else {
    return (term1 + term3) / 365.0; // Daily theta
  }
}

// Implied Volatility using Newton-Raphson
double implied_volatility(double target_price, double S, double K, double T,
                          double r, int is_call) {
  double sigma = 0.5; // Initial guess
  double epsilon = 1e-6;
  int max_iter = 100;

  // Bounds check
  double intrinsic = is_call ? std::max(0.0, S - K) : std::max(0.0, K - S);
  if (target_price <= intrinsic + 1e-5)
    return 0.0;

  for (int i = 0; i < max_iter; ++i) {
    double price = black_scholes_price(S, K, T, r, sigma, is_call);
    double vega = black_scholes_vega(S, K, T, r, sigma);

    double diff = target_price - price;

    if (std::abs(diff) < epsilon)
      return sigma;

    if (std::abs(vega) < 1e-8)
      break; // Avoid division by zero

    sigma += diff / vega;

    // Clamp sigma to reasonable bounds
    if (sigma < 0.001)
      sigma = 0.001;
    if (sigma > 5.0)
      sigma = 5.0; // Max 500% vol
  }

  return sigma; // Return best guess if not converged
}
}
