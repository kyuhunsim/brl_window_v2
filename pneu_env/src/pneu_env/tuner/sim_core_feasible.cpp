// Experimental calibration core. Not used by the deployed lib3 simulator.
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
// h=z*(beta+gamma)/A_bw, so |h|<=1 is invariant for this parameterization.
// Integrate in abs(delta current), exactly within each sign(h) branch.
double advance_h(double h, double delta, double rate, double beta_fraction) {
    if (delta == 0.0) return h;
    const double sign = delta > 0.0 ? 1.0 : -1.0;
    double remaining = std::abs(delta);
    const double source = sign * rate;
    if (h * sign < 0.0) {
        const double b = rate * (2.0 * beta_fraction - 1.0);
        const double crossing = std::abs(b) < 1e-12
            ? -h / source : std::log1p(-b * h / source) / b;
        if (crossing <= remaining) {
            h = 0.0;
            remaining -= crossing;
        } else {
            return std::abs(b) < 1e-12
                ? h + source * remaining
                : h * std::exp(-b * remaining)
                    + source * (-std::expm1(-b * remaining)) / b;
        }
    }
    return h * std::exp(-rate * remaining)
        + sign * (-std::expm1(-rate * remaining));
}
}

extern "C" {
double advance_h_feasible(double h, double delta, double rate, double beta_fraction) {
    return advance_h(h, delta, rate, beta_fraction);
}

// p uses the original 13-parameter order, with A_bw=beta+gamma and h replacing z.
// Return codes: 1 invalid parameters, 2 invalid h, 3 force guard, 4 invalid flow.
// There is no dataset-mean startup penalty and no state/force clipping.
int simulate_feasible(int n, const double* current, const double* direction,
    const double* pin, const double* phi, const double* measured, double dt,
    const double* p, double* qout, double* hout, double* areaout, double* forceout) {
    const double rate = p[6] + p[7];
    if (n < 1 || rate <= 0 || p[6] < p[7] || p[7] < 0
        || p[0] <= 0 || p[1] <= 0 || p[2] < 0 || p[8] <= 0
        || std::abs(p[5]-rate) > 1e-9 * (1+rate)) return 1;
    double h = 0, x1 = measured[0], x2 = 0;
    for (int k = 0; k < n; ++k) {
        if (k) h = advance_h(h, current[k]-current[k-1], rate, p[6]/rate);
        if (!std::isfinite(h) || std::abs(h) > 1+1e-10) return 2;
        const double force = current[k] + p[4]*h + p[3]*pin[k] - p[2];
        if (!std::isfinite(force) || std::abs(force) > 500) return 3;
        const double exponent = -p[1]*force;
        const double logdenom = p[8] * (std::max(0.0, exponent)
            + std::log1p(std::exp(-std::abs(exponent))));
        const double area = std::exp(-logdenom);
        const double target = p[0]*area*pin[k]*phi[k];
        const double wn = direction[k] == 1 ? p[9] : p[11];
        const double damping = direction[k] == 1 ? p[10] : p[12];
        // Preserve the previous flow dynamics discretization for this experiment.
        for (int sub = 0; sub < 20; ++sub) {
            const double dx2 = wn*wn*(target-x1)-2*damping*wn*x2;
            x1 += dt/20*x2;
            x2 += dt/20*dx2;
        }
        if (!std::isfinite(x1) || x1 < -1e-9) return 4;
        qout[k] = x1;
        hout[k] = h;
        areaout[k] = area;
        forceout[k] = force;
    }
    return 0;
}
}
