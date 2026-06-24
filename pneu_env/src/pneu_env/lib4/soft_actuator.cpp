#include "soft_actuator.h"
#include "pneumatic_simulator.h" 
#include <stdexcept>
#include <iostream>
#include <algorithm> // For std::max


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const double EPSILON = 1e-9;

// Constructor: Initializes with default SI unit values.
SoftActuator::SoftActuator()
    : L0(0.02), D(0.02), n_fold(2.0), shaft_radius(0.005), m_rod(5.0), min_length(0.0) {}

SoftActuator::~SoftActuator() {}

// Initializes parameters with provided SI unit values
void SoftActuator::initializeParameters(double il, double d, double nf, double sr, double rm) {
    this->L0 = il; this->D = d; this->n_fold = nf; this->shaft_radius = sr; this->m_rod = rm;
    this->min_length = std::min(std::max(this->min_length, 0.0), this->L0);
}

// Returns dimensionless contraction ratio
double SoftActuator::getContractionRatio(double cL) const {
    if (L0 < EPSILON) {
        return 0.0;
    }
    return (L0 - clampLength(cL)) / L0;
}

// Returns rod mass in [kg]
double SoftActuator::getRodMass() const {
    return this->m_rod;
}

// Returns initial length in [m]
double SoftActuator::getInitialLength() const {
    return this->L0;
}

// Returns minimum physically reachable cell length in [m]
double SoftActuator::getMinimumLength() const {
    return this->min_length;
}

// Sets minimum physically reachable cell length in [m]
void SoftActuator::setMinimumLength(double minimum_length) {
    this->min_length = std::min(std::max(minimum_length, 0.0), this->L0);
}

// Clamps cell length to the actuator's physical model range
double SoftActuator::clampLength(double cL) const {
    return std::min(std::max(cL, this->min_length), this->L0);
}

// Returns negative chamber volume in [m^3] - No logical change
double SoftActuator::getVolume_neg(double cL) const {
    cL = clampLength(cL);
    double r = getContractionRatio(cL);

    if (r <= 0.3634) {
        double th = L_to_theta(cL);
        if (th < EPSILON) return n_fold * 2 * D * D * L0 * (1 + sqrt(2));
        double Vo = 2 * D * D * L0 * (1 + sqrt(2))*(sin(th)/ th);
        double Vf = (2 * D * L0 * L0 / th) * (1 - (sin(th) * cos(th) / th));
        return n_fold * (Vo - Vf);
    } else {
        double Lh = (2 * L0 - cL * M_PI) / 4.0;
        double Vo = 2 * D * D * cL * (1 + sqrt(2));
        double Vf = cL * cL * M_PI * D + 8 * Lh * cL * D;
        return n_fold * (Vo - Vf);
    }
}

// Returns positive chamber volume in [m^3] - No logical change
double SoftActuator::getVolume_pos(double cL) const {
    cL = clampLength(cL);
    const double V_DEAD_SPACE = 6.19761e-5;

    double th = L_to_theta(cL);
    double calculated_volume;
    double r = getContractionRatio(cL);

    if (fabs(th) < EPSILON) {
        return V_DEAD_SPACE;
    }

    if (r <= 0.3634) {
        double Ao = 2 * D * D * (1 + sqrt(2));
        double As = M_PI * shaft_radius * shaft_radius;
        double Vf = (2 * D * L0 * L0 / th) * (1 - (sin(th) * cos(th) / th));
        calculated_volume = n_fold * ((L0 - cL) * (Ao - As) + Vf);
    } else {
        double Lh = (2 * L0 - cL * M_PI) / 4.0;
        double Ao = 2 * D * D * (1 + sqrt(2));
        double As = M_PI * shaft_radius * shaft_radius;
        double Vf = cL * cL * M_PI * D + 8 * Lh * cL * D;
        calculated_volume = n_fold * ((L0 - cL) * (Ao - As) + Vf);
    }

    return std::max(calculated_volume, 0.0) + V_DEAD_SPACE;
}

// Returns dV_neg/dL in [m^2] - MODIFIED
double SoftActuator::getVolumeDerivative_neg(double cL) const {
    cL = clampLength(cL);
    double r = getContractionRatio(cL);
    if (r <= 0.3634) {
        double th = L_to_theta(cL);

        if (fabs(th) < 0.1*M_PI / 180.0) {
            return 1;
            // return -1;
        }

        double t1 = 2 * D * D * (1 + sqrt(2));
        double t2d = th * cos(th) - sin(th);

        if (fabs(t2d) < EPSILON) return 1.0; // Avoid division by zero

        double t2n = 2 * D * L0 * (1 + cos(2 * th) - sin(2 * th) / th);

        return n_fold * (t1 + t2n / t2d);
        // return -n_fold * (t1 + t2n / t2d);
    }
    else {

        return n_fold * (2 * D * D * (1 + sqrt(2)) - 4 * L0 * D + 2 * cL * D * M_PI);
        // return -n_fold * (2 * D * D * (1 + sqrt(2)) - 4 * L0 * D + 2 * cL * D * M_PI);
    }
}

// Returns dV_pos/dL in [m^2] - MODIFIED
double SoftActuator::getVolumeDerivative_pos(double cL) const {
    cL = clampLength(cL);
    double r = getContractionRatio(cL);
    if (r <= 0.3634) {
        double th = L_to_theta(cL);

        if (fabs(th) < 0.1* M_PI / 180.0) {
            return -1;
            // return 1;
        }
    
        double t1 = -(2 * D * D * (1 + sqrt(2)) - M_PI * shaft_radius * shaft_radius);
        double t2d = th * cos(th) - sin(th);
        if (fabs(t2d) < EPSILON) return -1.0; // Avoid division by zero
        double t2n = 2 * D * L0 * (1 + cos(2 * th) - sin(2 * th) / th);
        // Changed to subtraction and t1 is negated
        return n_fold * (t1 - t2n / t2d);
        // return -n_fold * (t1 - t2n / t2d);
    } else {
        double t1 = -(2 * D * D * (1 + sqrt(2)) - M_PI * shaft_radius * shaft_radius);
        // Flipped signs of last two terms
        return n_fold * (t1 + 4 * L0 * D - 2 * cL * D * M_PI);
        // return -n_fold * (t1 + 4 * L0 * D - 2 * cL * D * M_PI);
    }
}

// Returns force from negative chamber in [N]
double SoftActuator::getForce_neg(double cL, double Pna_abs) const {
    cL = clampLength(cL);
    const double P_atm = 101325; // ATM constant from header
    double Pna = Pna_abs - P_atm; // Gauge pressure

    return -Pna * getVolumeDerivative_neg(cL);
}

// Returns force from positive chamber in [N]
double SoftActuator::getForce_pos(double cL, double Ppa_abs) const {
    cL = clampLength(cL);
    const double P_atm = 101325; // ATM constant from header
    double Ppa = Ppa_abs - P_atm; // Gauge pressure

    return -Ppa * getVolumeDerivative_pos(cL);
}

// Returns theta [rad] from length L [m]
double SoftActuator::L_to_theta(double cL) const {
    cL = clampLength(cL);
    if (cL >= L0 - EPSILON) {
        if (cL > L0 + EPSILON) { // Add tolerance
            // std::cerr << "[WARNING] cL > L0. Safety rail triggered. cL=" << cL << ", L0=" << L0 << std::endl;
        }
        return 0.0;
    }
    if (cL <= 0.0) {
        return M_PI;
    }

    double Lr = cL / L0;
    double th = sqrt(6.0 * (1.0 - Lr));

    for (int i = 0; i < 20; ++i) {
        double f = Lr * th - sin(th);
        if (fabs(f) < EPSILON) break;
        double df = Lr - cos(th);
        if (fabs(df) < EPSILON) break;
        th = th - f / df;
    }
    return th;
}
