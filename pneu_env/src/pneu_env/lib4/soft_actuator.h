#ifndef SOFT_ACTUATOR_H
#define SOFT_ACTUATOR_H
#include <cmath>
class SoftActuator {
public:
    SoftActuator();
    ~SoftActuator();
    void initializeParameters(double initial_length, double dimension_D, double num_folds, double shaft_r, double rod_mass);
    double getVolume_neg(double current_L) const;
    double getVolume_pos(double current_L) const;
    double getVolumeDerivative_neg(double current_L) const;
    double getVolumeDerivative_pos(double current_L) const;
    double getForce_neg(double current_L, double P_neg) const;
    double getForce_pos(double current_L, double P_pos) const;
    double getContractionRatio(double current_L) const;
    double getRodMass() const;
    double getInitialLength() const;
    double getMinimumLength() const;
    double clampLength(double current_L) const;
    void setMinimumLength(double minimum_length);
private:
    double L0, D, n_fold, shaft_radius, m_rod, min_length;
    double L_to_theta(double current_L) const;
};
#endif
