#include "pneumatic_simulator.h"
#include <algorithm>
#include <cmath>
#include <iostream>

static PneumaticSimulator pneumatic_simulator;
static constexpr double MIN_ABS_PRESSURE_PA_SIM = 100.0;

PneumaticSimulator::PneumaticSimulator() {
    pneumaticCT = new PneumaticCT();
    xk0 = new double[X_DIM];
    observation = new double[OBS];
    mass_flowrate = new double[M_DOT_DIM];
    k1 = new double[X_DIM]; k2 = new double[X_DIM];
    k3 = new double[X_DIM]; k4 = new double[X_DIM];
    i2 = new double[X_DIM]; i3 = new double[X_DIM]; i4 = new double[X_DIM];
    sum_pump_in = 0.0;
    sum_pump_out = 0.0;
    sum_valve_pos = 0.0;
    sum_valve_neg = 0.0;
    mean_mass_flowrate[0] = 0.0;
    mean_mass_flowrate[1] = 0.0;
    mean_mass_flowrate[2] = 0.0;
    mean_mass_flowrate[3] = 0.0;

    // Initialize state vector xk0 with SI units
    xk0[0] = 0.0;          // t [s]
    xk0[1] = ATM;          // Pch_pos [Pa]
    xk0[2] = ATM;          // Pch_neg [Pa]
    xk0[3] = 0.0;          // angle [rad]
    xk0[4] = ATM;          // Ppis1 [Pa]
    xk0[5] = ATM;          // Ppis2 [Pa]
    xk0[6] = ATM;          // P1_pos [Pa]
    xk0[7] = ATM;          // P1_neg [Pa]
    xk0[8] = pneumaticCT->getActuator1().getInitialLength();  // L1 [m]
    xk0[9] = 0.0;          // v1 [m/s]                                          // v1 [m/s]
    std::cout << "[ INFO] Pneumatic Simulator ==> Initialized with Full System Model" << std::endl;
}

PneumaticSimulator::~PneumaticSimulator() {
    delete pneumaticCT;
    delete[] xk0;
    delete[] observation;
    delete[] mass_flowrate;
    delete[] k1;
    delete[] k2;
    delete[] k3;
    delete[] k4;
    delete[] i2;
    delete[] i3;
    delete[] i4;
}

PneumaticSimulator& PneumaticSimulator::get_instance() {
    return pneumatic_simulator;
}

void PneumaticSimulator::set_actuator_parameters(
    double initial_length,
    double dimension_D,
    double num_folds,
    double shaft_r,
    double rod_mass
) {
    pneumaticCT->set_actuator_parameters(initial_length, dimension_D, num_folds, shaft_r, rod_mass);
    xk0[8] = pneumaticCT->getActuator1().getInitialLength();
}

void PneumaticSimulator::set_actuator_min_length(double min_length) {
    pneumaticCT->set_actuator_min_length(min_length);
    const double L_min = pneumaticCT->getActuator1().getMinimumLength();
    const double L_max = pneumaticCT->getActuator1().getInitialLength();
    if (!std::isfinite(xk0[8]) || xk0[8] < L_min) {
        xk0[8] = L_min;
        xk0[9] = 0.0;
    } else if (xk0[8] > L_max) {
        xk0[8] = L_max;
        xk0[9] = 0.0;
    }
}

double PneumaticSimulator::get_initial_length() {
    return pneumaticCT->getActuator1().getInitialLength();
}

double PneumaticSimulator::get_min_length() {
    return pneumaticCT->getActuator1().getMinimumLength();
}

void PneumaticSimulator::set_init_env(double pos_press_kPa, double neg_press_kPa, double p1_pos_kPa, double p1_neg_kPa) {
    xk0[0] = 0.0;
    xk0[1] = pos_press_kPa * 1000.0;
    xk0[2] = neg_press_kPa * 1000.0;
    xk0[3] = 0.0;
    xk0[4] = (0.98947 * pos_press_kPa + 0.27407 * neg_press_kPa - 0.30175) * 1000.0;
    xk0[5] = (0.023686 * pos_press_kPa + 0.017746 * neg_press_kPa + 0.47436) * 1000.0;
    // xk0[6] = ATM;
    // xk0[7] = ATM;
    xk0[6] = p1_pos_kPa * 1000.0;  // P1_pos [Pa]
    xk0[7] = p1_neg_kPa * 1000.0;  // P1_neg [Pa]  
    xk0[8] = pneumaticCT->getActuator1().getInitialLength();
    xk0[9] = 0.0;
    time_reset();
}

void PneumaticSimulator::set_init_state(
    double pos_press_kPa,
    double neg_press_kPa,
    double p1_pos_kPa,
    double p1_neg_kPa,
    double length_m,
    double vel_ms
) {
    set_init_env(pos_press_kPa, neg_press_kPa, p1_pos_kPa, p1_neg_kPa);
    const double L_min = pneumaticCT->getActuator1().getMinimumLength();
    const double L_max = pneumaticCT->getActuator1().getInitialLength();
    xk0[8] = std::min(std::max(length_m, L_min), L_max);
    xk0[9] = std::isfinite(vel_ms) ? vel_ms : 0.0;
}

void PneumaticSimulator::pneumaticDT(double* xk, double* uk, double Ts, double* xk1) {
    pneumaticCT->model(xk, uk, k1, 1);
    for (int i = 0; i < X_DIM; i++) { i2[i] = xk[i] + k1[i] * Ts / 3.0; }

    pneumaticCT->model(i2, uk, k2, 2);
    for (int i = 0; i < X_DIM; i++) { i3[i] = xk[i] - (k1[i] * Ts / 3.0) + (k2[i] * Ts); }

    pneumaticCT->model(i3, uk, k3, 3);
    for (int i = 0; i < X_DIM; i++) { i4[i] = xk[i] + (k1[i] * Ts) - (k2[i] * Ts) + (k3[i] * Ts); }

    pneumaticCT->model(i4, uk, k4, 4);

    for (int i = 0; i < X_DIM; i++) {
        xk1[i] = xk[i] + (k1[i] + 3.0 * k2[i] + 3.0 * k3[i] + k4[i]) * Ts / 8.0;
    }

    const int pressure_indices[] = {1, 2, 4, 5, 6, 7};
    for (int idx : pressure_indices) {
        if (!std::isfinite(xk1[idx]) || xk1[idx] < MIN_ABS_PRESSURE_PA_SIM) {
            xk1[idx] = MIN_ABS_PRESSURE_PA_SIM;
        }
    }

    const double L_min = pneumaticCT->getActuator1().getMinimumLength();
    const double L0 = pneumaticCT->getActuator1().getInitialLength();
    if (!std::isfinite(xk1[8]) || xk1[8] < L_min) {
        xk1[8] = L_min;
        xk1[9] = 0.0;
    } else if (xk1[8] > L0) {
        xk1[8] = L0;
        xk1[9] = 0.0;
    } else if (!std::isfinite(xk1[9])) {
        xk1[9] = 0.0;
    }

    xk1[3] = fmod(xk1[3], 2 * PI);
}

double* PneumaticSimulator::step(double* control, double time_step) {
    double* xk1 = new double[X_DIM];
    int n = static_cast<int>(time_step / TS);
    sum_pump_in = 0.0;
    sum_pump_out = 0.0;
    sum_valve_pos = 0.0;
    sum_valve_neg = 0.0;
    for (int i = 0; i < n; i++) {
        pneumaticDT(xk0, control, TS, xk1);
        for (int j = 0; j < X_DIM; j++) { xk0[j] = xk1[j]; }

        double* mf = pneumaticCT->get_mass_flowrate();
        double curr_pump_out = mf[0] + mf[2];
        double curr_pump_in = mf[1] + mf[3];
        double curr_valve_pos = mf[4];
        double curr_valve_neg = mf[5];

        sum_pump_in += curr_pump_in * TS;
        sum_pump_out += curr_pump_out * TS;
        sum_valve_pos += curr_valve_pos * TS;
        sum_valve_neg += curr_valve_neg * TS;
    }

    if (time_step > 0.0) {
        mean_mass_flowrate[0] = sum_pump_in / time_step;
        mean_mass_flowrate[1] = sum_pump_out / time_step;
        mean_mass_flowrate[2] = sum_valve_pos / time_step;
        mean_mass_flowrate[3] = sum_valve_neg / time_step;
    }

    // Return observation. Pressures are converted back to kPa for convenience in Python.
    observation[0] = xk0[0];          // time [s]
    observation[1] = xk0[1] / 1000.0; // Pch_pos [kPa]
    observation[2] = xk0[2] / 1000.0; // Pch_neg [kPa]
    observation[3] = xk0[6] / 1000.0; // P1_pos [kPa]
    observation[4] = xk0[7] / 1000.0; // P1_neg [kPa]
    observation[5] = xk0[8];          // L1 [m]
    observation[6] = xk0[9];          // v1 [m/s]
    
    delete[] xk1;
    return observation;
}

void PneumaticSimulator::set_volume(double v1, double v2) {
    pneumaticCT->set_volume(v1, v2);
    std::cout << "[ INFO] Pneumatic Simulator ==> Vol initialized: POS " << v1 << " NEG " << v2 << std::endl;
}

void PneumaticSimulator::set_discharge_coeff(double c1i, double c1o, double c2i, double c2o) {
    pneumaticCT->set_discharge_coeff(c1i, c1o, c2i, c2o);
}

void PneumaticSimulator::set_leak_coefficients(double pos_atm, double neg_atm, double cross) {
    pneumaticCT->set_leak_coefficients(pos_atm, neg_atm, cross);
}

double PneumaticSimulator::get_time() {
    return observation[0];
}

double* PneumaticSimulator::get_mass_flowrate() {
    return pneumaticCT->get_mass_flowrate();
}

double* PneumaticSimulator::get_mean_mass_flowrate() {
    return mean_mass_flowrate;
}

double* PneumaticSimulator::get_valve_debug() {
    return pneumaticCT->get_valve_debug();
}

void PneumaticSimulator::time_reset() {
    xk0[0] = 0;
    sum_pump_in = 0.0;
    sum_pump_out = 0.0;
    sum_valve_pos = 0.0;
    sum_valve_neg = 0.0;
    mean_mass_flowrate[0] = 0.0;
    mean_mass_flowrate[1] = 0.0;
    mean_mass_flowrate[2] = 0.0;
    mean_mass_flowrate[3] = 0.0;
    pneumaticCT->reset_valve_states();
}

void PneumaticCT::set_logging(bool enable) {
    this -> enable_logging_ = enable;
}

void PneumaticSimulator::set_logging(bool enable) {
    pneumaticCT->set_logging(enable);
}

PneumaticCT* PneumaticSimulator::get_pneumatic_ct() {
    return this->pneumaticCT;
}

extern "C" {
    double* step_c(double* control, double time_step) {
        return PneumaticSimulator::get_instance().step(control, time_step);
    }
    void set_init_env_c(double pos_press, double neg_press, double p1_pos_kPa, double p1_neg_kPa) {
        PneumaticSimulator::get_instance().set_init_env(pos_press, neg_press, p1_pos_kPa, p1_neg_kPa);
    }
    void set_init_state_c(
        double pos_press,
        double neg_press,
        double p1_pos_kPa,
        double p1_neg_kPa,
        double length_m,
        double vel_ms
    ) {
        PneumaticSimulator::get_instance().set_init_state(
            pos_press,
            neg_press,
            p1_pos_kPa,
            p1_neg_kPa,
            length_m,
            vel_ms
        );
    }
    double* get_mass_flowrate_c() {
        return PneumaticSimulator::get_instance().get_mass_flowrate();
    }
    double* get_mean_mass_flowrate_c() {
        return PneumaticSimulator::get_instance().get_mean_mass_flowrate();
    }
    double* get_valve_debug_c() {
        return PneumaticSimulator::get_instance().get_valve_debug();
    }
    double get_time_c() {
        return PneumaticSimulator::get_instance().get_time();
    }
    void set_volume_c(double v1, double v2) {
        PneumaticSimulator::get_instance().set_volume(v1, v2);
    }

    void set_discharge_coeff_c(double c1i, double c1o, double c2i, double c2o) {
        PneumaticSimulator::get_instance().set_discharge_coeff(c1i, c1o, c2i, c2o);
    }

    void set_leak_coefficients_c(double pos_atm, double neg_atm, double cross) {
        PneumaticSimulator::get_instance().set_leak_coefficients(pos_atm, neg_atm, cross);
    }

    void time_reset_c() {
        PneumaticSimulator::get_instance().time_reset();
    }

    void set_logging_c(bool enable) {
        PneumaticSimulator::get_instance().set_logging(enable);
    }
    
}
