#ifndef PNEUMATIC_SIMULATOR_H
#define PNEUMATIC_SIMULATOR_H
#include <iostream>
#include <fstream>
#include "soft_actuator.h"

// --- 상수 정의 (SI 단위계: m, kg, s, Pa) ---
#define PI 3.14159265358979

// Contact Force Parameters
#define CONTACT_DAMPING 1e3 // [N·s/m] Damping coefficient
#define CONTACT_STIFFNESS 1e6   // [N/m] Spring coefficient

// Thermodynamics
#define R 287.0              // [J/kg·K] Specific gas constant for air
#define K 1.4                //      [-] Heat capacity ratio for air
#define ATM 101325.0         //      [Pa] Standard atmospheric pressure
#define T_ 323.15            //       [K] Piston Temperature
#define T_OUT 293.15         //       [K] Chamber/Outlet Temperature

// Chamber
#define V1_ 0.75             //       [L] Positive chamber volume (will be converted to m^3)
#define V2_ 0.4              //       [L] Negative chamber volume (will be converted to m^3)

// Pump Mechanical
#define M_W_RPM_ 3000.0      //     [rpm] Motor RPM
#define SC_R_ 0.02           //       [m] Crank radius (2cm)
#define SC_L_ 0.07           //       [m] Connecting rod length (7cm)
#define V_D_ 0.07            //       [m] Piston diameter (7cm)

// Orifice & Valve
#define O_S_ 148*0.1/180     //       [-] Matched to lib3 (currently unused)
// #define SV_D_ 0.0016         //       [m] Solenoid valve diameter (1.6mm)
// #define SV_G_ 0.00016        //       [m] Solenoid valve gap (0.16mm)
#define SV_D_ 1.6           //       [mm] Solenoid valve diameter (1.6mm)
#define SV_G_ 0.16          //       [mm] Solenoid valve gap (
#define POS_VALVE_NUM 1      //       [-] Number of positive valves
#define NEG_VALVE_NUM 1      //       [-] Number of negative valves
#define CIN_ 3.594950818595695    //       [-] Inlet discharge coeff, tuned from coeff_result/260624_11_21_35...
#define COUT_ 9.425020467026517   //       [-] Outlet discharge coeff, tuned from coeff_result/260624_11_21_35...
// Paper parameter
// #define CIN_ 1.464271612858397
// #define COUT_ 33.47453817004828

// --- 시스템 차원 정의 ---
#define X_DIM 10
#define U_DIM 6
#define OBS 7
#define M_DOT_DIM 10
#define VALVE_DEBUG_DIM 54

// --- 시뮬레이션 파라미터 ---
#define TS 0.0001           // [s] Simulation time step (for stability)

class PneumaticCT;

class PneumaticSimulator {
public:
    PneumaticSimulator();
    ~PneumaticSimulator();
    static PneumaticSimulator& get_instance();
    void set_actuator_parameters(double initial_length, double dimension_D, double num_folds, double shaft_r, double rod_mass);
    void set_actuator_min_length(double min_length);
    double get_initial_length();
    double get_min_length();
    void set_init_env(double pos_press_kPa, double neg_press_kPa, double p1_pos_kPa, double p1_neg_kPa);
    void set_init_state(
        double pos_press_kPa,
        double neg_press_kPa,
        double p1_pos_kPa,
        double p1_neg_kPa,
        double length_m,
        double vel_ms
    );
    void pneumaticDT(double* xk, double* uk, double Ts, double* xk1);
    double* step(double* control, double time_step);
    double get_time();
    double* get_mass_flowrate();
    double* get_mean_mass_flowrate();
    double* get_valve_debug();
    void set_volume(double volume1_L, double volume2_L);
    void set_discharge_coeff(double c1i, double c1o, double c2i, double c2o);
    void set_leak_coefficients(double pos_atm, double neg_atm, double cross);
    void time_reset();
    void set_logging(bool enable);

    PneumaticCT* get_pneumatic_ct();

private:
    PneumaticCT *pneumaticCT;
    double* xk0, *k, *observation, *mass_flowrate;
    double* k1, *k2, *k3, *k4, *i2, *i3, *i4;
    double sum_pump_in, sum_pump_out, sum_valve_pos, sum_valve_neg;
    double mean_mass_flowrate[4];
};

class PneumaticCT {
private:
    struct ValveRuntimeState
    {
        double z;
        double x1;
        double x2;
        double I_prev;
        double state_prev;
    };

    // Parameters (all in SI units)
    double M_W_RPM, SC_R, SC_L, V_D, V_S, V_DV, V_MAX_V;
    double O_S, SV_D, SV_G, T, V1, V2;
    double C1OUT, C1IN, C2OUT, C2IN;
    double leak_pos_atm, leak_neg_atm, leak_cross;
    
    // System state & derivatives
    double* dxdt;
    double* mass_flowrate;
    double valve_debug[VALVE_DEBUG_DIM];
    ValveRuntimeState valve_state_ch_pos;
    ValveRuntimeState valve_state_ch_neg;
    ValveRuntimeState valve_state_act_pos_in;
    ValveRuntimeState valve_state_act_pos_out;
    ValveRuntimeState valve_state_act_neg_in;
    ValveRuntimeState valve_state_act_neg_out;
    SoftActuator actuator1;

    // Physics models
    void slider_crank(double angle, double angular_velocity, double phase, double* piston);
    void volume(double piston_pos, double piston_vel, double* V_dVdt);
    double orifice(double P_inlet_Pa, double P_outlet_Pa, double Cd);
    double pressure(double P_Pa, double V_m3, double dVdt_m3s, double dmdt_kgs);
    void reset_valve_state(ValveRuntimeState* state);
    double chamber(double dmdt, double V);

    bool enable_logging_ = false;

public:
    PneumaticCT();
    ~PneumaticCT();
    void set_actuator_parameters(double initial_length, double dimension_D, double num_folds, double shaft_r, double rod_mass);
    void set_actuator_min_length(double min_length);
    void set_volume(double volume1_m3, double volume2_m3);
    void set_discharge_coeff(double Cd1IN, double Cd1OUT, double Cd2IN, double Cd2OUT);
    void set_leak_coefficients(double pos_atm, double neg_atm, double cross);
    void model(const double* x, const double* u, double* dxdt, int step_num);
    double* get_mass_flowrate();
    double* get_valve_debug();
    void reset_valve_states();
    const SoftActuator& getActuator1() const { return actuator1; }
    void set_logging(bool enable);
    double solenoid_valve(double P_inlet_Pa, double P_outlet_Pa, double signal, double type, double num);
};
#endif
