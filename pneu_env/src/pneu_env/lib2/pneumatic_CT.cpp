#include "pneumatic_simulator.h"
#include <algorithm>
#include <cmath>

namespace
{
struct ValveModelParams
{
    double A_max;
    double k_shape;
    double C_k;
    double C_p;
    double C_z;
    double A_bw;
    double beta_bw;
    double gamma_bw;
    double alpha_shape;
    double wn_up;
    double zeta_up;
    double wn_down;
    double zeta_down;
};

constexpr double STD_RHO = 1.20411831637462;
constexpr double I_MAX = 0.30;
constexpr double VALVE_DT = TS / 4.0;
constexpr int VALVE_SUB_STEPS = 20;
constexpr double STATE_EPS = 1e-4;
constexpr double FORCE_LIMIT = 500.0;
constexpr double LOG_GUARD = 700.0;
constexpr double Z_LIMIT = 1e6;

const ValveModelParams POS_PARAMS = {
    0.177485, 24.9354, 0.0918, 0.000251, 0.00000,
    363318.0739, 1.6334, 0.1516, 83.5718,
    2.3474, 0.9792, 3.6058, 1.5719
};

const ValveModelParams NEG_PARAMS = {
    0.252364, 49.2420, 0.0821, 0.000124, -0.00002,
    77568.1783, 753.6405, 0.1452, 11752.9849,
    4.5513, 2.0892, 2.4077, 0.7968
};

double clamp01(double value)
{
    return std::min(1.0, std::max(0.0, value));
}

double logaddexp_c(double x, double y)
{
    if (x == y) return x + 0.6931471805599453; // log(2)
    const double diff = x - y;
    if (diff > 0.0) return x + std::log1p(std::exp(-diff));
    return y + std::log1p(std::exp(diff));
}

double compressible_phi(double p_in_abs_kpa, double p_out_abs_kpa)
{
    const double pin = std::max(p_in_abs_kpa, 1e-9);
    const double pout = std::max(p_out_abs_kpa, 0.0);
    const double pr = clamp01(pout / pin);
    const double pcr = std::pow(2.0 / (K + 1.0), K / (K - 1.0));

    if (pr <= pcr) {
        return std::sqrt(K * std::pow(2.0 / (K + 1.0), (K + 1.0) / (K - 1.0)));
    }

    if (pr <= 1.0) {
        const double term = std::pow(pr, 2.0 / K) - std::pow(pr, (K + 1.0) / K);
        return std::sqrt(2.0 * K / (K - 1.0)) * std::sqrt(std::max(term, 0.0));
    }

    return 0.0;
}
} // namespace

PneumaticCT::PneumaticCT()
{
    std::cout << "[ INFO] Pneumatic CT ==> Initialized" << std::endl;
    M_W_RPM = M_W_RPM_; 

    SC_R = SC_R_;
    SC_L = SC_L_;

    V_D = V_D_; // cm
    V_S = 0.25*PI*V_D*V_D; // cm^2
    V_DV = 0.001*0.1*V_S; // L
    V_MAX_V = 0.001*2*SC_R*V_S + V_DV; // L
    volume_ratio = V_MAX_V/V_DV;

    O_S = O_S_;

    SV_D = SV_D_;
    SV_G = SV_G_;

    T = T_;
    V1 = V1_;
    V2 = V2_;

    // C1IN = C1IN_;
    // C1OUT = C1OUT_;
    // C2IN = C2IN_;
    // C2OUT = C2OUT_;

    C1IN = 1e-6*CIN_;
    C1OUT = 1e-6*COUT_;
    C2IN = 1e-6*CIN_;
    C2OUT = 1e-6*COUT_;

    std::cout << "[ INFO] Pneumatic Simulator ==> Discharge Coefficient Initialized" << std::endl;
    std::cout << "[ INFO] C1IN : " << C1IN << " C1OUT: " << C1OUT << std::endl;
    std::cout << "[ INFO] C2IN : " << C2IN << " C2OUT: " << C2OUT << std::endl;

    dxdt = new double[6];
    mass_flowrate = new double[6];
    reset_valve_states();
}

void PneumaticCT::reset_valve_state(ValveRuntimeState* state)
{
    state->z = 0.0;
    state->x1 = 0.0;
    state->x2 = 0.0;
    state->I_prev = 0.0;
    state->state_prev = 0.0;
}

void PneumaticCT::reset_valve_states()
{
    reset_valve_state(&valve_state_pos);
    reset_valve_state(&valve_state_neg);
}

void PneumaticCT::set_volume(double volume1, double volume2)
{
    V1 = volume1;
    V2 = volume2;
}

void PneumaticCT::set_discharge_coeff(
    double Cd1IN,
    double Cd1OUT,
    double Cd2IN,
    double Cd2OUT
)
{
    C1IN = Cd1IN;
    C1OUT = Cd1OUT;
    C2IN = Cd2IN;
    C2OUT = Cd2OUT;
}

double* PneumaticCT::model(double* x, double* u)
{
    double* piston1 = new double[2];
    double* piston2 = new double[2];
    double* V1_dV1dt = new double[2];
    double* V2_dV2dt = new double[2];

    double Ppos = x[1];
    double Pneg = x[2];
    double angle = x[3];
    double P1 = x[4];
    double P2 = x[5];

    // // Change 1
    // double m1 = x[4];
    // double m2 = x[5];
    // // =====

    double pos_valve = u[0];
    double neg_valve = u[1];

    double angular_velocity = M_W_RPM * (2*PI/60); // rpm -> rad/s

    this -> slider_crank(angle, angular_velocity, 0, piston1); // cm, cm/s
    this -> slider_crank(angle, angular_velocity, PI, piston2);

    this -> volume(piston1[0], piston1[1], V1_dV1dt);
    this -> volume(piston2[0], piston2[1], V2_dV2dt);

    double dm1outdt = this -> orifice(P1, Ppos, C1OUT);
    double dm1indt = this -> orifice(Pneg, P1, C1IN);
    double dm2outdt = this -> orifice(P2, Ppos, C2OUT); 
    double dm2indt = this -> orifice(Pneg, P2, C2IN);

    double dP1dt = this -> pressure(P1, V1_dV1dt[0], V1_dV1dt[1], dm1indt - dm1outdt);
    double dP2dt = this -> pressure(P2, V2_dV2dt[0], V2_dV2dt[1], dm2indt - dm2outdt);

    double mdot_pos_valve = this -> solenoid_valve(Ppos, ATM, pos_valve, 1, POS_VALVE_NUM); // 1 means positive valve
    double mdot_neg_valve = this -> solenoid_valve(ATM, Pneg, neg_valve, 2, NEG_VALVE_NUM); // 2 means negative valve

    double mdot_out = dm1outdt + dm2outdt - mdot_pos_valve;
    double mdot_in = - dm1indt - dm2indt + mdot_neg_valve;
    
    double dPposdt = this -> chamber(mdot_out, V1);
    double dPnegdt = this -> chamber(mdot_in, V2);

    dxdt[0] = 1;
    dxdt[1] = dPposdt;
    dxdt[2] = dPnegdt;
    dxdt[3] = angular_velocity; // rad/s
    dxdt[4] = dP1dt;
    dxdt[5] = dP2dt;

    delete[] piston1;
    delete[] piston2;
    delete[] V1_dV1dt;
    delete[] V2_dV2dt;

    mass_flowrate[0] = dm1outdt + dm2outdt;
    mass_flowrate[1] = dm1indt + dm2indt;
    mass_flowrate[2] = mdot_pos_valve;
    mass_flowrate[3] = mdot_neg_valve;
    mass_flowrate[4] = mdot_out;
    mass_flowrate[5] = mdot_in;

    return dxdt;
}

void PneumaticCT::slider_crank(double angle, double angular_velocity, double phase, double* piston)
{
    double pos = SC_R*cos(angle + phase) + sqrt(SC_L*SC_L - pow(SC_R*sin(angle + phase), 2)) + SC_R - SC_L;
    double vel = (- SC_R*sin(angle + phase) - SC_R*SC_R*sin(angle + phase)*cos(angle + phase)/sqrt(SC_L*SC_L - pow(SC_R*sin(angle + phase), 2)))*angular_velocity;

    piston[0] = pos; // cm
    piston[1] = vel; // cm/s
}

void PneumaticCT::volume(double piston_pos, double piston_vel, double* V_dVdt)
{
    double V = - 0.001*V_S*piston_pos + V_MAX_V;
    double dVdt = - 0.001*V_S*piston_vel;

    V_dVdt[0] = V;
    V_dVdt[1] = dVdt;
}

double PneumaticCT::orifice(double P_inlet, double P_outlet, double Cd)
{
    double dmdt;

    double Pcr = pow(2/(K + 1), K/(K - 1));
    double Pin = 1000*P_inlet;
    double Pout = 1000*P_outlet;

    if (Pin >= Pout) 
    {
        if (Pout/Pin <= Pcr) {
            // dmdt = 1e-4*O_S * (Pin/sqrt(1000*R*T)) * sqrt(K*pow(2/(K + 1), (K + 1)/(K - 1)));
            dmdt = (Pin/sqrt(1000*R*T)) * sqrt(K*pow(2/(K + 1), (K + 1)/(K - 1)));
        } else {
            // dmdt = 1e-4*O_S * (Pin/sqrt(1000*R*T)) * sqrt(2*K/(K - 1)) * sqrt(pow(Pout/Pin, 2/K) - pow(Pout/Pin, (K + 1)/K));
            dmdt = (Pin/sqrt(1000*R*T)) * sqrt(2*K/(K - 1)) * sqrt(pow(Pout/Pin, 2/K) - pow(Pout/Pin, (K + 1)/K));
        }
    } else {
        dmdt = 0;
    }

    return Cd*dmdt;
}

double PneumaticCT::pressure(double P, double V, double dVdt, double dmdt)
{
    double dPdt;

    dPdt = - P*dVdt/V + 1000*R*T*dmdt/V;

    return dPdt;
}

double PneumaticCT::solenoid_valve(double P_inlet, double P_outlet, double signal, double type, double num)
{
    const bool is_pos = (type == 1.0);
    ValveRuntimeState* state = is_pos ? &valve_state_pos : &valve_state_neg;
    const ValveModelParams& params = is_pos ? POS_PARAMS : NEG_PARAMS;

    const double signal_clipped = clamp01(signal);
    const double u_eff = clamp01((signal_clipped - 0.5) * 2.0);
    const double current = I_MAX * u_eff;

    double state_curr = state->state_prev;
    if (current > state->I_prev + STATE_EPS) {
        state_curr = 1.0;
    } else if (current < state->I_prev - STATE_EPS) {
        state_curr = 0.0;
    }

    const double abs_dI = std::abs(current - state->I_prev);
    const double dI = abs_dI * (2.0 * state_curr - 1.0);

    const double dz = (
        params.A_bw * dI
        - params.beta_bw * std::abs(dI) * state->z
        - params.gamma_bw * dI * std::abs(state->z)
    );
    state->z += dz;
    state->z = std::min(Z_LIMIT, std::max(-Z_LIMIT, state->z));

    double force_net = current + params.C_z * state->z + params.C_p * P_inlet - params.C_k;
    force_net = std::min(FORCE_LIMIT, std::max(-FORCE_LIMIT, force_net));

    const double exp_arg = -params.k_shape * force_net;
    const double log_denom = params.alpha_shape * logaddexp_c(0.0, exp_arg);

    double area_eff = 0.0;
    if (log_denom <= LOG_GUARD) {
        area_eff = params.A_max * std::exp(-log_denom);
    }

    const double phi = compressible_phi(P_inlet, P_outlet);
    const double q_static_lpm = area_eff * P_inlet * phi;

    const double wn = (state_curr >= 0.5) ? params.wn_up : params.wn_down;
    const double zeta = (state_curr >= 0.5) ? params.zeta_up : params.zeta_down;
    const double dt_sub = VALVE_DT / static_cast<double>(VALVE_SUB_STEPS);

    for (int i = 0; i < VALVE_SUB_STEPS; i++) {
        const double dx1 = state->x2;
        const double dx2 = wn * wn * (q_static_lpm - state->x1) - 2.0 * zeta * wn * state->x2;
        state->x1 += dt_sub * dx1;
        state->x2 += dt_sub * dx2;
    }

    state->I_prev = current;
    state->state_prev = state_curr;

    const double q_pred_lpm = std::max(state->x1, 0.0);
    const double mdot = q_pred_lpm * STD_RHO / 60000.0;
    if (!std::isfinite(mdot)) return 0.0;

    return std::max(num * mdot, 0.0);
}

double PneumaticCT::chamber(double dmdt, double V)
{
    double dPdt = R*T_OUT*dmdt*1000/V;

    return dPdt;
}

double* PneumaticCT::get_mass_flowrate() { return mass_flowrate; }
