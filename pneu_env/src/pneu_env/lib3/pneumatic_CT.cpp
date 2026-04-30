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


const ValveModelParams CHAMBER_POS_PARAMS = {
    0.173765782728, 30.9289402344, 0.116436719311, 5.46783479371e-05, 1.03479472071e-06,
    374001.449832, 1.41144546597, 0.166872715454, 61.6216292581,
    2.84730702687, 1.22289656724, 1.9018360436, 0.344906637713
};


const ValveModelParams CHAMBER_NEG_PARAMS = {
    0.242692402717, 47.8401510535, 0.0847812469387, 0.000229289814369, 9.77560733274e-11,
    29699.940435, 320.125442332, 0.0566339612952, 11464.8641164,
    14.6808864961, 7.41271437975, 25.42294934, 0.466540190699
};


const ValveModelParams ACT_POS_IN_PARAMS = {
    0.176615662976, 35.8567971705, 0.0410902906033, 0.000125140241968, 3.10897782258e-06,
    342765.237631, 1.55234148422, 0.0931140221163, 40009.5348198,
    2.05383080476, 1.45653972629, 1.36538371888, 0.932842101505
};


const ValveModelParams ACT_POS_OUT_PARAMS = {
    0.141147778024, 21.6085736731, 0.050353667423, 0.00030994966622, 5.99171947936e-06,
    287636.806294, 8.82854815565e-14, 0.0207782439013, 841.089918154,
    44.6797853561, 8.31559275949, 49.9426537221, 1.00609852673
};


const ValveModelParams ACT_NEG_IN_PARAMS = {
    0.166803861296, 48.3965822138, 0.125721792181, 0.000133474187013, 7.67638801282e-06,
    47065.7506453, 1.29345836639, 2.80378931006, 2586.83453211,
    3.04249012193, 1.11227644648, 2.29343565369, 0.901083780449
};


const ValveModelParams ACT_NEG_OUT_PARAMS = {
    0.171456892317, 55.3406762135, 0.00325287760306, 0.000401485931598, 0.000153626179729,
    3950.78516302, 5.14936971591e-12, 0.212289034266, 654501.778604,
    2.62948474457, 1.18983816911, 4.04483946577, 1.53568873742
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

    O_S = O_S_;

    SV_D = SV_D_;
    SV_G = SV_G_;

    T = T_;
    V1 = V1_;
    V2 = V2_;
    V1P = V1P_;
    V1N = V1N_;

    C1IN = 1e-6*CIN_;
    C1OUT = 1e-6*COUT_;
    C2IN = 1e-6*CIN_;
    C2OUT = 1e-6*COUT_;

    dxdt = new double[X_DIM];
    mass_flowrate = new double[M_DOT_DIM];
    reset_valve_states();
}

PneumaticCT::~PneumaticCT() { delete[] dxdt; delete[] mass_flowrate; }

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
    reset_valve_state(&valve_state_ch_pos);
    reset_valve_state(&valve_state_ch_neg);
    reset_valve_state(&valve_state_act_pos_in);
    reset_valve_state(&valve_state_act_pos_out);
    reset_valve_state(&valve_state_act_neg_in);
    reset_valve_state(&valve_state_act_neg_out);
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

void PneumaticCT::model(const double* x, const double* u, double* dxdt, int step_num)
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
    double P1_pos = x[6];
    double P1_neg = x[7];

    double u_ch_pos = u[0];
    double u_ch_neg = u[1];
    double u11_pos = u[2];
    double u12_pos = u[3];
    double u11_neg = u[4];
    double u12_neg = u[5];

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

    double mdot_pos_valve = this -> solenoid_valve(Ppos, ATM, u_ch_pos, 1, POS_VALVE_NUM);
    double mdot_neg_valve = this -> solenoid_valve(ATM, Pneg, u_ch_neg, 2, NEG_VALVE_NUM);

    double m_sv11p = this -> solenoid_valve(Ppos, P1_pos, u11_pos, 3, 1);
    double m_sv12p = this -> solenoid_valve(P1_pos, ATM, u12_pos, 4, 1);
    double m_sv11n = this -> solenoid_valve(ATM, P1_neg, u11_neg, 5, 1);
    double m_sv12n = this -> solenoid_valve(P1_neg, Pneg, u12_neg, 6, 1);

    double m_act_pos_net = m_sv11p - m_sv12p;
    double m_act_neg_net = m_sv11n - m_sv12n;

    double mdot_out = dm1outdt + dm2outdt - mdot_pos_valve - m_sv11p;
    double mdot_in = - dm1indt - dm2indt + mdot_neg_valve + m_sv12n;
    
    double dPposdt = this -> chamber(mdot_out, V1);
    double dPnegdt = this -> chamber(mdot_in, V2);

    double dP1posdt = this -> pressure(P1_pos, V1P, 0.0, m_act_pos_net);
    double dP1negdt = this -> pressure(P1_neg, V1N, 0.0, m_act_neg_net);

    dxdt[0] = 1;
    dxdt[1] = dPposdt;
    dxdt[2] = dPnegdt;
    dxdt[3] = angular_velocity; // rad/s
    dxdt[4] = dP1dt;
    dxdt[5] = dP2dt;
    dxdt[6] = dP1posdt;
    dxdt[7] = dP1negdt;

    if (this->enable_logging_) {
        const char* step_label =
            (step_num == 1) ? "k1" :
            (step_num == 2) ? "k2" :
            (step_num == 3) ? "k3" : "k4";
        std::cout << "\n===== [LOG START] Time: " << x[0]
                  << "s, RK Step: " << step_label
                  << " =====" << std::endl;

        std::cout << "INPUTS:" << std::endl;
        std::cout << "  - Pressures [kPa]: Pch_p=" << Ppos << ", Pch_n=" << Pneg << std::endl;
        std::cout << "  - Pump State [kPa]: Ppis1=" << P1 << ", Ppis2=" << P2 << std::endl;
        std::cout << "  - Act1 State [kPa]: P1_p=" << P1_pos << ", P1_n=" << P1_neg << std::endl;
        std::cout << "  - Controls: u_ch_p=" << u_ch_pos
                  << ", u_ch_n=" << u_ch_neg
                  << ", u11p=" << u11_pos
                  << ", u12p=" << u12_pos
                  << ", u11n=" << u11_neg
                  << ", u12n=" << u12_neg << std::endl;

        std::cout << "MASS FLOW [kg/s]:" << std::endl;
        std::cout << "  m_dot_net_pos_ch = " << dm1outdt + dm2outdt
                  << " - " << mdot_pos_valve
                  << " - " << m_sv11p
                  << " = " << mdot_out << std::endl;
        std::cout << "  m_dot_net_neg_ch = " << mdot_neg_valve
                  << " + " << m_sv12n
                  << " - (" << dm1indt << " + " << dm2indt << ")"
                  << " = " << mdot_in << std::endl;
        std::cout << "  - Pump: m_po1=" << dm1outdt << ", m_pi1=" << dm1indt
                  << ", m_po2=" << dm2outdt << ", m_pi2=" << dm2indt << std::endl;
        std::cout << "  - Chamber->Atmos: m_sv_cp=" << mdot_pos_valve
                  << ", m_sv_cn=" << mdot_neg_valve << std::endl;
        std::cout << "  - Act1 Valves: m_sv11p=" << m_sv11p
                  << ", m_sv12p=" << m_sv12p
                  << ", m_sv11n=" << m_sv11n
                  << ", m_sv12n=" << m_sv12n << std::endl;
        std::cout << "  - Act1 Net: m_act_pos_net=" << m_act_pos_net
                  << ", m_act_neg_net=" << m_act_neg_net << std::endl;

        std::cout << "ACTUATOR MODEL (FIXED VOLUME):" << std::endl;
        std::cout << "  - Volumes [L]: V1P=" << V1P << ", V1N=" << V1N << std::endl;
        std::cout << "  - Volume Rates [L/s]: V1P_dot=0, V1N_dot=0" << std::endl;

        std::cout << "PRESSURE DERIVATIVES [kPa/s]:" << std::endl;
        std::cout << "  - dPch_p: " << dPposdt
                  << " (RTmdot=" << (R * T_OUT * mdot_out * 1000)
                  << ", PVdot=0, V_total=" << V1 << ")" << std::endl;
        std::cout << "  - dPch_n: " << dPnegdt
                  << " (RTmdot=" << (R * T_OUT * mdot_in * 1000)
                  << ", PVdot=0, V_total=" << V2 << ")" << std::endl;
        std::cout << "  - dPpis1: " << dP1dt
                  << " (RTmdot=" << (1000 * R * T * (dm1indt - dm1outdt))
                  << ", PVdot=" << (P1 * V1_dV1dt[1])
                  << ", V_total=" << V1_dV1dt[0] << ")" << std::endl;
        std::cout << "  - dPpis2: " << dP2dt
                  << " (RTmdot=" << (1000 * R * T * (dm2indt - dm2outdt))
                  << ", PVdot=" << (P2 * V2_dV2dt[1])
                  << ", V_total=" << V2_dV2dt[0] << ")" << std::endl;
        std::cout << "  - dP1p:   " << dP1posdt
                  << " (RTmdot=" << (1000 * R * T * m_act_pos_net)
                  << ", PVdot=0, V_total=" << V1P << ")" << std::endl;
        std::cout << "  - dP1n:   " << dP1negdt
                  << " (RTmdot=" << (1000 * R * T * m_act_neg_net)
                  << ", PVdot=0, V_total=" << V1N << ")" << std::endl;

        std::cout << "OUTPUT (dxdt): [";
        for (int i = 0; i < X_DIM; ++i) {
            std::cout << dxdt[i] << (i == X_DIM - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;
        std::cout << "===== [LOG END] =====" << std::endl;
    }

    delete[] piston1;
    delete[] piston2;
    delete[] V1_dV1dt;
    delete[] V2_dV2dt;

    // Mass flowrate vector [kg/s] (lib6-compatible order)
    // 0: m_po1, 1: m_pi1, 2: m_po2, 3: m_pi2
    // 4: m_sv_cp, 5: m_sv_cn
    // 6: m_sv11p, 7: m_sv12p, 8: m_sv11n, 9: m_sv12n
    mass_flowrate[0] = dm1outdt;
    mass_flowrate[1] = dm1indt;
    mass_flowrate[2] = dm2outdt;
    mass_flowrate[3] = dm2indt;
    mass_flowrate[4] = mdot_pos_valve;
    mass_flowrate[5] = mdot_neg_valve;
    mass_flowrate[6] = m_sv11p;
    mass_flowrate[7] = m_sv12p;
    mass_flowrate[8] = m_sv11n;
    mass_flowrate[9] = m_sv12n;
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
            dmdt = (Pin/sqrt(1000*R*T)) * sqrt(K*pow(2/(K + 1), (K + 1)/(K - 1)));
        } else {
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
    ValveRuntimeState* state = &valve_state_ch_pos;
    const ValveModelParams* params = &CHAMBER_POS_PARAMS;

    switch (static_cast<int>(type)) {
    case 1:
        state = &valve_state_ch_pos;
        params = &CHAMBER_POS_PARAMS;
        break;
    case 2:
        state = &valve_state_ch_neg;
        params = &CHAMBER_NEG_PARAMS;
        break;
    case 3:
        state = &valve_state_act_pos_in;
        params = &ACT_POS_IN_PARAMS;
        break;
    case 4:
        state = &valve_state_act_pos_out;
        params = &ACT_POS_OUT_PARAMS;
        break;
    case 5:
        state = &valve_state_act_neg_in;
        params = &ACT_NEG_IN_PARAMS;
        break;
    case 6:
        state = &valve_state_act_neg_out;
        params = &ACT_NEG_OUT_PARAMS;
        break;
    default:
        return 0.0;
    }

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
        params->A_bw * dI
        - params->beta_bw * std::abs(dI) * state->z
        - params->gamma_bw * dI * std::abs(state->z)
    );
    state->z += dz;
    state->z = std::min(Z_LIMIT, std::max(-Z_LIMIT, state->z));

    double force_net = current + params->C_z * state->z + params->C_p * P_inlet - params->C_k;
    force_net = std::min(FORCE_LIMIT, std::max(-FORCE_LIMIT, force_net));

    const double exp_arg = -params->k_shape * force_net;
    const double log_denom = params->alpha_shape * logaddexp_c(0.0, exp_arg);

    double area_eff = 0.0;
    if (log_denom <= LOG_GUARD) {
        area_eff = params->A_max * std::exp(-log_denom);
    }

    const double phi = compressible_phi(P_inlet, P_outlet);
    const double q_static_lpm = area_eff * P_inlet * phi;

    const double wn = (state_curr >= 0.5) ? params->wn_up : params->wn_down;
    const double zeta = (state_curr >= 0.5) ? params->zeta_up : params->zeta_down;
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

void PneumaticCT::set_logging(bool enable) {
    this->enable_logging_ = enable;
}
