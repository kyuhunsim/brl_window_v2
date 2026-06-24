#include "pneumatic_simulator.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

const double EPSILON_CT = 1e-9;
const double g = 9.81; // [m/s^2]
const double MIN_ABS_PRESSURE_PA = 100.0; // Prevent non-physical negative/zero absolute pressures.

namespace {
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
    0.182088009151, 38.5935820143, 0.222495326173, -0.00024259434466, 1.17749602593e-06,
    133401.60873, 0.317323941447, 0.108264934791, 1.72343412436,
    11, 0.6, 11, 0.6
};
const ValveModelParams CHAMBER_NEG_PARAMS = {
    0.230596704456, 49.2349823439, 0.077360350444, 0.000126358883866, 0.000601866716225,
    9.58924983462e-06, 811.479927158, 0.129990958936, 8684.05676053,
    25, 0.6, 22.8693861168, 0.6
};
const ValveModelParams ACT_POS_IN_PARAMS = {
    0.119192679043, 49.2936789273, 0.0558727179485, -4.09778610123e-05, 3.88446743246e-06,
    22741.1187614, 1.77940442906, 0.42122526239, 51214.1648005,
    11.0167425688, 0.6, 11, 0.6
};
const ValveModelParams ACT_POS_OUT_PARAMS = {
    0.266034385508, 23.2030505728, 0.0295736756483, -7.40609568515e-05, 6.01411586558e-07,
    326660.429485, 2.45774688444, 0.0124443885432, 67.6271687104,
    11.3601006694, 0.6, 25, 0.6
};
const ValveModelParams ACT_NEG_IN_PARAMS = {
    0.186920013518, 44.5908169431, 0.0823299570369, 1.62453098374e-05, 1.13064900337e-06,
    123908.654969, 1.51294915451, 1.3484821451, 6346.0154244,
    11, 0.6, 11, 0.6
};
const ValveModelParams ACT_NEG_OUT_PARAMS = {
    0.227395697276, 28.4795871901, 0.0137302587747, -0.000493432450588, 0.000219131995948,
    633.338396595, 1.70729273937, 2.43765129498, 261.227981819,
    11, 0.6, 25, 0.6
};

double clamp01(double value)
{
    return std::min(1.0, std::max(0.0, value));
}

double logaddexp_c(double x, double y)
{
    if (x == y) return x + 0.6931471805599453;
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
}

PneumaticCT::PneumaticCT() {
    std::cout << "[ INFO] Pneumatic CT ==> Initialized" << std::endl;
    // Load constants from header (already in SI)
    M_W_RPM = M_W_RPM_; 
    SC_R = SC_R_; 
    SC_L = SC_L_; 
    V_D = V_D_;   
    O_S = O_S_; 
    SV_D = SV_D_; 
    SV_G = SV_G_;
    T = T_;
    
    // Perform calculations in SI units
    V_S = 0.25 * PI * V_D * V_D; // [m^2] Piston area
    V_DV = 0.001 * V_S;            // [m^3] Dead volume (assuming 10cm stroke)
    V_MAX_V = 2 * SC_R * V_S + V_DV; // [m^3] Max piston chamber volume

    // Convert chamber volumes from Liters to m^3
    this->V1 = V1_ / 1000.0; // [m^3]
    this->V2 = V2_ / 1000.0; // [m^3]

    // Coeffs are dimensionless
    C1IN = 1e-6 * CIN_; C1OUT = 1e-6 * COUT_; 
    C2IN = 1e-6 * CIN_; C2OUT = 1e-6 * COUT_;
    leak_pos_atm = 0.0;
    leak_neg_atm = 4.003162042168857e-09;
    leak_cross = 8.424896141149943e-10;

    actuator1.initializeParameters(0.02, 0.02, 2.0, 0.005, 5.0);

    dxdt = new double[X_DIM];
    mass_flowrate = new double[M_DOT_DIM];
    for (int i = 0; i < VALVE_DEBUG_DIM; i++) valve_debug[i] = 0.0;
    reset_valve_states();
}

PneumaticCT::~PneumaticCT() { delete[] dxdt; delete[] mass_flowrate; }

void PneumaticCT::set_actuator_parameters(
    double initial_length,
    double dimension_D,
    double num_folds,
    double shaft_r,
    double rod_mass
) {
    actuator1.initializeParameters(initial_length, dimension_D, num_folds, shaft_r, rod_mass);
}

void PneumaticCT::set_actuator_min_length(double min_length) {
    actuator1.setMinimumLength(min_length);
}

void PneumaticCT::set_volume(double volume1_m3, double volume2_m3) { 
    this->V1 = volume1_m3 / 1000.0;
    this->V2 = volume2_m3/ 1000.0;
}

void PneumaticCT::set_discharge_coeff(double c1i, double c1o, double c2i, double c2o) {
    // Updated on 2026-01-10: keep direct Cd convention (same as sim4/lib4 wrapper).
    this->C1IN = c1i; this->C1OUT = c1o;
    this->C2IN = c2i; this->C2OUT = c2o;
}

void PneumaticCT::set_leak_coefficients(double pos_atm, double neg_atm, double cross) {
    this->leak_pos_atm = std::max(0.0, pos_atm);
    this->leak_neg_atm = std::max(0.0, neg_atm);
    this->leak_cross = std::max(0.0, cross);
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
    reset_valve_state(&valve_state_ch_pos);
    reset_valve_state(&valve_state_ch_neg);
    reset_valve_state(&valve_state_act_pos_in);
    reset_valve_state(&valve_state_act_pos_out);
    reset_valve_state(&valve_state_act_neg_in);
    reset_valve_state(&valve_state_act_neg_out);
}

void PneumaticCT::model(const double* x, const double* u, double* dxdt, int step_num) {
    // 1. 상태 및 제어 벡터 읽기 (단일 액추에이터 모델)
    const double L_max_model = actuator1.getInitialLength();
    const double L_min_model = actuator1.getMinimumLength();

    double Pch_pos_Pa = std::max(x[1], MIN_ABS_PRESSURE_PA);
    double Pch_neg_Pa = std::max(x[2], MIN_ABS_PRESSURE_PA);
    double angle_rad  = x[3];
    double Ppis1_Pa   = std::max(x[4], MIN_ABS_PRESSURE_PA);
    double Ppis2_Pa   = std::max(x[5], MIN_ABS_PRESSURE_PA);
    double P1_pos_Pa  = std::max(x[6], MIN_ABS_PRESSURE_PA);
    double P1_neg_Pa  = std::max(x[7], MIN_ABS_PRESSURE_PA);
    double L1_m       = std::min(std::max(x[8], L_min_model), L_max_model);
    double v1_ms      = std::isfinite(x[9]) ? x[9] : 0.0;

    double u_ch_pos = u[0], u_ch_neg = u[1], u11_pos = u[2], u12_pos = u[3];
    double u11_neg = u[4], u12_neg = u[5];

    // 2. 펌프 동역학 및 질량 유량 계산
    double angular_velocity_rads = M_W_RPM * (2 * PI / 60.0);
    double p1[2], p2[2], Vd1[2], Vd2[2];
    slider_crank(angle_rad, angular_velocity_rads, 0, p1);
    slider_crank(angle_rad, angular_velocity_rads, PI, p2);
    volume(p1[0], p1[1], Vd1);
    volume(p2[0], p2[1], Vd2);

    // 질량 유량 [kg/s]
    double m_po1 = orifice(Ppis1_Pa, Pch_pos_Pa, C1OUT);
    double m_pi1 = orifice(Pch_neg_Pa, Ppis1_Pa, C1IN);
    double m_po2 = orifice(Ppis2_Pa, Pch_pos_Pa, C2OUT);
    double m_pi2 = orifice(Pch_neg_Pa, Ppis2_Pa, C2IN);
    double m_sv_cp = solenoid_valve(Pch_pos_Pa, ATM, u_ch_pos, 1, POS_VALVE_NUM);
    double m_sv_cn = solenoid_valve(ATM, Pch_neg_Pa, u_ch_neg, 2, NEG_VALVE_NUM);
    double m_sv11p = solenoid_valve(Pch_pos_Pa, P1_pos_Pa, u11_pos, 3, 1);
    double m_sv12p = solenoid_valve(P1_pos_Pa, ATM, u12_pos, 4, 1);
    double m_sv11n = solenoid_valve(ATM, P1_neg_Pa, u11_neg, 5, 1);
    double m_sv12n = solenoid_valve(P1_neg_Pa, Pch_neg_Pa, u12_neg, 6, 1);

    // Expose mass flowrates for Python-side logging/tuning.
    // Order matches the "Pump / Chamber Valves / Act1 Valves" logging order (10 values).
    //   [0] m_po1, [1] m_pi1, [2] m_po2, [3] m_pi2,
    //   [4] m_sv_cp, [5] m_sv_cn,
    //   [6] m_sv11p, [7] m_sv12p, [8] m_sv11n, [9] m_sv12n
    mass_flowrate[0] = m_po1;
    mass_flowrate[1] = m_pi1;
    mass_flowrate[2] = m_po2;
    mass_flowrate[3] = m_pi2;
    mass_flowrate[4] = m_sv_cp;
    mass_flowrate[5] = m_sv_cn;
    mass_flowrate[6] = m_sv11p;
    mass_flowrate[7] = m_sv12p;
    mass_flowrate[8] = m_sv11n;
    mass_flowrate[9] = m_sv12n;

    // 3. 액추에이터 동역학 계산
    double F1p = actuator1.getForce_pos(L1_m, P1_pos_Pa);
    double F1n = actuator1.getForce_neg(L1_m, P1_neg_Pa);

    // test about damping and spring
    double F_support = 0.0;

    const double L_max = L_max_model;
    const double L_min = L_min_model;
    
    if ((L1_m - L_max) > EPSILON_CT ) {
        // 액추에이터가 최대 길이를 '초과'했을 때
        double penetration = L1_m - L_max;
        // 힘의 방향이 아래(수축) 방향이므로 전체적으로 음수(-)가 되어야 함
        double F_spring = -CONTACT_STIFFNESS * penetration; // 아래(-) 방향 반발력
        double F_damper = CONTACT_DAMPING * (-v1_ms);       // 아래(-) 방향 저항력 (위로 움직이는 속도에 저항)
        F_support = F_spring + F_damper;
    }
    else if ((L_min - L1_m) > EPSILON_CT ) {
        // 액추에이터가 최소 길이보다 '미만'일 때
        double penetration = L_min - L1_m;
        // 힘의 방향이 위(팽창) 방향이므로 전체적으로 양수(+)가 되어야 함
        double F_spring = CONTACT_STIFFNESS * penetration; // 위(+) 방향 반발력
        double F_damper = CONTACT_DAMPING * (-v1_ms);       // 위(+) 방향 저항력 (아래로 움직이는 속도에 저항)
        F_support = F_spring + F_damper;
    }

    else if (fabs(L1_m - L_max) < EPSILON_CT && fabs(v1_ms) < EPSILON_CT) {
        F_support = actuator1.getRodMass() * g;
    }

    // double a1 = (F1p + F1n + F_support) / (actuator1.getRodMass() + EPSILON_CT) - g;
    double a1 = (-F1p - F1n - F_support) / (actuator1.getRodMass() + EPSILON_CT) + g;

    // 로깅을 위해 부피 및 부피 변화율 미리 계산
    double V1_act_pos_m3 = actuator1.getVolume_pos(L1_m);
    double V1_act_neg_m3 = actuator1.getVolume_neg(L1_m);
    double V1dot_derivative_pos = actuator1.getVolumeDerivative_pos(L1_m);
    double V1dot_derivative_neg = actuator1.getVolumeDerivative_neg(L1_m);

    double V1dot_act_pos_m3s = V1dot_derivative_pos * v1_ms;
    double V1dot_act_neg_m3s = V1dot_derivative_neg * v1_ms;
    
    // 4. 압력 변화율 계산
    // 로깅을 위해 순 질량 유량 미리 계산
    double m_dot_net_pos_ch = m_po1 + m_po2 - m_sv_cp - m_sv11p;
    double m_dot_net_neg_ch = m_sv_cn + m_sv12n - (m_pi1 + m_pi2);
    double m_dot_net_pis1 = m_pi1 - m_po1;
    double m_dot_net_pis2 = m_pi2 - m_po2;
    const double m_leak_pos_atm = leak_pos_atm * (P1_pos_Pa - ATM);
    const double m_leak_neg_atm = leak_neg_atm * (P1_neg_Pa - ATM);
    const double m_leak_cross = leak_cross * (P1_pos_Pa - P1_neg_Pa);
    double m_dot_net_act1_pos = m_sv11p - m_sv12p - m_leak_pos_atm - m_leak_cross;
    double m_dot_net_act1_neg = m_sv11n - m_sv12n - m_leak_neg_atm + m_leak_cross;

    // 압력 변화율 [Pa/s]
    double dPch_p = chamber(m_dot_net_pos_ch, V1);
    double dPch_n = chamber(m_dot_net_neg_ch, V2);
    double dPpis1 = pressure(Ppis1_Pa, Vd1[0], Vd1[1], m_dot_net_pis1);
    double dPpis2 = pressure(Ppis2_Pa, Vd2[0], Vd2[1], m_dot_net_pis2);
    double dP1p = pressure(P1_pos_Pa, V1_act_pos_m3, V1dot_act_pos_m3s, m_dot_net_act1_pos);
    double dP1n = pressure(P1_neg_Pa, V1_act_neg_m3, V1dot_act_neg_m3s, m_dot_net_act1_neg);

    // 5. dxdt 벡터 종합
    dxdt[0] = 1.0;
    dxdt[1] = dPch_p;   dxdt[2] = dPch_n;
    dxdt[3] = angular_velocity_rads;
    dxdt[4] = dPpis1;   dxdt[5] = dPpis2;
    dxdt[6] = dP1p;     dxdt[7] = dP1n;
    dxdt[8] = v1_ms;    dxdt[9] = a1;

    // =================================================================
    // 6. 상세 디버깅 로그 출력부 (요청하신 형식으로 수정)
    // =================================================================
    if (this->enable_logging_){
        std::string step_label = (step_num == 1) ? "k1" : (step_num == 2) ? "k2" : (step_num == 3) ? "k3" : "k4";
        std::cout << "\n===== [LOG START] Time: " << x[0] << "s, RK Step: " << step_label << " =====" << std::endl;

        // --- 1. 입력 상태 및 제어 신호 출력 ---
        std::cout << "INPUTS:" << std::endl;
        std::cout << "  - Pressures [kPa]: Pch_p=" << Pch_pos_Pa/1000.0 << ", Pch_n=" << Pch_neg_Pa/1000.0 << std::endl;
        std::cout << "  - Act1 State: L1=" << L1_m << ", v1=" << v1_ms << ", P1_p=" << P1_pos_Pa/1000.0 << ", P1_n=" << P1_neg_Pa/1000.0 << " [kPa]" << std::endl;
        std::cout << "  - Controls: u_ch_p=" << u_ch_pos << ", u_ch_n=" << u_ch_neg << ", u11p=" << u11_pos << ", u12p=" << u12_pos << ", u11n=" << u11_neg << ", u12n=" << u12_neg << std::endl;

        // --- 2. 질량 유량 계산 결과 출력 ---
        std::cout << "MASS FLOW [kg/s]:" << std::endl;
        std::cout << "  m_dot_net_pos_ch = " << m_po1 + m_po2 << " - " << m_sv_cp << " - " << m_sv11p << " = " << m_dot_net_pos_ch << std::endl;
        std::cout << "  m_dot_net_neg_ch = " << m_sv_cn << " + " << m_sv12n << " - (" << m_pi1 << " + " << m_pi2 << ") = " << m_dot_net_neg_ch << std::endl;
        std::cout << "  - Chamber->Atmos: m_sv_cp=" << m_sv_cp << ", m_sv_cn=" << m_sv_cn << std::endl;
        std::cout << "  - Act1 Valves: m_sv11p=" << m_sv11p << ", m_sv12p=" << m_sv12p << ", m_sv11n=" << m_sv11n << ", m_sv12n=" << m_sv12n << std::endl;
        std::cout << "  - Act1 Leaks: pos_atm=" << m_leak_pos_atm << ", neg_atm=" << m_leak_neg_atm << ", cross_pos_to_neg=" << m_leak_cross << std::endl;

        // --- 3. 액추에이터 동역학 계산 결과 출력 ---
        std::cout << "ACTUATOR DYNAMICS:" << std::endl;
        std::cout << "  - Forces [N]: F_pos=" << F1p << ", F_neg=" << F1n << std::endl;
        std::cout << "  - Contact Force [N]: F_support=" << F_support << std::endl;
        std::cout << "  - Net Force [N]: F_net=" << (F1p + F1n + F_support) << std::endl;
        std::cout << "  - Motion [m/s^2]: a1=" << a1 << std::endl;
        std::cout << "  - Volumes [m^3]: V_pos=" << V1_act_pos_m3 << ", V_neg=" << V1_act_neg_m3 << std::endl;
        std::cout << "  - Volume Rates [m^3/s]:" << std::endl;
        std::cout << "      Vdot_pos = " << V1dot_derivative_pos << " * " << v1_ms
                << " = " << V1dot_act_pos_m3s << std::endl;
        std::cout << "      Vdot_neg = " << V1dot_derivative_neg << " * " << v1_ms
                << " = " << V1dot_act_neg_m3s << std::endl;

        // --- 4. 압력 변화율 상세 계산 과정 출력 ---
        std::cout << "PRESSURE DERIVATIVES [Pa/s]:" << std::endl;
        std::cout << "  - dPch_p: " << dPch_p << " (RTmdot=" << (R*T_OUT*m_dot_net_pos_ch) << ", PVdot=0, V_total=" << V1 << ")" << std::endl;
        std::cout << "  - dPch_n: " << dPch_n << " (RTmdot=" << (R*T_OUT*m_dot_net_neg_ch) << ", PVdot=0, V_total=" << V2 << ")" << std::endl;
        std::cout << "  - dP1p:   " << dP1p << " (RTmdot=" << (R * T_OUT * m_dot_net_act1_pos) << ", PVdot=" << P1_pos_Pa << " * " << V1dot_act_pos_m3s << " = " << (P1_pos_Pa * V1dot_act_pos_m3s)<< ", V_total=" << V1_act_pos_m3 << ")" << std::endl;
        std::cout << "  - dP1n:   " << dP1n << " (RTmdot=" << (R * T_OUT * m_dot_net_act1_neg) << ", PVdot=" << P1_neg_Pa << " * " << V1dot_act_neg_m3s<< " = " << (P1_neg_Pa * V1dot_act_neg_m3s) << ", V_total=" << V1_act_neg_m3 << ")" << std::endl;

        // --- 5. 최종 출력 dxdt 벡터 출력 ---
        std::cout << "OUTPUT (dxdt): [";
        for(int i=0; i < X_DIM; ++i) { // X_DIM = 10
            std::cout << dxdt[i] << (i == X_DIM - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;
        std::cout << "===== [LOG END] =====" << std::endl;
    }
}

double* PneumaticCT::get_mass_flowrate() { return mass_flowrate; }

void PneumaticCT::slider_crank(double angle_rad, double angular_velocity_rads, double phase_rad, double* piston_m_ms){
    double pos_m = SC_R * cos(angle_rad + phase_rad) + sqrt(SC_L*SC_L - pow(SC_R*sin(angle_rad + phase_rad), 2)) + SC_R - SC_L;
    double vel_ms = (-SC_R*sin(angle_rad + phase_rad) - pow(SC_R, 2)*sin(angle_rad + phase_rad)*cos(angle_rad + phase_rad)/sqrt(pow(SC_L, 2) - pow(SC_R*sin(angle_rad + phase_rad), 2))) * angular_velocity_rads;
    piston_m_ms[0] = pos_m;   // [m]
    piston_m_ms[1] = vel_ms;  // [m/s]
}

void PneumaticCT::volume(double piston_pos_m, double piston_vel_ms, double* V_dVdt){
    V_dVdt[0] = V_S * (2*SC_R - piston_pos_m) + V_DV; // [m^3]
    V_dVdt[1] = -V_S * piston_vel_ms;                 // [m^3/s]
}

double PneumaticCT::orifice(double P_inlet_Pa, double P_outlet_Pa, double Cd){
    P_inlet_Pa = std::max(P_inlet_Pa, MIN_ABS_PRESSURE_PA);
    P_outlet_Pa = std::max(P_outlet_Pa, MIN_ABS_PRESSURE_PA);
    double dmdt_kgs;
    double Pcr = pow(2/(K + 1), K/(K - 1));

    if (P_inlet_Pa >= P_outlet_Pa) {
        double P_ratio = P_outlet_Pa / P_inlet_Pa;
        if (P_ratio <= Pcr) { // Choked flow
            dmdt_kgs = (P_inlet_Pa/sqrt(R*T)) * sqrt(K*pow(2/(K + 1), (K + 1)/(K - 1)));
        } else { // Subsonic flow
            dmdt_kgs = (P_inlet_Pa/sqrt(R*T)) * sqrt(2*K/(K - 1)) * sqrt(pow(P_ratio, 2/K) - pow(P_ratio, (K + 1)/K));
        }
    } else { dmdt_kgs = 0; }
    return Cd * dmdt_kgs;
}

double PneumaticCT::pressure(double P_Pa, double V_m3, double dVdt_m3s, double dmdt_kgs){
    P_Pa = std::max(P_Pa, MIN_ABS_PRESSURE_PA);
    if (V_m3 < EPSILON_CT) return 0;
    return (-P_Pa * dVdt_m3s + R * T * dmdt_kgs) / V_m3;
}

double PneumaticCT::solenoid_valve(
    double P_inlet_Pa,
    double P_outlet_Pa,
    double signal,
    double type,
    double num
)
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

    const double P_inlet_kPa = P_inlet_Pa * 1e-3;
    const double P_outlet_kPa = P_outlet_Pa * 1e-3;
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

    double force_net = current + params->C_z * state->z + params->C_p * P_inlet_kPa - params->C_k;
    force_net = std::min(FORCE_LIMIT, std::max(-FORCE_LIMIT, force_net));

    const double exp_arg = -params->k_shape * force_net;
    const double log_denom = params->alpha_shape * logaddexp_c(0.0, exp_arg);

    double area_eff = 0.0;
    if (log_denom <= LOG_GUARD) {
        area_eff = params->A_max * std::exp(-log_denom);
    }

    const double phi = compressible_phi(P_inlet_kPa, P_outlet_kPa);
    const double q_static_lpm = area_eff * P_inlet_kPa * phi;

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
    const int debug_base = (static_cast<int>(type) - 1) * 9;
    if (debug_base >= 0 && debug_base + 8 < VALVE_DEBUG_DIM) {
        valve_debug[debug_base + 0] = u_eff;
        valve_debug[debug_base + 1] = current;
        valve_debug[debug_base + 2] = state_curr;
        valve_debug[debug_base + 3] = state->z;
        valve_debug[debug_base + 4] = force_net;
        valve_debug[debug_base + 5] = area_eff;
        valve_debug[debug_base + 6] = q_static_lpm;
        valve_debug[debug_base + 7] = q_pred_lpm;
        valve_debug[debug_base + 8] = mdot;
    }
    if (!std::isfinite(mdot)) return 0.0;

    return std::max(num * mdot, 0.0);
}

double PneumaticCT::chamber(double dmdt, double V)
{
    // double dPdt = R*T_OUT*dmdt*1000/V;
    double dPdt = (R * T_OUT * dmdt) / V;

    return dPdt;
}

double* PneumaticCT::get_valve_debug() { return valve_debug; }
