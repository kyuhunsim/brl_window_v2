#include <iostream>
#include <fstream>

#define V1_ 0.75
#define V2_ 0.4
// #define V1P_ 0.062
#define V1P_ 0.1
// #define V1N_ 0.077
#define V1N_ 0.156
#define TS 0.0001
#define T_ 323.15
#define T_OUT 293.15
#define PI 3.14159265358979
#define R 0.287
#define K 1.4
#define ATM 101.325
#define X_DIM 8
#define OBS 5
#define U_DIM 6
#define M_W_RPM_ 3000
#define SC_R_ 2
#define SC_L_ 7
#define V_D_ 7
#define O_S_ 148*0.1/180
#define SV_D_ 1.6
#define SV_G_ 0.16

#define POS_VALVE_NUM 1
#define NEG_VALVE_NUM 1


// #define CIN_ 3.79683467
// #define COUT_ 7.73269091
# define CIN_ 3.57059304
# define COUT_ 28.58182353

#define M_DOT_DIM 10
// mass_flowrate (lib5) index map:
// 0 m_po1, 1 m_pi1, 2 m_po2, 3 m_pi2,
// 4 m_sv_cp, 5 m_sv_cn,
// 6 m_sv11p, 7 m_sv12p, 8 m_sv11n, 9 m_sv12n

struct PneumaticCT
{
private:
    struct ValveRuntimeState
    {
        double z;
        double x1;
        double x2;
        double I_prev;
        double state_prev;
    };

    double M_W_RPM;
    double SC_R, SC_L;
    double V_D, V_S, V_DV, V_MAX_V;
    double O_S;
    double SV_D, SV_G;
    double T;
    double V1, V2, V1P, V1N;
    double C1OUT, C1IN, C2OUT, C2IN;
    double* dxdt;
    double* mass_flowrate;
    ValveRuntimeState valve_state_ch_pos;
    ValveRuntimeState valve_state_ch_neg;
    ValveRuntimeState valve_state_act_pos_in;
    ValveRuntimeState valve_state_act_pos_out;
    ValveRuntimeState valve_state_act_neg_in;
    ValveRuntimeState valve_state_act_neg_out;

    void reset_valve_state(ValveRuntimeState* state);
    void slider_crank(double angle, double angular_velocity, double phase, double* piston);
    void volume(double piston_pos, double piston_vel, double* V_dVdt);
    double orifice(double P_inlet, double P_outlet, double Cd);
    double pressure(double P, double V, double dVdt, double dmdt);
    double chamber(double dmdt, double V);

    bool enable_logging_ = false;

public:
    PneumaticCT();
    ~PneumaticCT();
    void reset_valve_states();
    void set_volume(double volume1, double volume2);
    void set_discharge_coeff(
        double Cd1IN,
        double Cd1OUT,
        double Cd2IN,
        double Cd2OUT
    );
    void model(const double* x, const double* u, double* dxdt, int step_num);
    double* get_mass_flowrate();
    void set_logging(bool enable);
    
    double solenoid_valve(double P_inlet, double P_outlet, double signal, double type, double num);
};

struct PneumaticSimulator
{
public:
    PneumaticSimulator();
    ~PneumaticSimulator();
    static PneumaticSimulator& get_instance();
    void set_init_env(double pos_press, double neg_press);
    void set_init_env_act(
        double pos_press,
        double neg_press,
        double act_pos_press,
        double act_neg_press
    );
    void pneumaticDT(double* xk, double* uk, double Ts, double* xk1);
    double get_time();
    double* get_mass_flowrate();
    double* step(double* control, double time_step);
    void set_volume(double volume1, double volume2);
    void set_discharge_coeff(
        double Cd1IN,
        double Cd1OUT,
        double Cd2IN,
        double C2OUT
    );
    void time_reset();
    void set_logging(bool enable);

private:
    PneumaticCT *pneumaticCT;
    double* xk0;
    double* k;
    double* observation;
    double* mass_flowrate;

    double* k1;
    double* k2;
    double* k3;
    double* k4;
    double* i2;
    double* i3;
    double* i4;
};
