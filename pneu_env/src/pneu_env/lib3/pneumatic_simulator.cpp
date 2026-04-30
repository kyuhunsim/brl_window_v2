#include "pneumatic_simulator.h"
#include <cmath>


static PneumaticSimulator pneumatic_simulator;

PneumaticSimulator::PneumaticSimulator() 
{
    pneumaticCT = new PneumaticCT;
    
    xk0 = new double[X_DIM];
    xk0[0] = 0;
    xk0[1] = ATM;
    xk0[2] = ATM;
    xk0[3] = PI/2;
    xk0[4] = ATM;
    xk0[5] = ATM;
    xk0[6] = ATM;
    xk0[7] = ATM;

    k = nullptr;
    observation = new double[OBS];
    mass_flowrate = new double[M_DOT_DIM];

    k1 = new double[X_DIM];
    k2 = new double[X_DIM];
    k3 = new double[X_DIM];
    k4 = new double[X_DIM];
    i2 = new double[X_DIM];
    i3 = new double[X_DIM];
    i4 = new double[X_DIM];

    std::cout << "[ INFO] Pneumatic Simulator ==> Initialized with Pneumatic-Only 6-Valve Model" << std::endl;
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

PneumaticSimulator& PneumaticSimulator::get_instance() { return pneumatic_simulator; }

void PneumaticSimulator::set_init_env(double pos_press, double neg_press)
{
    set_init_env_act(pos_press, neg_press, ATM, ATM);
}

void PneumaticSimulator::set_init_env_act(
    double pos_press,
    double neg_press,
    double act_pos_press,
    double act_neg_press
)
{
    pneumaticCT -> reset_valve_states();

    xk0[1] = pos_press;
    xk0[2] = neg_press;
    xk0[3] = 0;

    // lib2 one-valve pump pressure initialization.
    xk0[4] = 0.98947*pos_press + 0.27407*neg_press - 0.30175; // Max
    xk0[5] = 0.023686*pos_press + 0.017746*neg_press + 0.47436; // Min

    xk0[6] = act_pos_press;
    xk0[7] = act_neg_press;
}

void PneumaticSimulator::set_volume(double volume1, double volume2)
{
    pneumaticCT -> set_volume(volume1, volume2);
    std::cout << "[ INFO] Pneumatic Simulator ==> Vol initialized: POS " << volume1 << " NEG " << volume2 << std::endl; 
}

void PneumaticSimulator::set_discharge_coeff(
    double Cd1IN,
    double Cd1OUT,
    double Cd2IN,
    double Cd2OUT
)
{
    pneumaticCT -> set_discharge_coeff(
        Cd1IN,
        Cd1OUT,
        Cd2IN,
        Cd2OUT
    );
}

void PneumaticSimulator::pneumaticDT(double* xk, double* uk, double Ts, double* xk1)
{
    pneumaticCT->model(xk, uk, k1, 1);
    for (int i = 0; i < X_DIM; i++) i2[i] = xk[i] + k1[i]*Ts/3;

    pneumaticCT->model(i2, uk, k2, 2);
    for (int i = 0; i < X_DIM; i++) i3[i] = xk[i] - k1[i]*Ts/3 + k2[i]*Ts;

    pneumaticCT->model(i3, uk, k3, 3);
    for (int i = 0; i < X_DIM; i++) i4[i] = xk[i] + k1[i]*Ts - k2[i]*Ts + k3[i]*Ts;

    pneumaticCT->model(i4, uk, k4, 4);

    for (int i = 0; i < X_DIM; i++)
        xk1[i] = xk[i] + Ts*k1[i]/8 + Ts*3*k2[i]/8 + Ts*3*k3[i]/8 + Ts*k4[i]/8;

    xk1[3]= fmod(xk1[3], 2*PI);
}

double* PneumaticSimulator::step(double* control, double time_step)
{
    double* xk1 = new double[X_DIM];
    int n = time_step/TS;

    for (int i = 0; i < n; i++)
    {
        pneumaticDT(xk0, control, TS, xk1);
        for (int j = 0; j < X_DIM; j++) xk0[j] = xk1[j];
    }

    observation[0] = xk0[0];
    observation[1] = xk0[1];
    observation[2] = xk0[2];
    observation[3] = xk0[6];
    observation[4] = xk0[7];

    delete[] xk1;

    return observation;
}

double PneumaticSimulator::get_time() { return observation[0]; }

double* PneumaticSimulator::get_mass_flowrate() 
{   
    double* mf = pneumaticCT->get_mass_flowrate();
    for (int i = 0; i < M_DOT_DIM; i++) {
        mass_flowrate[i] = mf[i];
    }
    return mass_flowrate;
}

void PneumaticSimulator::time_reset() { 
    xk0[0] = 0; 
    pneumaticCT -> reset_valve_states();
}

void PneumaticSimulator::set_logging(bool enable) {
    this->pneumaticCT->set_logging(enable);
}

extern "C" {
    double get_time_c() {return PneumaticSimulator::get_instance().get_time();}
    double* step_c(double* control, double time_step) { return PneumaticSimulator::get_instance().step(control, time_step); }
    void set_init_env_c(double pos_press, double neg_press){ return PneumaticSimulator::get_instance().set_init_env(pos_press, neg_press); }
    void set_init_env_act_c(
        double pos_press,
        double neg_press,
        double act_pos_press,
        double act_neg_press
    )
    {
        return PneumaticSimulator::get_instance().set_init_env_act(
            pos_press, neg_press, act_pos_press, act_neg_press
        );
    }
    void set_volume_c(double volume1, double volume2) { return PneumaticSimulator::get_instance().set_volume(volume1, volume2); }
    void set_discharge_coeff_c(
        double Cd1IN, double Cd1OUT, 
        double Cd2IN, double C2OUT 
    )
    {
        return PneumaticSimulator::get_instance().set_discharge_coeff(
            Cd1IN, Cd1OUT, Cd2IN, C2OUT
        );
    }
    double* get_mass_flowrate_c() { return PneumaticSimulator::get_instance().get_mass_flowrate(); }
    void time_reset_c() {return PneumaticSimulator::get_instance().time_reset();}
    void set_logging_c(bool enable) { return PneumaticSimulator::get_instance().set_logging(enable); }
}
