#include "pneumatic_simulator.h"
#include <cmath>

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
    // type means that valve is postive or negative
    // type 1: positive
    // type 2: negative

    // double dmdt;

    double _R = 1000*R;
    
    double D = SV_D*0.01;
    double d = SV_G*0.01;
    double S = 0.25*PI*D*D;

    double current = 0.165*signal;

    double Pin = 1000*P_inlet;
    double Pout = 1000*P_outlet;
    double Patm = 1000*ATM;

    double Phi;
    double Pcr = pow(2/(K + 1), K/(K - 1));
    if (Pin >= Pout) 
    {
        if (Pout/Pin <= Pcr) {
            Phi = (Pin/sqrt(_R*T_OUT)) * sqrt(K*pow(2/(K + 1), (K + 1)/(K - 1)));
        } else {
            Phi = (Pin/sqrt(_R*T_OUT)) * sqrt(2*K/(K - 1)) * sqrt(pow(Pout/Pin, 2/K) - pow(Pout/Pin, (K + 1)/K));
        }
    } else {
        Phi = 0;
    }

    double Cdkx;
    if (type == 1) {
        // Three valve
        // double Cpos1 = 0.00112413533696979;
        // double Cpos2 = 0.000169651700761525;
        // double Cpos3 = 2.27440482592866e-7;
        // One valve
        double Cpos1 = 0.000590126854331190;
        double Cpos2 = 8.90726972374690e-05;
        double Cpos3 = 6.05278829960062e-08;
        
        Cdkx = Cpos1*current - Cpos2 + Cpos3*(Pin - Patm)*S > 0 ? Cpos1*current - Cpos2 + Cpos3*(Pin - Patm)*S : 0;
    } else {
        // Three valve
        // double Cneg1 = 0.000675536259277645;
        // double Cneg2 = 9.78352859637934e-5;
        // One valve
        double Cneg1 = 0.000966211405733290;
        double Cneg2 = 0.000145744566455925;

        Cdkx = Cneg1*current - Cneg2 > 0 ? Cneg1*current - Cneg2 : 0;
    }

    return num*Cdkx*D*PI*Phi;
}

double PneumaticCT::chamber(double dmdt, double V)
{
    double dPdt = R*T_OUT*dmdt*1000/V;

    return dPdt;
}

double* PneumaticCT::get_mass_flowrate() { return mass_flowrate; }