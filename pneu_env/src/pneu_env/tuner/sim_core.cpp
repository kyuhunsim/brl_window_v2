#include <cmath>
#include <algorithm>

extern "C" {

// Overflow 방지용 logaddexp 함수
double logaddexp_c(double x, double y) {
    if (x == y) return x + 0.6931471805599453; // log(2)
    double diff = x - y;
    if (diff > 0) return x + log1p(exp(-diff));
    else return y + log1p(exp(diff));
}

// 초고속 물리 모델 시뮬레이션 코어
double simulate_core(
    int N, 
    const double* I_arr, const double* State_arr, 
    const double* P_in_abs_arr, const double* Phi_arr, const double* Q_actual, 
    double dt,
    double A_max, double k_shape, double C_k, double C_p, double C_z,
    double A_bw, double beta_bw, double gamma_bw, double alpha,
    double wn_up, double zeta_up, double wn_down, double zeta_down,
    double* Q_pred_out
) {
    // 절대값 처리 (Python의 abs() 대응)
    A_max = std::abs(A_max);   k_shape = std::abs(k_shape);
    C_k = std::abs(C_k);       A_bw = std::abs(A_bw);
    beta_bw = std::abs(beta_bw); gamma_bw = std::abs(gamma_bw);
    alpha = std::abs(alpha);
    wn_up = std::min(std::abs(wn_up), 150.0);
    zeta_up = std::abs(zeta_up);
    wn_down = std::min(std::abs(wn_down), 150.0);
    zeta_down = std::abs(zeta_down);

    double z = 0.0;
    double x1 = Q_actual[0];
    double x2 = 0.0;
    
    int num_sub_steps = 20;
    double dt_sub = dt / num_sub_steps;
    
    double total_error = 0.0;
    double q_mean = 0.0;
    for(int i = 0; i < N; ++i) q_mean += Q_actual[i];
    q_mean /= N;

    for (int k = 0; k < N; ++k) {
        if (k > 0) {
            double abs_dI = std::abs(I_arr[k] - I_arr[k-1]);
            double dI = abs_dI * (2.0 * State_arr[k] - 1.0);
            double dz = A_bw * dI - beta_bw * std::abs(dI) * z - gamma_bw * dI * std::abs(z);
            z += dz;
            if (z > 1e6) z = 1e6;
            if (z < -1e6) z = -1e6;
        }
        
        double Force_net = I_arr[k] + C_z * z + C_p * P_in_abs_arr[k] - C_k;
        if (Force_net < -500.0) Force_net = -500.0;
        if (Force_net > 500.0) Force_net = 500.0;
        
        double exp_arg = -k_shape * Force_net;
        double log_denom = alpha * logaddexp_c(0.0, exp_arg);
        
        double Area_eff = 0.0;
        if (log_denom <= 700.0) {
            Area_eff = A_max * std::exp(-log_denom);
        }
        
        double Q_static = Area_eff * P_in_abs_arr[k] * Phi_arr[k];
        double wn = (State_arr[k] == 1.0) ? wn_up : wn_down;
        double zeta = (State_arr[k] == 1.0) ? zeta_up : zeta_down;
        
        for (int sub = 0; sub < num_sub_steps; ++sub) {
            double dx1 = x2;
            double dx2 = wn * wn * (Q_static - x1) - 2.0 * zeta * wn * x2;
            x1 += dt_sub * dx1;
            x2 += dt_sub * dx2;
        }
        
        double q_pred = std::max(x1, 0.0);
        Q_pred_out[k] = q_pred;
        double diff = Q_actual[k] - q_pred;
        total_error += diff * diff;
    }
    
    // 비정상 튐 방지 페널티
    if (N > 1 && Q_pred_out[1] > q_mean * 2.0) {
        total_error += 1e7;
    }
    
    return total_error;
}

} // extern "C"
