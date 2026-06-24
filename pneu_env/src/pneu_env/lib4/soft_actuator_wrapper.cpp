#include "soft_actuator.h"
#include "pneumatic_simulator.h"
#include <iostream>


// 파이썬에서 공유할 액추에이터 객체를 하나 생성합니다.
SoftActuator actuator;

// C 형식의 함수들을 정의합니다. 파이썬은 이 함수들을 통해 C++ 객체에 접근합니다.
extern "C" {
    /**
     * @brief 액추에이터의 파라미터를 초기화합니다.
     */
    void initialize_parameters(double initial_length, double dimension_D, double num_folds, double shaft_r, double rod_mass) {
        actuator.initializeParameters(initial_length, dimension_D, num_folds, shaft_r, rod_mass);
        PneumaticSimulator::get_instance().set_actuator_parameters(
            initial_length,
            dimension_D,
            num_folds,
            shaft_r,
            rod_mass
        );
        std::cout << "[C++] Actuator parameters initialized." << std::endl;
    }

    /**
     * @brief Positive 압력에 대한 힘(Force)을 계산합니다.
     * @param current_L 현재 액추에이터 길이
     * @param P_pos Positive 압력 (kPa)
     * @return 계산된 힘
     */
    double get_force_pos(double current_L, double P_pos) {
        return actuator.getForce_pos(current_L, P_pos);
    }

    /**
     * @brief Negative 압력에 대한 힘(Force)을 계산합니다.
     * @param current_L 현재 액추에이터 길이
     * @param P_neg Negative 압력 (kPa)
     * @return 계산된 힘
     */
    double get_force_neg(double current_L, double P_neg) {
        return actuator.getForce_neg(current_L, P_neg);
    }

    /**
     * @brief 수축률(Contraction Ratio)을 계산합니다.
     * @param current_L 현재 액추에이터 길이
     * @return 수축률
     */
    double get_contraction_ratio(double current_L) {
        return actuator.getContractionRatio(current_L);
    }

    /**
     * @brief 초기 길이를 반환합니다.
     * @return 초기 길이
     */
    double get_initial_length() {
        return PneumaticSimulator::get_instance().get_initial_length();
    }

    double get_min_length_c() {
        return PneumaticSimulator::get_instance().get_min_length();
    }

    void set_actuator_min_length_c(double min_length) {
        actuator.setMinimumLength(min_length);
        PneumaticSimulator::get_instance().set_actuator_min_length(min_length);
    }

    double get_volume_pos(double current_L) {
        return actuator.getVolume_pos(current_L);
    }
    double get_volume_neg(double current_L) {
        return actuator.getVolume_neg(current_L);
    }
}
