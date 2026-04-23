echo "Compile start..."
g++ -shared -o pneumatic_simulator.so \
    -fPIC pneumatic_simulator.cpp \
    -fPIC pneumatic_CT.cpp
echo "Simulator done!"
g++ -shared -o pneumatic_simulator_pred.so \
    -fPIC pneumatic_simulator.cpp \
    -fPIC pneumatic_CT.cpp
echo "Predict simulator done!"
