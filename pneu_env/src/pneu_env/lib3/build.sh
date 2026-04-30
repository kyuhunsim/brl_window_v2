#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

rm -f *.o libpneumatic_simulator.so libpneumatic_simulator_pred.so
echo "=========================================="
echo "Compiling C++ Full System Simulator..."
echo "=========================================="

g++ -shared -o libpneumatic_simulator.so -fPIC pneumatic_simulator.cpp pneumatic_CT.cpp
g++ -shared -o libpneumatic_simulator_pred.so -fPIC pneumatic_simulator.cpp pneumatic_CT.cpp

if [ -f "libpneumatic_simulator.so" ] && [ -f "libpneumatic_simulator_pred.so" ]; then
    echo ""
    echo "SUCCESS: 'libpneumatic_simulator.so' has been created successfully."
    echo "SUCCESS: 'libpneumatic_simulator_pred.so' has been created successfully."
else
    echo ""
    echo "FAILED: Compilation failed."
    exit 1
fi
