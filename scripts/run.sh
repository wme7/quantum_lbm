#!/bin/bash

lbm run classical --nx 201 --re 20 --u-inf 0.1 --t-inf 0.5 --out results/classical --vorticity --steps 1000
lbm run quantum --statistic BE --h 1.0 --nx 201 --re 20 --u-inf 0.1 --t-inf 0.5 --out results/quantum_bose --vorticity --steps 1000
lbm run quantum --statistic FD --h 1.0 --nx 201 --re 20 --u-inf 0.1 --t-inf 0.5 --out results/quantum_fermi --vorticity --steps 1000