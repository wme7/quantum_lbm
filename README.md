# Quantum LBM

2D D2Q9 Lattice Boltzmann solvers for flow past a cylinder in a square far-field box.

- **Classical** Shan-He N=2 thermal LBM — verification baseline
- **Quantum** Yang-Hung UUB-BGK (FD / BE / MB) — separate formulation

## Install

```bash
uv sync --group dev
uv pip install -e .
```

## Run

Classical (Shan-He):

```bash
lbm run classical --nx 201 --ny 201 --re 20 --u-inf 0.1 --t-inf 0.5 --diameter 20 --steps 5000 --out results/classical/
```

Quantum (Yang-Hung):

```bash
lbm run quantum --statistic FD --h 1.0 --nx 201 --ny 201 --re 20 --u-inf 0.1 --t-inf 0.5 --diameter 20 --steps 5000 --out results/quantum-fd/
lbm run quantum --statistic BE --h 1.0 --nx 201 --re 20 --u-inf 0.1 --t-inf 0.5 --out results/quantum-be/
lbm run quantum --statistic MB --h 1.0 --nx 201 --re 20 --u-inf 0.1 --t-inf 0.5 --out results/quantum-mb/
```

`--h` is the Planck/degeneracy parameter in `ideal_gases.find_fugacity`; free-stream fugacity is $z_\infty=\mathrm{find\_fugacity}(\rho_\infty,T_\infty,h,\eta)$. Use `MB` to cross-check against classical (same hydrodynamics when $\chi\equiv 1$).

Useful shared flags:

- `--cx` / `--cy` — cylinder center (defaults: `nx/2`, `ny/2`)
- `--rho-inf` — free-stream density
- `--save-every N` — write field PNGs every N steps (requires `--out`)
- `--vorticity` — stack a vorticity panel under the velocity PNG

## Tests

```bash
uv run pytest
```

## Method

### Classical (`lbm.classical`)

- Lattice: D2Q9, reference $c_s^2 = 1/3$
- Equilibrium: Shan-He N=2 Hermite Maxwellian with variable $T$
- EOS: $P = \rho T$
- Collision: BGK with $\tau = \nu/T + 1/2$

### Quantum (`lbm.quantum`)

- Same D2Q9 / N=2 skeleton
- Equilibrium: Yang-Hung with factor $\chi(z)=g_{5/2}(z)/g_{3/2}(z)$
- Collision: BGK with $\tau = \nu/(T\chi) + 1/2$
- Fugacity from `ideal_gases` (3D density law, spatial energy $D=2$)

### Shared

- Domain: square far-field box; cylinder at center
- Outer boundary: free-stream $f^{\mathrm{eq}}$ on all four edges
- Cylinder: fullway bounce-back


### References

- Xiaowen Shan and Xiaoyi He. Discretization of the velocity space in the solution of the Boltzmann equation. *Physical Review Letters* **80**, 65–68 (1998). [doi:10.1103/PhysRevLett.80.65](https://doi.org/10.1103/PhysRevLett.80.65). ArXiv copy: [arXiv:comp-gas/9712001](https://arxiv.org/abs/comp-gas/9712001).
- Jaw-Yen Yang and Li-Hsin Hung. Lattice Uehling-Uhlenbeck Boltzmann-Bhatnagar-Gross-Krook hydrodynamics of quantum gases. *Physical Review E* **79**, 056708 (2009). [doi:10.1103/PhysRevE.79.056708](https://doi.org/10.1103/PhysRevE.79.056708).
- Rodrigo C. V. Coelho, Anderson Ilha, Mauro M. Doria, R. M. Pereira, and Valter Yoshihiko Aibe. Lattice Boltzmann method for bosons and fermions and the fourth-order Hermite polynomial expansion. *Physical Review E* **89**, 043302 (2014). [doi:10.1103/PhysRevE.89.043302](https://doi.org/10.1103/PhysRevE.89.043302). ArXiv copy: [arXiv:1311.6535v1](https://arxiv.org/abs/1311.6535v1).
