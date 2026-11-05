# rscm-fair

Components that reimplement the FaIR v2 simple climate model (Leach et al. 2021, Smith et al. 2024) for RSCM.

Each component ports one step of FaIR's main loop:
the gas cycle, CH4 lifetime, greenhouse gas, aerosol, ozone and other forcing,
forcing aggregation and the n-layer energy balance model.
They are tested against FaIR v2.2.5 output.
