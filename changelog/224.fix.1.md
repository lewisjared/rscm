`rscm.fair` validation now rejects more inputs that FaIR cannot run, naming each one:

- Time axes with fewer than two bounds, non-finite or decreasing bounds, or uneven steps.
  An uneven axis used to give wrong cumulative emissions.
  `from_fair` checks FaIR's timebounds before building a time axis.
- Repeated species names, and a second species of a type FaIR allows once, such as `co2`, `cfc-11` or `ari`.
- A negative `unperturbed_lifetime`, a CO2 `baseline_concentration` or `forcing_reference_concentration`
  that is not positive, and a negative one for CH4 or N2O.
- An `aci_shape` or prescribed aerosol emissions that leave the aerosol-cloud logarithm undefined,
  naming the first year.
- Strings where numbers belong, and arrays where a parameter takes one number.
- Input variables no species takes, suggesting the closest name.
- An empty ensemble, and ensemble configurations with other species, species order or layers,
  naming the differences.

`from_fair` with an unknown `config` or `scenario` now lists the available ones.
