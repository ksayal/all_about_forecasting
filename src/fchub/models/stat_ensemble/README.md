# Stat. Ensemble

Equal-weight average of **AutoETS + AutoARIMA + AutoTheta** (point forecast and every quantile).

- **Implementation:** [Nixtla statsforecast](https://github.com/Nixtla/statsforecast), Apache-2.0
- **Why it's slow:** it fits all three members per series, including AutoARIMA.
- Quantiles from different members can cross after averaging, so the base class re-sorts them.
