# IV_MNAR

This repository contains the replication code and outputs for the paper **Identifiability of the instrumental variable model with the treatment and outcome missing not at random**. The code reproduces the numerical study and the empirical illustration based on the National Job Corps Study (NJCS).

## Repository structure

```text
IV_MNAR/
├── sim.py
├── sim_results/
│   ├── figures/
│   │   ├── fig_identification_headline.pdf
│   │   └── fig_misspecification_heatmap.pdf
│   └── tables/
│       ├── tab_identification_diagnostics.tex
│       └── numbers_for_text.tex
└── njcs_application/
    ├── code/
    │   └── njcs.py
    ├── data/
    │   ├── mpr_jobcorps_team5_nrw_upd_r_nositeid.dta
    │   ├── key_vars.dta
    │   └── jobcorps_everjc30.dta
    └── output/
        ├── njcs_cace_forest.png
        └── njcs_manuscript_numbers.tex
```

## Dependencies

The code requires Python 3 and the following Python packages:

```bash
python -m pip install numpy pandas scipy matplotlib
```

## Reproducing the numerical study

From the repository root, run:

```bash
python sim.py --n 5000 --reps 1000 --n-jobs 8
```

This regenerates the simulation figures and LaTeX tables in `sim_results/`:

- `sim_results/figures/fig_identification_headline.pdf`
- `sim_results/figures/fig_misspecification_heatmap.pdf`
- `sim_results/tables/tab_identification_diagnostics.tex`
- `sim_results/tables/numbers_for_text.tex`

## Reproducing the NJCS empirical illustration

From inside the `njcs_application/` folder, run:

```bash
python code/njcs.py --B 500 --n-jobs 8
```

This regenerates the empirical figure and manuscript-number file:

- `njcs_application/output/njcs_cace_forest.png`
- `njcs_application/output/njcs_manuscript_numbers.tex`

The NJCS analysis uses assignment to Job Corps as the instrument, actual enrollment status as the received treatment, and weekly fourth-year earnings as the outcome. The analysis adjusts for gender, age, race, education, baseline earnings, child status, and arrest history. The empirical figure reports CACE estimates and 95% percentile bootstrap confidence intervals under the identified missingness mechanisms compatible with one-sided noncompliance.

## Manuscript outputs

The manuscript uses the following generated files:

- Numerical study: `sim_results/figures/fig_identification_headline.pdf`
- Misspecification analysis: `sim_results/figures/fig_misspecification_heatmap.pdf`
- Identification diagnostics table: `sim_results/tables/tab_identification_diagnostics.tex`
- Simulation text macros: `sim_results/tables/numbers_for_text.tex`
- NJCS empirical figure: `njcs_application/output/njcs_cace_forest.png`
- NJCS text macros: `njcs_application/output/njcs_manuscript_numbers.tex`
