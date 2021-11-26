# Site-specific antibody and T-cell immune response to particular components of SARS-CoV-2

This repository holds the code for the analysis of the association between the polyclonal antibody response against particular components of SARS-CoV-2 including the prefusion S1 and S2 spike, the RBD of the spike and the N protein on the one hand, and the T-cell response against particular parts of the virus including S1, full S, N protein and M on the other. In addition, we assess which T-cell types and viral components feature the strongest association.

### Requirements

All required libraries are listed in the conda environment specified by `environment.yml`. To install it, execute the commands below:
```
conda env create -f environment.yml   # install dependencies
conda activate ...                    # activate environment
```

### Usage

The analysis can be reproduced by running the following [Jupyter](https://jupyter.org/) notebooks:
- `thresholds.ipynb`: computing optimal antibody level cutoffs
- `diagnostics.ipynb`: histograms, correlation matrices, cumulative distributions, coefficients of variation
- `final_analysis.ipynb`: principal component analysis, predictive models for antibody response, variable importance

### Maintainer

This repository is maintained by Ričards Marcinkevičs ([ricards.marcinkevics@inf.ethz.ch](mailto:ricards.marcinkevics@inf.ethz.ch)).
