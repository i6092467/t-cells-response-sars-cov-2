# Machine learning analysis of humoral and cellular responses to SARS-CoV-2 infection in young adult

**Abstract**: The severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) induces B and T cell responses, 
contributing to virus neutralization. In a cohort of 2,911 young adults, we identified 65 individuals who had an 
asymptomatic or mildly symptomatic SARS-CoV-2 infection and characterized their humoral and T cell responses to the 
Spike (S), Nucleocapsid (N) and Membrane (M) proteins. We found that previous infection induced CD4 T cells that 
vigorously responded to pools of peptides derived from the S and N proteins. By using statistical and machine learning 
models, we observed that the T cell response highly correlated with a compound titer of antibodies against the Receptor 
Binding Domain (RBD), S and N. However, while serum antibodies decayed over time, the cellular phenotype of these 
individuals remained stable over four months. Our computational analysis demonstrates that in young adults, asymptomatic 
and paucisymptomatic SARS-CoV-2 infections can induce robust and long-lasting CD4 T cell responses that exhibit slower 
decays than antibody titers. These observations imply that next-generation COVID-19 vaccines should be designed to induce 
stronger cellular responses to sustain the generation of potent neutralizing antibodies.

This repository holds the code and data for replicating the statistical and machine learning analysis of the associations 
between the antibody titers and T cell response to SARS-CoV-2.

### Requirements

All required libraries are listed in the conda environment specified by [`environment.yml`](/environment.yml). 
To install it, execute the commands below:
```
conda env create -f environment.yml   # install dependencies
conda activate t-cells-cov            # activate environment
```

### Usage

The analysis can be reproduced by running the following [Jupyter](https://jupyter.org/) notebooks:
- [`thresholds.ipynb`](/thresholds.ipynb): computing optimal antibody level cutoffs
- [`diagnostics.ipynb`](/diagnostics.ipynb): histograms; correlation matrices; cumulative distributions; coefficients of variation
- [`final_analysis.ipynb`](/final_analysis.ipynb): principal component analysis; predictive models for antibody response; variable importances

Additionally, the [R](https://www.r-project.org/) script [`correltion_matrices.R`](/correlation_matrices.R) can be used to visualise pairwise correlation matrices.

For the further details on utility functions, consult documentation.

### Data

All the data used in the analysis are publicly available. The data are available as `.xlsx` and  `.csv` files in the [`/data`](/data/) folder:
- `C+_data`: data from the PCR-tested positive controls (C+ cohort)
- `C-_data`: data from the pre-pandemic negative controls (C- cohort)
- `CoV-ETH_data`: data from the current study (CoV-ETH cohort)
- `IAC_data`: data for intra-assay repeatability assessment