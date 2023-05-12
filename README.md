# Machine learning analysis of humoral and cellular responses to SARS-CoV-2 infection in young adults

**Abstract**: The severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) induces B and T cell responses, contributing to virus neutralization. In a cohort of 2,911 young adults, we identified 65 individuals who had an asymptomatic or mildly symptomatic SARS-CoV-2 infection and characterized their humoral and T cell
responses to the Spike (S), Nucleocapsid (N) and Membrane (M) proteins. We found that previous infection induced CD4 T cells that vigorously responded to pools of peptides derived from the S and N proteins. By using statistical and machine learning models, we observed that the T cell response highly correlated with a compound titer of antibodies against the Receptor Binding Domain (RBD), S and N. However, while serum antibodies decayed over time, the cellular phenotype of these individuals remained stable over four months. Our computational analysis demonstrates that in young adults, asymptomatic and paucisymptomatic SARS-CoV-2 infections can induce robust and long-lasting CD4 T cell responses that exhibit slower decays than antibody titers. These observations imply that next-generation COVID-19 vaccines should be designed to induce stronger cellular responses to sustain the generation of potent neutralizing antibodies.

This repository holds the code and data for replicating the statistical and machine learning analysis of the associations
between T cell and antibody response in SARS-CoV-2.

### Requirements

All required libraries are listed in the conda environment specified by [`environment.yml`](/environment.yml). 
To install it, execute the commands below:
```
conda env create -f environment.yml   # install dependencies
conda activate t-cells-cov            # activate environment
```

### Usage

The analysis can be reproduced by running the following [Jupyter](https://jupyter.org/) notebooks:
- [`thresholds.ipynb`](/thresholds.ipynb): computes optimal antibody level cutoffs based on prepandemic negative and PCR-tested positive controls
- [`diagnostics.ipynb`](/diagnostics.ipynb): histograms; correlation matrices; cumulative distributions; coefficients of variation
- [`final_analysis.ipynb`](/final_analysis.ipynb): principal component analysis; predictive models for antibody response; variable importances

Additionally, the [R](https://www.r-project.org/) script [`correlation_matrices.R`](/correlation_matrices.R) can be used to visualise pairwise correlation matrices.

For the further details on utility functions, consult the documentation.

### Data

All the data used in the analysis are publicly available. The data are available as `.xlsx` and  `.csv` files in the [`/data`](/data/) folder:
- `C+_data`: data from the PCR-tested positive controls (C+ cohort)
- `C-_data`: data from the pre-pandemic negative controls (C- cohort)
- `CoV-ETH_data`: data from the current study (CoV-ETH cohort)
- `IAC_data`: data for intra-assay repeatability assessment

### Maintainer

This repository is maintained by [Ričards Marcinkevičs](https://rmarcinkevics.github.io/) ([ricards.marcinkevics@inf.ethz.ch](mailto:ricards.marcinkevics@inf.ethz.ch)).

### Citation

If you use the code or the dataset from this repository, please cite the paper below:

Marcinkevics R, Silva PN, Hankele A-K, Dörnte C, Kadelka S, Csik K, Godbersen S, Goga A, Hasenöhrl L, Hirschi P, Kabakci H, LaPierre MP, Mayrhofer J, Title AC,
Shu X, Baiioud N, Bernal S, Dassisti L, Saenz-de-Juano MD, Schmidhauser M, Silvestrelli G, Ulbrich SZ, Ulbrich TJ, Wyss T, Stekhoven DJ, Al-Quaddoomi FS, Yu S, Binder M, Schultheiss C, Zindel C, Kolling C, Goldhahn J, Seighalani BK, Zjablovskaja P, Hardung F, Schuster M, Richter A, Huang Y-J, Lauer G, Baurmann H, Low JS, Vaqueirinho D, Jovic S, Piccoli L, Ciesek S, Vogt J, Sallusto F, Stoffel M and Ulbrich SE (2023) Machine learning analysis of humoral and cellular responses to SARS-CoV-2 infection in young adults. *Front. Immunol.* 14:1158905. doi: 10.3389/fimmu.2023.1158905

```
@article{MarcinkevicsSilvaHankele2023,
  title=      {Machine learning analysis of humoral and cellular responses to SARS-CoV-2 infection in young adults},
  author=     {Ricards Marcinkevics and Pamuditha N. Silva and Anna-Katharina Hankele and Charlyn Dörnte and Sarah Kadelka 
                    and Katharina Csik and Svenja Godbersen and Algera Goga and Lynn Hasenöhr and Pascale Hirschi 
                    and Hasan Kabakci and Mary P. LaPierre and Johanna Mayrhofer and Alexandra C. Title and Xuan Shu 
                    and Nouell Baiioud and Sandra Bernal and Laura Dassisti and Mara D. Saenz-de-Juano 
                    and Meret Schmidhauser and Giulia Silvestrelli and Simon Z. Ulbrich and Thea J. Ulbrich 
                    and Tamara Wyss and Daniel J. Stekhoven and Faisal S. Al-Quaddoomi and Shuqing Yu and Mascha Binder 
                    and Christoph Schultheiss and Claudia Zindel and Christoph Kolling and Jörg Goldhahn 
                    and Bahram Kasmapour Seighalani and Polina Zjablovskaja and Frank Hardung and Marc Schuster 
                    and Anne Richter and Yi-Ju Huang and Gereon Lauer and Herrad Baurmann and Jun Siong Low 
                    and Daniela Vaqueirinho and Sandra Jovic and Luca Piccoli and Sandra Ciesek 
                    and Julia Vogt and Federica Sallusto and Markus Stoffel and Susanne E. Ulbrich},
  journal=    {Frontiers in Immunology},
  volume=     {14},
  year=       {2023},
  publisher=  {Frontiers}
}
```
