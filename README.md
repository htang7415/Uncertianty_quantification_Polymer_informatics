# Uncertainty Quantification in Machine Learning for Polymer Properties: A Benchmark Study
---
**Code repository for the above titled paper**
## Workflow
![alt text](workflow.png)
## Dataset
![alt text](datasets.png)
Tg: Glass transition temperature; 
Eg: Band gap; Tm: Melting temperature; 
Td: Decomposition temperature.
Tg^EXP: represents out-of-distribution experimental Tg data from journal papers.
Tg^MD: represents out-of-distribution Tg data derived from MD simulations in the PoLyInfo.
Polymer_type: four polymer types data from Tg dataset.
## Methods
![alt text](ML_methods.png)
* Neural network ensemble: Pytorch
* Gaussian process regression (GPR): [GPy](https://github.com/SheffieldML/GPy/tree/deploy)
* Monte Carlo dropout (MCD): Pytorch
* Mean-variance estimation (MVE): Pytorch
* Bayesian neural network (BNN): Pytorch
* Evidential deep learning (EDL): Pytorch, [Chemprop](https://github.com/aamini/chemprop)
* Quantile regression (QR): Pytorch
* Natural gradient boosting (NGBoosting): [NGBoost](https://github.com/stanfordmlgroup/ngboost)

### Input
* Morgan fingerprint with frequency (MFF): Considering the number of substructures

### Output
* Mean and standard deviations for polymer properties

## Metrics
* R2, MAE, RMSE
* Spearman's rank correlation coefficient
* Calibration

More details can be found in the paper.