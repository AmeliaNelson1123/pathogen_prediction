# Project Name: Decision Support Tool for Farmers to Identify Listeria spp. Risk
Project For: IAFP Student Competition

## Authors:
YeonJin Jung ()
Leonie ()
Linda Kalunga ()
Amelia Nelson (aln65@cornell.edu)

# Last Edit Date:
02/18/2026

## Goal:
The intention of this project is to develop a decsion support tool for farmers and growers to identify the risk of Listeria Presense.

## Problem Statement:
- what does this enable
- what is missing
- why is this important
- why is this novel

## Workflow Diagram and Features
### Training
- Listeria data was analyzed using an Exploratory Data Analysis (listeria_eda.ipynb)
- 6 models were tested to evaluate predictive ability to identify Listeria spp. presense (Run_Models_and_Analyze.ipynb OR run_models_basic.py)
- The top 3 models were selected and optimized by hyperparameters and data engineering. Data engineering was done through feature selection, log transformations, and more (listeria_eda.ipynb and Run_Models_and_Analyze.ipynb). 
- Feature Selection was done through literature reviews, expert insight, and permuntation and feature importance looked at through model results and PCA transformations (listeria_eda.ipynb and Run_Models_and_Analyze.ipynb).
### Deployment
- Risk catagories were developed through literature review and expert opinion.
- Decision support tools were then deployed in the form of a website.
- API calls were attatched to the website so that the farmer could input their field location to get specific results.


## Installation Instructions:

### Python version: 3.10
### Requred dependencies are 
### RAM requirements: suggusted to have at least 16 GB of RAM available for running models


git clone https://github.com/username/project.git
cd project
pip install -r requirements.txt


### To Run Website Interface

### To Run Exploratory Data Analysis:


### To Run Models:


## Repository Structure:
/data -> Raw and processed data
/website -> website interface
/preparation -> exploratory data analysis and model selection files


## Reproducibility Statement:
Random Seed, test size, model results, and context was documented throughout our process. This allows for reproducibility to be as great as possible across machines and time.

We recognize that API calls, package deployments (as listed in the requirements), and more will change over time, but we hope to diminish the amount of change as much as possble by keeping our work as reproducible as possible.

## Design Decisions:
We chose to keep our preliminary work on evaluating the data (listeria_eda.ipynb) and the model selection (Run_Models_and_Analyze.ipynb) within our repository to show context to our decisions and show our process for the IAFP Student Competition Hackathon. 

## Citations, Thanks, and Recognitions
Data was provided by:
* Liao, J., Guo, X., Weller, D.L. et al. Nationwide genomic atlas of soil-dwelling Listeria reveals effects of selection and population ecology on pangenome evolution. Nat Microbiol 6, 1021–1030 (2021). https://doi.org/10.1038/s41564-021-00935-7

Influence from Models Chosen was developed from:
* Chenhao Qian, Huan Yang, Jayadev Acharya, Jingqiu Liao, Renata Ivanek, Martin Wiedmann,
Initializing a Public Repository for Hosting Benchmark Datasets to Facilitate Machine Learning Model Development in Food Safety, Journal of Food Protection, Volume 88, Issue 3, 2025, 100463, ISSN 0362-028X, https://doi.org/10.1016/j.jfp.2025.100463.

API Calls were made to:
* Livneh daily CONUS near-surface gridded meteorological and derived hydrometeorological data (1915-2011). https://psl.noaa.gov/data/gridded/data.livneh.html


Inspiration for development was prompted by IAFP Student Competition Hackathon 2026.