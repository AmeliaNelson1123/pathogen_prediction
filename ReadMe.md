# Project Name: Decision Support Tool for Farmers to Identify Listeria spp. Risk
Project For: IAFP Student Competition

## Authors:
YeonJin Jung (yj354@cornell.edu)
Leonie Kemmerling (lk483@cornell.edu)
Linda Kalunga (lk549@cornell.edu)
Amelia Nelson (aln65@cornell.edu)

# Last Edit Date:
02/20/2026

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
Create 2 powershel terminals:

1 terminal will be designated as the backend, and one terminal will be designated as the frontend

### Backend terminal
``` bash
cd \pathogen_prediction_comp\website\backend
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
``` 

### Frontend terminal
``` bash
cd \pathogen_prediction_comp\website\frontend\farm-app
npm install
npm run dev
```

Once the backend, then frontend have started, click the http link in your terminal to open the webapp.


### To Run Exploratory Data Analysis:
Open preparation\listeria_eda.ipynb, and run all cells with a python 3.10 python interpreter.

### To Run Models used in the website:

### To Run our Modeling Testing Process:
Open preparation\Run_Models_and_Analyze.ipynb, and run all cells with a python 3.10 python interpreter.

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

Influence from Models Chosen was developed from (and public repository created by):
* Chenhao Qian, Huan Yang, Jayadev Acharya, Jingqiu Liao, Renata Ivanek, Martin Wiedmann,
Initializing a Public Repository for Hosting Benchmark Datasets to Facilitate Machine Learning Model Development in Food Safety, Journal of Food Protection, Volume 88, Issue 3, 2025, 100463, ISSN 0362-028X, https://doi.org/10.1016/j.jfp.2025.100463.

API Calls were made to or External Data Downloads were made to:
* Livneh daily CONUS near-surface gridded meteorological and derived hydrometeorological data (1915-2011). https://psl.noaa.gov/data/gridded/data.livneh.html
* For geospatial, land coverage data, to be the most comparable to ARC-GIS, we decided to work with the National Land Cover Database by USGS as hosted by the Google Earch Engine. https://www.usgs.gov/node/279743.


Inspiration for development was prompted by IAFP Student Competition Hackathon 2026.