# Decision Support Tool for Farmers to Identify *Listeria* species Risk
Project for the IAFP Student Competition.

Last edit date: 02/28/2026

## Authors
- YeonJin Jung (yj354@cornell.edu)
- Leonie Kemmerling (lk483@cornell.edu)
- Linda Kalunga (lk549@cornell.edu)
- Amelia Nelson (aln65@cornell.edu)

## Table Of Contents
- [Goal](#goal)
- [Problem Description](#problem-description)
- [Objectives](#objectives)
- [Quick Start](#quick-start)
- [Run The Website (Manual Fallback)](#run-the-website-manual-fallback)
- [Google Earth Engine Setup](#google-earth-engine-setup)
- [How To Use The Website](#how-to-use-the-website)
- [Workflow And Model Results](#workflow-and-model-results)
- [Run Preparation Files](#run-preparation-files)
- [Repository Structure](#repository-structure)
- [Reproducibility Statement](#reproducibility-statement)
- [Design Decisions](#design-decisions)
- [Citations, Thanks, And Recognitions](#citations-thanks-and-recognitions)
- [License](#license)

## Goal
Development of a decision support tool for farmers to identify *Listeria* species (spp.) risk in soil.

## Problem Description
Soil serves as an environmental reservoir for *Listeria* spp., including pathogenic strains such as *Listeria* monocytogenes, which can contaminate fresh produce via preharvest routes such as irrigation runoff, animal intrusion, and rain splash (2, 3). Produce growers have been facing the need to implement proactive risk management, particularly under frameworks such as the Food Safety Modernization Act (FSMA). However, current soil testing strategies are:
1. largely reactive rather than predictive,
2. resource intensive, and
3. lacking standard guidance (1, 4).

While growers often collect data on soil properties (for example pH, nutrients, and organic matter), these data are not routinely leveraged to assess microbial risk. Therefore, a data-driven approach that integrates these data to predict *Listeria* presence would allow for:
1. risk-based soil sampling,
2. development of targeted interventions, and
3. efficient allocation of resources for testing.

## Objectives
The objectives of this project are to:
1. develop predictive models for *Listeria* presence in soil,
2. identify environmental drivers of *Listeria* risk,
3. evaluate model robustness and generalizability, and
4. develop a grower-friendly decision-support tool.

## Quick Start
### Requirements
- Python version: 3.10 to 3.13
- Required website dependencies are listed in `website/requirements.txt`
- RAM: suggested at least 16 GB for model work
- Node.js is required for future frontend developers, not required for website users/testers

### Retrieving The Code
```bash
git clone https://github.com/AmeliaNelson1123/pathogen_prediction.git
cd pathogen_prediction
```

Optional high-level requirements:
```bash
pip install -r requirements.txt
```

### Competition Recommended Startup (Windows), after cloning the repository
Use this if you have a shared competition Earth Engine key file.

1. Copy and paste the json file in the `competition_secrets` folder like the following:
   - `competition_secrets/gee-service-account-key.json`
2. Open Powershell, or in Visual Studio, open a Powershell terminal.
3. cd into your repository (i.e. cd /path/to/pathogen_prediction)
4. From repository root, run:
```powershell
powershell -ExecutionPolicy Bypass -File .\setup_competition.ps1
```
5. When you are done running the website or to refresh it, click your terminal, then press "CTRL" + "c".

The script:
- copies the key to `website/backend/gee-service-account-key.json` for runtime,
- creates `.venv` if needed,
- installs `website/requirements.txt`,
- starts `uvicorn` at `http://127.0.0.1:8000`.

### Competition Recommended Startup (macOS/Linux), after cloning the repository
Use this if you have a shared competition Earth Engine key file.

1. Copy and paste the json file in the `competition_secrets` folder like the following:
   - `competition_secrets/gee-service-account-key.json`
2. Open Terminal (App on Mac computer). (Cmd + Space, type Terminal, Enter)
3. cd into your repository (i.e. cd /path/to/pathogen_prediction)
4. From repository root, run:
```bash
chmod +x ./setup_competition.sh
./setup_competition.sh
```
5. When you are done running the website or to refresh it, click your terminal, then press "CTRL" + "c".

## Run The Website (Manual Fallback)
### Windows (PowerShell), from project root
```powershell
cd website
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### macOS/Linux (zsh), from project root
```bash
cd website
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser.

### Troubleshooting
If you have multiple Python versions, run `python3 --version` (or `python --version`), then use the matching command to create the venv.  
Example 1:
```bash
python3 -m venv .venv
```
Example 2:
``` bash
py -3.10 -m venv .venv
```

## Google Earth Engine Setup
For longitude/latitude-based model runs, Earth Engine credentials are required.

### Option 1: Key Handoff (ONLY if you have access to the drive with the service account keys)
No change is needed from the automatic startup options referenced above:
- [Competition Recommended Startup (Windows)](#competition-recommended-startup-windows)
- [Competition Recommended Startup (macoslinux)](#competition-recommended-startup-macoslinux)

Notes:
- End users do not enter secrets in the web interface.
- End users do not need to edit `main.py`.
- Keys are ignored by `.gitignore` and should never be committed.

### Option 2: Create Your Own Service Account And Key
REQUIREMENTS: Google Earth Engine Account, and Node.js

STEP 1: 

Official setup docs:
- Earth Engine service accounts: https://developers.google.com/earth-engine/guides/service_account
- Earth Engine access control and roles: https://developers.google.com/earth-engine/guides/access_control
- Create service accounts: https://cloud.google.com/iam/docs/service-accounts-create
- Create/manage JSON keys: https://cloud.google.com/iam/docs/keys-create-delete
- IAM role reference: https://cloud.google.com/iam/docs/roles-permissions

Minimum service-account roles for this app's read-only Earth Engine usage:
- `roles/earthengine.viewer` (Earth Engine Resource Viewer)
- `roles/serviceusage.serviceUsageConsumer` (if required by project API access policy)
- `roles/earthengine.writer` (included in this project setup for easier API-related workflows)

Roles typically needed by the human/admin creating service accounts and keys:
- `roles/iam.serviceAccountAdmin`
- `roles/iam.serviceAccountKeyAdmin`


STEP 2:

Once the service account is made, create and download a key.

STEP 3:

Rename the downloaded file to `gee-service-account-key.json`

STEP 4:

Copy the `gee-service-account-key.json` into the folder `competition_secrets` such that `competition_secrets/gee-service-account-key.json`.

STEP 5

Edit line 208, and 211 in `website/backend/main.py` according to the following.

Line 208, change
`` website/backend/main.py
EE_PROJECT = os.getenv("EE_PROJECT", "listeria-prediction-tool")
`` 
to 
`` website/backend/main.py
EE_PROJECT = os.getenv("EE_PROJECT", "your-google-earth-engine-project-name")
``

Line 211, change
`` website/backend/main.py
    "temp-for-iafp-competition@listeria-prediction-tool.iam.gserviceaccount.com",
`` 
to 
`` website/backend/main.py
    "your-service-account-email@your-google-earth-engine-project-name.iam.gserviceaccount.com",
`` 

STEP 6

Create 2 terminals (one for the backend, and one for the frontend)
- In the backend terminal, run (Windows version) `powershell -ExecutionPolicy Bypass -File .\setup_competition.ps1` or (Mac version) `./setup_competition.sh`.
- In the frontend run the following in your terminal
`` bash
cd website/frontend/farm-app
rm -rf
npm run build
``

## How To Use The Website
Run one of the following:

1. Soil-only model:
- Upload a CSV in the `Soil CSV Upload (optional)` section.
- In `Model Mode (with or without soil/coordinates)`, select `Soil Information Only`.
- In `Model Type`, choose one of:
  - Gradient Boosted Model (recommended),
  - Neural Network,
  - SVM.

2. Longitude/latitude-only model (weather/elevation fetched automatically):
- Enter date as `MM/DD/YYYY` (after 2010 and up to 14 days in the future, for example `02/14/2026`).
- Select coordinates by map click or manual entry.
- In `Model Mode (with or without soil/coordinates)`, select `Latitude and Longitude Information Only`.
- In `Model Type`, choose one of:
  - Gradient Boosted Model (recommended),
  - Neural Network,
  - SVM.

3. Combined soil + longitude/latitude model:
- Provide both soil input and coordinate/date input, then run.

For more soil-data format details, use the in-app help button and example CSV files.

Excel to CSV reference:
https://support.microsoft.com/en-us/office/save-a-workbook-to-text-format-txt-or-csv-3e9a9d6c-70da-4255-aa28-fcacf1f081e6  
Use `CSV (comma delimited)` format.

## Workflow And Model Results
### Workflow Diagram
![Workflow Diagram](workflow-diagram.png)

### Training Summary
- *Listeria* data was analyzed in `preparation/listeria_eda.ipynb`.
- Seven model families were tested to evaluate predictive ability (`preparation/Run_and_Test_Models.ipynb`, `preparation/Analyze_Models.ipynb`.
- Top 3 models were selected and tuned using hyperparameters and data engineering.
- Feature selection used literature review, expert insight, permutation/feature importance, and PCA-based evaluation.

### Table 1: Performance Metrics For Best Model Variants (sorted by accuracy)
| model used | scalar_status | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| gbm | standard_scalar | 0.941606 | 0.959459 | 0.934211 | 0.946667 |
| gbm | orig | 0.934307 | 0.946667 | 0.934211 | 0.940397 |
| neural net | standard_scalar | 0.905109 | 0.956522 | 0.868421 | 0.910345 |
| svm | standard_scalar | 0.890511 | 0.896104 | 0.907895 | 0.901961 |
| decision_tree | orig | 0.883212 | 0.905405 | 0.881579 | 0.893333 |
| decision_tree | standard_scalar | 0.868613 | 0.881579 | 0.881579 | 0.881579 |
| random_forest | orig | 0.861314 | 0.880000 | 0.868421 | 0.874172 |
| random_forest | standard_scalar | 0.861314 | 0.880000 | 0.868421 | 0.874172 |
| logistic regression | standard_scalar | 0.846715 | 0.857143 | 0.868421 | 0.862745 |
| knn | standard_scalar | 0.839416 | 0.875000 | 0.828947 | 0.851351 |
| knn | orig | 0.810219 | 0.847222 | 0.802632 | 0.824324 |
| neural net | orig | 0.715328 | 0.753425 | 0.723684 | 0.738255 |
| logistic regression | orig | 0.656934 | 0.716418 | 0.631579 | 0.671329 |
| svm | orig | 0.627737 | 0.676056 | 0.631579 | 0.653061 |

### Deployment Summary
- Top models were saved in `preparation/saving_selected_models_for_pipeline.py`
- Risk categories were developed through literature review and expert opinion.
- Decision support outputs were deployed as a web application.
- API calls are used so growers can input field location and receive location-specific results.

## Run Preparation Files
### Exploratory Data Analysis
Open `preparation/listeria_eda.ipynb` and run all cells with a Python 3.10 interpreter.

### Save Models Used In Website
Run:
- `preparation/saving_selected_models_for_pipeline.py`

### Modeling Test Process
Open `preparation/Run_and_Test_Models.ipynb` and run all cells with a Python 3.10 to 3.13 interpreter to get the models. Then, open `preparation/Analyze_Models.ipynb` and run all cells with a Python 3.10 to 3.13 interpreter to analyze the model results.

## Repository Structure
- `/data` -> Raw and processed data
- `/website` -> Website interface
- `/website/backend/main.py` -> API calls, models, and risk adjustments
- `/website/backend/models/` -> Saved models
- `/website/frontend/farm-app/src/` -> Frontend behavior and UI logic
- `/website/frontend/farm-app/public/` -> Example soil files and static assets
- `/preparation` -> Exploratory analysis, model testing, model analysis, and selection files

## Reproducibility Statement
Random seed, test size, model results, and process context were documented throughout the project to improve reproducibility across machines and time.

API responses, package versions, and external dependencies can change over time. This repository aims to minimize that impact by documenting dependencies and processing choices.

## Design Decisions
Preliminary EDA (`preparation/listeria_eda.ipynb`) and model testing (`Run_and_Test_Models.ipynb`) and selection (`preparation/Analyze_Models.ipynb`) are included to show decision context and full process history for the IAFP 2026 Student Competition Hackathon.  
Selected models are saved by `preparation/saving_selected_models_for_pipeline.py`.

## Citations, Thanks, And Recognitions
Data source:
- Liao, J., Guo, X., Weller, D.L. et al. Nationwide genomic atlas of soil-dwelling *Listeria* reveals effects of selection and population ecology on pangenome evolution. Nat Microbiol 6, 1021-1030 (2021). https://doi.org/10.1038/s41564-021-00935-7

Modeling influence and repository context:
- Chenhao Qian, Huan Yang, Jayadev Acharya, Jingqiu Liao, Renata Ivanek, Martin Wiedmann. Initializing a Public Repository for Hosting Benchmark Datasets to Facilitate Machine Learning Model Development in Food Safety. Journal of Food Protection, Volume 88, Issue 3, 2025, 100463. https://doi.org/10.1016/j.jfp.2025.100463

External API / geospatial references:
- Livneh daily CONUS near-surface gridded meteorological and derived hydrometeorological data (1915-2011): https://psl.noaa.gov/data/gridded/data.livneh.html
- National Land Cover Database (USGS) via Google Earth Engine: https://www.usgs.gov/node/279743

Inspiration for development was prompted by the IAFP Student Competition Hackathon 2026.

## License
This project is licensed under the terms of the Apache 2.0 license.
