# Affordable Housing

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Analysing housing projects in California applying for tax-exempt bond financing.  

![data pipeline](images/data_pipeline.png)

## Usage
Run Jupyter Notebook to explore the data:
  ```bash
  jupyter-notebook --no-browser notebooks
  ```

## Data Processing
- `affordable_housing/dataset.py`: Merges `award_list.xlsx` and `2025-Applicant-list-4-per-R1.xlsx` from `data/external`, handles NaNs, and saves to `data/processed/merged_dataset.csv`. 

  Run with:
  ```bash
  python affordable_housing/dataset.py
  ``` 
  ```python
  from affordable_housing.dataset import main
  main()
  ```
  ```bash
  make data
  ```

- `affordable_housing/features.py`: Generates machine learning features from `data/processed/merged_dataset.csv`.  
  - Extracts key numeric and categorical columns, splits into train/test, applies preprocessing (including one-hot encoding, scaling, and custom transformations), and saves processed features to `data/processed/`.

  Run with:
  ```bash
  python affordable_housing/features.py
  ```

## Virtual Environment & Package Management

### Setup
- This project uses Python *virtualenvwrapper* for environment management
    > Note: Environments created with virtualenvwrapper (via mkvirtualenv) are stored in ~/.virtualenvs/ and can be activated with workon
- Python version: 3.10
- Create new environment:
  ```bash
  make create_environment
  ```
- Show available environments:
  ```bash
  workon
  ```
- Activate environment:
  ```bash
  workon affordable_housing
  ```
- Deactivate environment:
  ```bash
  deactivate
  ```

### Package Management
- Install all dependencies:
  ```bash
  make requirements
  ```
- When using WSL, you may need to install the package in development mode:
  ```bash
  pip install -e .
  ```
- Key dependencies:
  - pandas, numpy: Data processing
  - scikit-learn: Modeling
  - matplotlib, seaborn: Visualization
  - ruff: Code formatting and linting
  - typer: CLI interface
  - loguru: Logging

### Development Workflow
- After installing new packages:
  ```bash
  pip freeze > requirements.txt
  ```
- Format code:
  ```bash
  make format
  ```
- Check code style:
  ```bash
  make lint
  ```
- Clean Python cache files:
  ```bash
  make clean
  ```

### Getting Help
- List all available make commands:
  ```bash
  make help
  ```





## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         affordable_housing and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── affordable_housing   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes affordable_housing a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```
## References
- `references/jh_grok_applicant_data_overview.pdf`: A PDF containing column descriptions and insights for the applicant dataset at `data/external/2025-Applicant-list-4-per-R1.xlsx`

![applicant data overview](images/applicant-data-overview.png)


- What is this tax-exempt bond allocation via CDLAC?  
Federal law limits how much tax-exempt debt a state can issue in a calendar year for private projects that have a qualified public benefit. This cap is determined by a population-based formula. The California Debt Limit Allocation Committee (CDLAC) was created to set and allocate California’s annual debt ceiling and administer the State’s tax-exempt bond program to allocate the debt authority. CDLAC’s programs are used to finance affordable housing developments for low-income Californians, build solid waste disposal and waste recycling facilities, and to finance industrial development projects.  
https://www.treasurer.ca.gov/cdlac/

- How is the tax-exempt debt allocated?   
Program Categories: CDLAC allocates bonds to multiple programs (e.g., QRRP, Single-Family Housing, Industrial Development Bonds), with QRRP being the largest category. Within QRRP, subcategories like New Construction, Preservation, Rural, and BIPOC address different housing needs (e.g., new builds vs. rehab, rural vs. urban equity).

- What is QRRP?  
The State of California is short 1.5 million affordable housing units. The Qualified Residential Rental Project Program helps spur affordable housing production by assisting developers of multifamily rental housing units with the acquisition and construction of new units, or the purchase and rehabilitation of existing units.  
https://www.treasurer.ca.gov/cdlac/applications/qrrp/index.asp

- How is this tax-exempt bond allocation related to tax credits and CTCAC?  
 Projects that receive an award of bond authority have the right to apply for non-competitive 4% tax credits, administered by the California Tax Credit Allocation Committee.  
 https://www.treasurer.ca.gov/cdlac/applications/qrrp/index.asp

- Who determines the tax-exempt bond allocation via CDLAC?  
Staff review applications, and the committee votes on the final list

- What does the CTCAC do?  
The California Tax Credit Allocation Committee (CTCAC) administers the **federal and state** Low-Income Housing Tax Credit Programs (LIHTC). Both programs were created to promote private investment in affordable rental housing for low-income Californians.

- What is federal LIHTC?  
The Low-Income Housing Tax Credit (LIHTC) program is the most important resource for creating affordable housing in the United States today. Created by the Tax Reform Act of 1986, the LIHTC program gives State and local LIHTC-allocating agencies the equivalent of approximately $10.5 billion in annual budget authority to issue tax credits for the acquisition, rehabilitation, or new construction of rental housing targeted to lower-income households.  
https://www.huduser.gov/portal/datasets/lihtc.html

- Why are there multiple rounds?  
CDLAC’s tax-exempt bond allocations are organized into multiple rounds (e.g., Round 1, Round 2) within a calendar year, with a competitive, points-based system to distribute a limited annual bond volume cap (e.g., $5.126 billion for 2025).

- Source of the award and applicantion list?  
From the CTCAC website  
https://www.treasurer.ca.gov/ctcac/2025/application.asp

- So what does the AWARD column stand for in award_list? 
I am going to assume it means that the housing projects has won the tax-exempt bond allocation (via CDLAC's QRRP program) AND the non-competitive 4% tax credit (via CTCAC's LIHTC program)
--------

