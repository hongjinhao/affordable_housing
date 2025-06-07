# Affordable Housing

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Analysing afforable housing projects in California with Low-Income Housing Tax Credit (LIHTC)

![CDLAC image](images/CDLAC.png)

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
  ``` python
  from affordable_housing.dataset import main
  main()
  ```
  ```bash
  make data
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
--------

