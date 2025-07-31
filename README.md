## Synthetic data evaluation

### Aim
Suggest evaluation metrics for assessing multivariate synthetic time-series data.

### metrics_pipeline
<br>├── `main.py`                    # Runs everything
<br>├── `config.py`                  # Configurations paths, folders
<br>├── `data_loader.py`             # Load data
<br>├── `analysis.py`                # metrics: CFID, skew, kurtosis, CFID, DTW, NNDR
<br>├── `report_generator.py`        # HTML report generation
<br>├── `templates/`
<br>│   └── `report_template.html`   # template
<br>├── `outputs/`
<br>│   └── `report.html `           # Output report file       
<br>├── `requirements.txt`           # List of required libraries
<br>├── `data/`						 # Not loaded here 
<br>├── `scripts/`					 # general scripts written in `.ipynb`

### Run script 
Run `main.py`


