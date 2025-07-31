## Synthetic data evaluation

### Aim
Suggest evaluation metrics for assessing multivariate synthetic time-series data.

### metrics_pipeline
├── `main.py`                    # Runs everything
├── `config.py`                  # Configurations paths, folders
├── `data_loader.py`             # Load data
├── `analysis.py`                # metrics: CFID, skew, kurtosis, CFID, DTW, NNDR
├── `report_generator.py`        # HTML report generation
├── `templates/`
│   └── `report_template.html`   # template
├── `outputs/`
│   └── `report.html `           # Output report file       
├── `requirements.txt`           # List of required libraries
├── `data/`						 # Not loaded here 
├── `scripts/`					 # general scripts written in `.ipynb`

### Run script 
Run `main.py`


