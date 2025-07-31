## Synthetic data evaluation

### Aim
Suggest evaluation metrics for assessing multivariate synthetic time-series data.

### metrics_pipeline
├── `main.py`
├── `config.py`
<br>├── `data_loader.py`
<br>├── `encoder.py`
<br>├── `analysis.py`
<br>├── `report_generator.py`
<br>├── `templates/`
<br>│   └── `report_template.html`
<br>├── `outputs/`
<br>│   └── `report.html `
<br>├── `requirements.txt`
<br>├── `data/`
<br>├── `scripts/`

### Run script 
Run `main.py`

| script                      | description                    |
|-------------------------------|--------------------------------|
|main.py | Runs everything |
|config.py | Configurations paths, folders |
|data_loader.py  | Load data |
|encoder.py  | Encodes data|
|analysis.py| metrics: CFID, skew, kurtosis, CFID, DTW, NNDR |
|report_generator.py | HTML report generation, saved in outputs |
|requirements.txt| List of required libraries |
|-- | -- |

| folder                     | description                    |
|-------------------------------|--------------------------------|
|data| nothing here |
|scripts| general scripts written in `.ipynb` |
|outputs | results saved here |
|-- | -- |