# Bias in BERT Models

## By Reece Shuttleworth, Samir Amin, and Abhaya Ravikumar

### Setup:

- create and activate venv:

```
python -m venv 63950
source 63950/bin/activate
```

- install dependencies:

```
pip install -r requirements.txt
```

### Usage:

For now just run this command to run the models on our dataset:

```
python eval_model.py
```

Notes:
You can safely ignore the BERT weight missing warning due to https://huggingface.co/bert-base-uncased/discussions/4.
