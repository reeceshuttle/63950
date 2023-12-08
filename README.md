# Bias in BERT Models

## By Reece Shuttleworth, Samir Amin, and Abhaya Ravikumar

### Setup:

- create and activate virtual environment:

```
python -m venv 63950
source 63950/bin/activate
```

- install dependencies:

```
pip install -r requirements.txt
```

- Optional: to run finetuning and use WandB:

```
export WANDB_API_KEY="_KEY_HERE_"
```

### Usage:

For now just run this command to run the models on our dataset:

```
python eval_model.py
```

To run finetuning, run this command:

```
python finetune_model.py
```

To see bias score results, look at `results.ipynb`.

### Notes:

- You can safely ignore the BERT weight missing warning. https://huggingface.co/bert-base-uncased/discussions/4.

- If you get this error `AttributeError: 'BertConfig' object has no attribute 'alibi_starting_size'`, this is due to some weird behavior of the mosaic bert config persisting when you try to load the plain bert model after having loaded the mosaic bert model. Hack fix: just reload/rerun the code without the previous model beforehand.
