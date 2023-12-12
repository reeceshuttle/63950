# Bias in BERT Models

## By Reece Shuttleworth, Samir Amin, and Abhaya Ravikumar

### Links: [[WandB runs]](https://wandb.ai/finetuning-bert/finetuning-bert)

### Note: We have removed the data from this repository. Therefore, the code will no longer run!

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

- If you want to run our finetuning script, install WandB and export your API key(alternatively, delete the lines of code using WandB. It is only used for logging):

```
pip install wandb
export WANDB_API_KEY="_YOUR_KEY_HERE_"
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

To see bias score results, look at `race_results.ipynb` and `gender_results.ipynb`.

### Notes:

- You can safely ignore the BERT weight missing warning. https://huggingface.co/bert-base-uncased/discussions/4.

- If you get this error `AttributeError: 'BertConfig' object has no attribute 'alibi_starting_size'`, this is due to some weird behavior of the mosaic bert config persisting when you try to load the plain bert model after having loaded the mosaic bert model. Hack fix: just reload/rerun the code without the previous model beforehand.
