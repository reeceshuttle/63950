import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
from eval_model import test_model_preready
import time
from utils import *

import wandb
import random

    
def word_smoothing_loss(mask_logits, loss_fn: nn.MSELoss, tokens_to_even, scaling=True):
    """Logits is of shape (nummask, dictsize)"""
    output = torch.softmax(mask_logits, dim=-1)

    total_score = torch.zeros((mask_logits.shape[0], 1)) # (nummask, 1)
    for token_id in tokens_to_even:
        # print(f'total_score:{total_score}, adding:{output[:, token_id]}')
        total_score += output[:, token_id]
    target_scores = total_score / 2 # since classes are length 2
    target_output = torch.softmax(mask_logits, dim=-1).detach() # this is a waste of an op, but is used to copy
    # target_output = output.detach() # since detach returns a new tensor
    target_output
    for token_id in tokens_to_even:
        target_output[:, token_id] = target_scores
    init_loss = loss_fn(input=output, target=target_output)

    # here we scale the loss up but clip it to be 1.
    if scaling:
        scaling_factor = 1000 # we want loss to be in a better range

        if 1/scaling_factor < init_loss:
            return init_loss*1/(init_loss.detach())
        else:
            return scaling_factor*init_loss
    else:
        return init_loss


def train_epoch(model, tokenizer, training_data, optimizer, loss_fn):
    model.train()
    total_loss = torch.zeros((1))
    for sentence, classes in training_data: # since batch size 1, sentence and classes each have 1 training example.
        # print(f'{sentence}:{classes}')
        input_tokens = tokenizer.encode(sentence, return_tensors='pt')
        # print(f'input_tokens:{input_tokens}')
        mask_token_index = torch.where(input_tokens == tokenizer.mask_token_id)[1]
        # print(f'mask_token_index:{mask_token_index}')
        logits = model(input_tokens).logits
        # print(f'logits:{logits.shape}')
        mask_token_logits = logits[0, mask_token_index, :]
        # print(f'mask_logits:{mask_token_logits.shape}')
        tokens_to_even = [tokenizer.encode(word, add_special_tokens=False) for word in classes]
        # print(f'tokens_to_even:{tokens_to_even}')
        loss = word_smoothing_loss(mask_token_logits, loss_fn, tokens_to_even)
        total_loss += loss
        # print(f'loss:{loss}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss

def test(model, tokenizer, testing_data, loss_fn):
    model.eval()
    total_loss = torch.zeros((1))
    for sentence, classes in testing_data:
        input_tokens = tokenizer.encode(sentence, return_tensors='pt')
        mask_token_index = torch.where(input_tokens == tokenizer.mask_token_id)[1]
        logits = model(input_tokens).logits
        mask_token_logits = logits[0, mask_token_index, :]
        tokens_to_even = [tokenizer.encode(word, add_special_tokens=False) for word in classes]
        loss = word_smoothing_loss(mask_token_logits, loss_fn, tokens_to_even)
        total_loss += loss
    return total_loss

def wandb_logging(model, tokenizer, training_data, testing_data, generic_data, loss_fn, group, epoch):
    # print(f'doing train loss:\n')
    train_loss = test(model, tokenizer, training_data, loss_fn)
    # print('doing train bias score')
    train_bias_score = test_model_preready(model, tokenizer, training_data)
    # print(f'train bias score:{train_bias_score}\n\n') # debugging nan behavior
    test_loss = test(model, tokenizer, testing_data, loss_fn)
    test_bias_score = test_model_preready(model, tokenizer, testing_data)
    to_log = {'epoch':epoch,
              'train_loss':train_loss.item(), 
              'test_loss': test_loss.item(),
              'train_bias_score': train_bias_score,
              'test_bias_score': test_bias_score,
              }
    generic_loss = test(model, tokenizer, generic_data, loss_fn)
    generic_bias_score = test_model_preready(model, tokenizer, generic_data)
    to_log['generic_loss_race'] = generic_loss.item()
    to_log['generic_bias_score_race'] = generic_bias_score

    wandb.log(to_log)

def full_training_loop(group):
    assert group in ['race', 'gender']
    # --------
    # vars:
    lr = 0.00001
    epochs = 15
    # --------
    # initing things:
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    parsed_data = parse_data("created_data/", tokenizer)
    training_data = extract_synthetic_sentences(parsed_data, group=group)
    print(f'training data size: {len(training_data)}')
    testing_data = extract_real_sentences(parsed_data, group=group)
    print(f'testing data size: {len(testing_data)}')
    generic_data = extract_generic_sentences(parsed_data)

    loss_fn = nn.MSELoss(reduction='sum') # dont average because the loss is super small
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # --------
    
    print(f'training for {epochs} epochs')
    wandb.init(
    project="finetuning-bert",
    name = f"{lr}-{group}",
    config={
    "learning_rate": lr,
    "architecture": "BERT",
    "dataset": "handmade bias data. train: synthetic. test: real",
    "total_epochs": epochs,
    "group": group})
    # before training logging:
    wandb_logging(model, tokenizer, training_data, testing_data, generic_data, loss_fn, group, epoch=0)
    
    shuffled_data = training_data
    for epoch in range(epochs):
        epoch_start = time.time()
        random.shuffle(shuffled_data)
        total_loss_of_epoch = train_epoch(model, tokenizer, shuffled_data, optimizer, loss_fn)
        wandb_logging(model, tokenizer, training_data, testing_data, generic_data, loss_fn, group, epoch=epoch+1)
        print(f'epoch {epoch+1} time: {round(time.time()-epoch_start,2)} sec')
    return model

if __name__ == "__main__":
    torch.manual_seed(0) # for reproducibility
    random.seed(0) # for reproducibility
    group = 'race'
    finetuned_model = full_training_loop(group=group)
    torch.save(finetuned_model.state_dict(), f'finetuned_bert_{group}.pth')

    # should we do batch size 1? Pure SGD
    # should we use adamW?