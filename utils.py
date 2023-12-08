import os
import torch

def parse_data(directory: str, tokenizer):
    """
    This parses the data that we created and puts them in the form
    {file_name:{sentence1:targets,...}}
    where file_name and sentence1 are strings and targets is a list of strings.
    """
    all_data = {}
    for file in os.listdir(directory):
        with open(directory+file, 'r') as f:
            print(f'processing {file}...')
            file_data={}
            text_data = f.read()
            lines = text_data.split('\n')
            for line in lines:
                if line.strip() == '': continue
                sentence, targets = line.split('<')
                sentence = sentence.strip()
                sentence = sentence.replace("MASK", f"{tokenizer.mask_token}")
                targets = targets.split('=')[1][:-1]
                # make sure to strip all the targets here in case
                targets = [val.strip('][') for val in targets.split(',')]
                targets = [val.strip() for val in targets]
                targets = [val.strip("'") for val in targets]
                file_data[sentence] = targets
        all_data[file] = file_data
    return all_data

def combine_sentences(data):
    """recieves an INDIVIDIAL FILES data."""
    combined = []
    for sentence in data:
        combined.append([sentence, data[sentence]])
    return combined

def extract_loaded_sentences(parsed_data):
    """
    given parsed data,
    this takes the loaded data only and combines them into one list of tuples,
    where each tuple is of the form (sentence, classes)
    """
    combined_data = []
    for file in parsed_data:
        if "neutral" in file: continue
        combined_data += combine_sentences(parsed_data[file])
    return combined_data

def extract_synthetic_sentences(parsed_data, group):
    assert group in ['race', 'gender']
    synthetic_combined_data = []
    for file in parsed_data:
        if "synthetic" in file and "neutral" not in file and group in file:
            print(f'accepted {file}')
            synthetic_combined_data += combine_sentences(parsed_data[file])
    return synthetic_combined_data

def extract_real_sentences(parsed_data, group):
    assert group in ['race', 'gender']
    real_combined_data = []
    for file in parsed_data:
        if "synthetic" not in file and "neutral" not in file and group in file:
            print(f'accepted {file}')
            real_combined_data += combine_sentences(parsed_data[file])
    return real_combined_data

def get_word_probability(model, tokenizer, sentence: str, target_word: str):
    """
    adapted from:
    https://github.com/huggingface/transformers/issues/4612#issuecomment-634834656
    """
    input_ids = tokenizer.encode(sentence, return_tensors='pt')

    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    token_logits = model(input_ids)[0]
    mask_token_logits = token_logits[0, mask_token_index, :]
    mask_token_logits = torch.softmax(mask_token_logits, dim=1)

    # Get the score of token_id
    sought_after_token_id = tokenizer.encode(target_word, add_special_tokens=False) # prefix_space=True)[0]  # 928

    token_score = mask_token_logits[:, sought_after_token_id]
    return torch.mean(token_score).item()