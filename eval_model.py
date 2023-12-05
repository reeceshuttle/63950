import os
import torch
from transformers import AutoTokenizer, BertForMaskedLM
import time

def parse_data(directory: str, tokenizer):
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
                targets = [val.strip("'") for val in targets]
                file_data[sentence] = targets
        all_data[file] = file_data
    return all_data

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

if __name__ == "__main__":
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    data = parse_data("created_data/", tokenizer)
    for file in data:
        print(f'testing based on {file}')
        for sentence in data[file]:
            targets = data[file][sentence]
            assert len(targets) == 2
            ans = [round(get_word_probability(model, tokenizer, sentence, target), 8) for target in targets]
            print(f'{sentence} {targets[0]}:{ans[0]}, {targets[1]}:{ans[1]}')
            difference = abs(ans[0]-ans[1])

    print(f'script run time: {round(time.time()-start,2)} sec')
    
    # right now we need to find a way to handle the examples with multiple masks.
    # should we just average them? edit: we just average them.
