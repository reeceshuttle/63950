import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig, BertForMaskedLM
import time
from utils import get_word_probability, parse_data

def test_model(model_name: str, tokenizer, data, group):
    assert group in ['race', 'gender']
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, config=config, trust_remote_code=True)
    model.eval()
    total_difference = 0
    example_count = 0
    for file in data:
        if "neutral" in file or "synthetic" in file or group not in file: 
            # print(f'skipping {file}')
            continue

        print(f'testing based on {file}...')
        for sentence in data[file]:
            example_count+=1
            targets = data[file][sentence]
            assert len(targets) == 2
            ans = [round(get_word_probability(model, tokenizer, sentence, target), 8) for target in targets]
            # print(f'{sentence} {targets[0]}:{ans[0]}, {targets[1]}:{ans[1]}\n')
            total_difference += abs(ans[0]-ans[1])
    print(f'\ntotal average difference (across {example_count} sentences) for {model_name} on {group}: {total_difference/example_count}. w/o norm: {total_difference}\n')
    return total_difference/example_count

def test_finetuned_model(model_name: str, path, tokenizer, data, group):
    assert group in ['race', 'gender']

    config = AutoConfig.from_pretrained(model_name)
    model = BertForMaskedLM(config)
    model.load_state_dict(torch.load(path))
    model.eval()

    total_difference = 0
    example_count = 0
    for file in data:
        if "neutral" in file or "synthetic" in file or group not in file: 
            # print(f'skipping {file}')
            continue

        print(f'testing based on {file}...')
        for sentence in data[file]:
            example_count+=1
            targets = data[file][sentence]
            assert len(targets) == 2
            ans = [round(get_word_probability(model, tokenizer, sentence, target), 8) for target in targets]
            # print(f'{sentence} {targets[0]}:{ans[0]}, {targets[1]}:{ans[1]}\n')
            total_difference += abs(ans[0]-ans[1])
    print(f'\ntotal average difference (across {example_count} sentences) for {model_name} on {group}: {total_difference/example_count}. w/o norm: {total_difference}\n')
    return total_difference/example_count

def test_model_preready(model, tokenizer, parsed_data):
    model.eval()
    total_difference = 0
    example_count = 0
    for sentence, targets in parsed_data:
        example_count+=1
        assert len(targets) == 2
        ans = [round(get_word_probability(model, tokenizer, sentence, target), 8) for target in targets]
        # print(f'{sentence}: {ans}')
        total_difference += abs(ans[0]-ans[1])
    return total_difference/example_count

if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # mosaic uses this same tokenizer
    parsed_data = parse_data("created_data/", tokenizer)
    test_model('bert-base-uncased', tokenizer, parsed_data, group='race')
    test_model('mosaicml/mosaic-bert-base', tokenizer, parsed_data, group='race')
    
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # mosaic uses this same tokenizer
    # test_model('bert-base-uncased', tokenizer, parsed_data, group='gender')
    # test_model('mosaicml/mosaic-bert-base', tokenizer, parsed_data, group='gender')

    
    # right now we need to find a way to handle the examples with multiple masks.
    # should we just average them? edit: we just average them.