import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
import time
from utils import get_word_probability, parse_data

def test_model(model_name: str, tokenizer, data):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, config=config, trust_remote_code=True)
    total_difference = 0
    example_count = 0
    for file in data:
        if "neutral" in file: 
            print(f'skipping {file}')
            continue
        print(f'testing based on {file}...')
        for sentence in data[file]:
            example_count+=1
            targets = data[file][sentence]
            assert len(targets) == 2
            ans = [round(get_word_probability(model, tokenizer, sentence, target), 8) for target in targets]
            # print(f'{sentence} {targets[0]}:{ans[0]}, {targets[1]}:{ans[1]}')
            total_difference += abs(ans[0]-ans[1])
    print(f'total average difference for {model_name}:{total_difference/example_count}')
    return total_difference/example_count

if __name__ == "__main__":
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # mosaic uses this same tokenizer
    parsed_data = parse_data("created_data/", tokenizer)
    test_model('bert-base-uncased', tokenizer, parsed_data)
    test_model('mosaicml/mosaic-bert-base', tokenizer, parsed_data)
    print(f'script run time: {round(time.time()-start,2)} sec')
    
    # right now we need to find a way to handle the examples with multiple masks.
    # should we just average them? edit: we just average them.
