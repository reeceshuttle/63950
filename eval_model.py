import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
import time
from utils import get_word_probability, parse_data

def test_model(model_name: str, tokenizer, data, group):
    assert group in ['race', 'gender']
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, config=config, trust_remote_code=True)
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
            print(f'{sentence} {targets[0]}:{ans[0]}, {targets[1]}:{ans[1]}\n')
            total_difference += abs(ans[0]-ans[1])
    print(f'\ntotal average difference (across {example_count} sentences) for {model_name} on {group}: {total_difference/example_count}\n')
    return total_difference/example_count

if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)
    # start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # mosaic uses this same tokenizer
    parsed_data = parse_data("created_data/", tokenizer)
    # test_model('bert-base-uncased', tokenizer, parsed_data, group='race')
    # test_model('mosaicml/mosaic-bert-base', tokenizer, parsed_data, group='race')
    
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # mosaic uses this same tokenizer
    test_model('bert-base-uncased', tokenizer, parsed_data, group='gender')
    # test_model('mosaicml/mosaic-bert-base', tokenizer, parsed_data, group='gender')


    # print(f'script run time: {round(time.time()-start,2)} sec')
    
    # right now we need to find a way to handle the examples with multiple masks.
    # should we just average them? edit: we just average them.
