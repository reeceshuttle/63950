# from created_data import *
import os
def parse_data():
    directory = "created_data/"
    all_data = {}
    for file in os.listdir(directory):
        with open(directory+file, 'r') as f:
            print(f'processing {file}...')
            file_data={}
            text_data = f.read()
            lines = text_data.split('\n')
            # print(lines)
            for line in lines:
                if line.strip() == '': continue
                sentence, targets = line.split('<')
                sentence = sentence.strip()
                targets = targets.split('=')[1][:-1] # .strip('][').split(',')
                # make sure to strip all the targets here in case
                # print(f'{sentence}<-{targets}')
                targets = [val.strip('][') for val in targets.split(',')]
                targets = [val.strip("'") for val in targets]
                file_data[sentence] = targets
        all_data[file] = file_data
    return all_data

if __name__ == "__main__":
    data = parse_data()
    print(data.keys())
    print(data['reece_race_neutral.txt'].keys())
