import json

paths = ['data/datasets/librispeech/dev-clean_index.json',
        'data/datasets/librispeech/train-clean-100_index.json',
        'data/datasets/librispeech/train-clean-360_index.json',
        'data/datasets/librispeech/train-other-500_index.json']

to_save = 'texts.txt'
with open(to_save, 'w') as f:
    for path in paths:
        with open(path, 'r') as g:
            corps = json.load(g)
        for corpus in corps:
            f.write(corpus['text'] + '\n')