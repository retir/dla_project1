import gdown
import os

FILE_IDS = {
    "bpe_token_500": ("1onHWT7QAoVmWLWCntwzCaD4fPNdA6JGZ", "500-tokens.tokenizer.json"),
    "bpe_token_100": ("1KsRAVOdBZ6TGDGhy6-Da9x0OH7AMMGTd", "100-tokens.tokenizer.json"),
    "arpa_5_gram": ("1AWz5meMPeJPOcDfb2Gg_RhCIkx14Nv7y", "model_5.arpa"),
    "arpa_6_gram" : ("1-2Hylb8N3SLFm6syvWSOd3W6yUwMGyPh", "model_6.arpa",),
    "best_pretrained" : ("", "best_pretrained.pt")
}


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-f",
        "--files",
        default=None,
        type=str,
        help="List of files to download",
    )
    args.add_argument(
        "-h",
        "--hard",
        default=False,
        type=bool,
        help="Overwrite owloaded files?",
    )

    args = args.parse_args()
    if args.files is None:
        to_load = list(FILE_IDS.keys())
    else:
        to_load = args.files.split(',')
    
    models_dir = './pretrained_models/'
    if not os.path.exists(models_dir):
        os.makedirs(directory)
    
    
    for file_name in to_load:
        print(f'Loading {file_name}...')
        url = 'https://drive.google.com/uc?id=' + FILE_IDS[file_name][0]
        output = models_dir + FILE_IDS[file_name][1]
        if os.path.exists(output) and not args.hard:
            print(f'File {file_name} already exists ({output}), skipping...')
            continue
        gdown.download(url, output, quiet=True)
    print('Done')


   
    