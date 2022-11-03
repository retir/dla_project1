# ASR project

## Installation guide

First of all you need to download the repository: 

```shell
git clone https://github.com/retir/dla_project1
cd dla_project1
```
Then install necessary dependencies:

```shell
pip install -r ./requirements.txt
pip install https://github.com/kpu/kenlm/archive/master.zip
```

If you want to use pretrained model, you can load it as follow:

```shell
python3 downloader.py
```
Нou can download only the files you need by listing them via `,` in flag `--files` (as example `--files='bpe_token_500,arpa_5_gram'`). Also you can rewrite already loaded files using flag `--hard`, in other case loading such a files will be skipped.

To calculate metrics use follows:

```shell
python3 calc_metr.py -r path/to/model.pt -c path/to/config.json
```

where model.pt means pretrained model, and config.json is a config with datasets, metrics and beam search parameters in it (you can find example at hw_asr/configs/eval.json). It is necessary that the folder with the model contains the `config.json` file - the config on which the model was trained. 

Pretrained model from `downloader.py` was learnd with config `deepspeech.json` (`hw_asr/configs/deepspeech.json`). To learn model use follow:

```shell
python3 train.py -c path/to/config.json
```

You can find more detailed report about experiments at [wandb report](https://wandb.ai/retir/asr_project/reports/DLA-HW-1--VmlldzoyODAyNjIz?accessToken=g3ixhc6s1e2nlgls5i9flw0e51y2qj6fq0gxowwez4214eo3y6f0lz9znz03q7nt)

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

## Docker

You can use this project with docker. Quick start:

```bash 
docker build -t my_hw_asr_image . 
docker run \
   --gpus '"device=0"' \
   -it --rm \
   -v /path/to/local/storage/dir:/repos/asr_project_template/data/datasets \
   -e WANDB_API_KEY=<your_wandb_api_key> \
	my_hw_asr_image python -m unittest 
```

Notes:

* `-v /out/of/container/path:/inside/container/path` -- bind mount a path, so you wouldn't have to download datasets at
  the start of every docker run.
* `-e WANDB_API_KEY=<your_wandb_api_key>` -- set envvar for wandb (if you want to use it). You can find your API key
  here: https://wandb.ai/authorize
