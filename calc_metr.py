import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import hw_asr.model as module_model
from hw_asr.trainer import Trainer
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser
import hw_asr.metric as module_metric
from hw_asr.utils import inf_loop, MetricTracker
from hw_asr.logger import get_visualizer
from hw_asr.metric.bs_predictors import BSPredicor

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"



class Evaluator:
    def __init__(self, config, usable_sets):
        self.config = config
        self.usable_sets = usable_sets
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_encoder = config.get_text_encoder()
        self.dataloaders = get_dataloaders(config, self.text_encoder)
        self.test_metrics = [config.init_obj(metric_dict, module_metric, text_encoder=self.text_encoder)
            for metric_dict in config["test_metrics"]]
        self.bs_predictor = config.init_obj(config["bs_predictor"], module_metric, text_encoder=self.text_encoder)
        #self.bs_predictor = BSPredicor(self.text_encoder, config.config['beam_size'])
        self.use_bs_predictor = False
        for met in self.test_metrics:
            if met.use_bs_pred:
                self.use_bs_predictor = True
    
        self.metrics = MetricTracker(*[m.name for m in self.test_metrics])
                
        self.model = config.init_obj(config["arch"], module_model, n_class=len(self.text_encoder))
        
        print("Loading checkpoint: {} ...".format(config.resume))
        checkpoint = torch.load(config.resume, map_location=self.device)
        state_dict = checkpoint["state_dict"]
        if config["n_gpu"] > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(state_dict)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        
    def get_results(self):
        results = {}
        with torch.no_grad():
            for loader_name in self.usable_sets:
                loader = self.dataloaders[loader_name]
                for batch_num, batch in enumerate(tqdm(loader)):
                    batch = Trainer.move_batch_to_device(batch, self.device)
                    output = self.model(**batch)
                    if type(output) is dict:
                        batch.update(output)
                    else:
                        batch["logits"] = output
                    batch["log_probs"] = torch.log_softmax(batch["logits"], dim=-1)
                    batch["log_probs_length"] = self.model.transform_input_lengths(
                        batch["spectrogram_length"]
                    )
                    batch["probs"] = batch["log_probs"].exp().cpu()
                    batch["argmax"] = batch["probs"].argmax(-1)
                    if self.use_bs_predictor:
                        batch["predictions"] = self.bs_predictor(**batch)
                    for met in self.test_metrics:
                        self.metrics.update(met.name, met(**batch))
                
                
                results[loader_name] = self.metrics.result()
        self.metrics.reset()
        return results
        

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=50,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )
    args.add_argument(
        "-u",
        "--usable_sets",
        default=None,
        type=str,
        help="List of datasets to eval",
    )
    

    args = args.parse_args()


    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)
    
    if args.config is not None:
        with Path(args.config).open() as f:
            data_config = json.load(f)
        for key in data_config.keys():
            config.config[key] = data_config[key]
    


    if args.usable_sets is None:
        usable_sets = [k for k in config["data"].keys() if k != 'train']
    else:
        usable_sets = list(args.usable_sets.split(','))
    
    
    print(usable_sets)
    for set in usable_sets:
        assert config["data"].get(set, None) is not None
    
    for key in config["data"].keys():
        config["data"][key]["batch_size"] = args.batch_size
        config["data"][key]["n_jobs"] = args.jobs

    #main(config)
    evaluator = Evaluator(config, usable_sets)
    results = evaluator.get_results()
    
    for dataset, res in results.items():
        print()
        print(dataset)
        for metric, val in res.items():
            print(f'{metric}: {val}')
    
    
    
    
