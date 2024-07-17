import sys
sys.path.append('../')
import mowl
mowl.init_jvm("10g")
from mowl.models import SyntacticPlusW2VModel
from mowl.utils.random import seed_everything
from mowl.owlapi import OWLAPIAdapter

from evaluators import GDAEvaluator
from dataset import GDADatasetReasoned
from utils import print_as_md
import click as ck
import os
import wandb
import logging
import torch as th
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@ck.command()
@ck.option('--embedding_dimension', '-dim', required=True, type=int)
@ck.option('--epochs', '-ep', required=True, type=int)
@ck.option('--window_size', '-ws', required=True, type=int)
@ck.option("--wandb_description", "-desc", default="default")
@ck.option("--no_sweep", "-ns", is_flag=True)
@ck.option("--only_test", "-ot", is_flag=True)
def main(embedding_dimension, epochs, window_size, wandb_description, no_sweep, only_test):

    wandb_logger = wandb.init(entity='ferzcam', project='gda_analysis', group='opa2vec', name=wandb_description)

    if no_sweep:
        wandb_logger.log({'embedding_dimension': embedding_dimension,
                          'epochs': epochs,
                          'window_size': window_size})
    else:
        embedding_dimension = wandb.config.embedding_dimension
        epochs = wandb.config.epochs
        window_size = wandb.config.window_size
    
    seed_everything(42)

    root_dir = "../../data/"
    dataset = GDADatasetReasoned(root_dir)    
    
    out_dir = "../../models"

    
    model_filepath = os.path.join(out_dir, f"opa2vec_{embedding_dimension}_{epochs}_{window_size}.model")
    corpus_filepath = os.path.join(out_dir, f"corpus_{embedding_dimension}_{epochs}_{window_size}.txt")

    model = SyntacticPlusW2VModel(dataset, model_filepath=model_filepath, corpus_filepath=corpus_filepath)
    model.set_w2v_model(vector_size=embedding_dimension, workers=16, epochs=epochs, min_count=1, window=window_size)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model.set_evaluator(GDAEvaluator, device)

    if not only_test:
        model.generate_corpus(save=True, with_annotations=True)
        model.train(epochs=epochs)
        model.w2v_model.save(model_filepath)
    else:
        model.from_pretrained(model_filepath)
        
    model.evaluate()

    micro_metrics, macro_metrics = model.metrics

    
    print("Test macro metrics")
    print_as_md(macro_metrics)
    print("\nTest micro metrics")
    print_as_md(micro_metrics)
    
    micro_metrics = {f"micro_{k}": v for k, v in micro_metrics.items()}
    macro_metrics = {f"macro_{k}": v for k, v in macro_metrics.items()}
    wandb_logger.log({**micro_metrics, **macro_metrics})


if __name__ == "__main__":
    main()
