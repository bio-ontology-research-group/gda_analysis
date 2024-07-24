import sys
sys.path.append('../')
import mowl
mowl.init_jvm("10g")
from mowl.models import RandomWalkPlusW2VModel
from mowl.utils.random import seed_everything
from mowl.owlapi import OWLAPIAdapter
from mowl.projection import OWL2VecStarProjector
from mowl.walking import Node2Vec
from evaluators import GDAEvaluator
from dataset import GDADataset
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
@ck.option('--num_walks', '-nw', required=True, type=int)
@ck.option('--walk_length', '-wl', required=True, type=int)
@ck.option('--p', '-p', required=True, type=float)
@ck.option('--q', '-q', required=True, type=float)
@ck.option("--wandb_description", "-desc", default="default")
@ck.option("--no_sweep", "-ns", is_flag=True)
@ck.option("--only_test", "-ot", is_flag=True)
def main(embedding_dimension, epochs, window_size, num_walks,
         walk_length, p, q, wandb_description, no_sweep, only_test):

    wandb_logger = wandb.init(entity='ferzcam', project='gda_analysis', group='owl2vec', name=wandb_description)

    if no_sweep:
        wandb_logger.log({'embedding_dimension': embedding_dimension,
                          'epochs': epochs,
                          'window_size': window_size,
                          'num_walks': num_walks,
                          'walk_length': walk_length,
                          'p': p,
                          'q': q
                          })
    else:
        embedding_dimension = wandb.config.embedding_dimension
        epochs = wandb.config.epochs
        window_size = wandb.config.window_size
        num_walks = wandb.config.num_walks
        walk_length = wandb.config.walk_length
        p = wandb.config.p
        q = wandb.config.q
    
    seed_everything(42)

    root_dir = "../../data/"
    dataset = GDADataset(root_dir)    
    
    out_dir = "../../models"

    
    model_filepath = os.path.join(out_dir, f"owl2vec_{embedding_dimension}_{epochs}_{window_size}_{num_walks}_{walk_length}_p_{p}_q_{q}.model")
    corpus_filepath = os.path.join(out_dir, f"corpus_owl2vec_{embedding_dimension}_{epochs}_{window_size}_{num_walks}_{walk_length}_p_{p}_q_{q}.txt")

    model = RandomWalkPlusW2VModel(dataset, model_filepath=model_filepath)
    model.set_projector(OWL2VecStarProjector(bidirectional_taxonomy = False, include_literals = True))
    model.set_walker(Node2Vec(num_walks, walk_length, p=p, q=q, workers=10, outfile=corpus_filepath))
    model.set_w2v_model(vector_size=embedding_dimension, workers=16, epochs=epochs, min_count=1, window=window_size)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model.set_evaluator(GDAEvaluator, device)

    if not only_test:
        model.train(epochs=epochs)
        model.w2v_model.save(model_filepath)
    else:
        model.from_pretrained(model_filepath)

    os.remove(corpus_filepath)
        
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
