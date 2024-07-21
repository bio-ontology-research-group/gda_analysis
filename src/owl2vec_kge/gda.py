import sys
sys.path.append('../')
import mowl
mowl.init_jvm("10g")
from mowl.models import GraphPlusPyKEENModel
from mowl.utils.random import seed_everything
from mowl.owlapi import OWLAPIAdapter
from mowl.projection import OWL2VecStarProjector
from mowl.walking import DeepWalk

from pykeen.models import TransE

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
@ck.option('--kge_model', '-kge', required=True, type=str)
@ck.option('--embedding_dimension', '-dim', required=True, type=int)
@ck.option('--learning_rate', '-lr', default=0.001, type=float)
@ck.option('--batch_size', '-bs', default=1024, type=int)
@ck.option('--epochs', '-ep', required=True, type=int)
@ck.option("--wandb_description", "-desc", default="default")
@ck.option("--no_sweep", "-ns", is_flag=True)
@ck.option("--only_test", "-ot", is_flag=True)
def main(kge_model, embedding_dimension, learning_rate, batch_size,
         epochs, wandb_description, no_sweep, only_test):

    wandb_logger = wandb.init(entity='ferzcam', project='gda_analysis', group='owl2vec_kge', name=wandb_description)

    if no_sweep:
        wandb_logger.log({'kge_model': kge_model,
                          'embedding_dimension': embedding_dimension,
                          'learning_rate': learning_rate,
                          'batch_size': batch_size,
                          'epochs': epochs,                          
                          })
    else:
        kge_model = wandb_logger.config.kge_model
        embedding_dimension = wandb.config.embedding_dimension
        learning_rate = wandb.config.learning_rate
        batch_size = wandb.config.batch_size
        epochs = wandb.config.epochs
                        
    seed_everything(42)

    root_dir = "../../data/"
    dataset = GDADataset(root_dir)    
    
    out_dir = "../../models"

    
    model_filepath = os.path.join(out_dir, f"owl2vec_{kge_model}_{embedding_dimension}_{learning_rate}_{batch_size}_{epochs}_.model")

    model = GraphPlusPyKEENModel(dataset, model_filepath)
    model.set_projector(OWL2VecStarProjector(bidirectional_taxonomy = False, include_literals = False))
    model.optimizer = th.optim.Adam
    model.lr = learning_rate
    model.batch_size = batch_size
    model.set_kge_method(kge_method_resolver(kge_model), embedding_dim=embedding_dimension, random_seed=42)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model.set_evaluator(GDAEvaluator, device)
    model._kge_method = model._kge_method.to(device)
    
    if not only_test:
        
        model.train(epochs=epochs)
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


def kge_method_resolver(kge_method):
    if kge_method.lower() == "transe":
        return TransE
    else:
        raise ValueError(f"KGE method {kge_method} not supported")

    
if __name__ == "__main__":
    main()
