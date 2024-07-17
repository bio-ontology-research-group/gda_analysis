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
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@ck.command()
@ck.option("--wandb_description", "-desc", default="default")
@ck.option("--no_sweep", "-ns", is_flag=True)
@ck.option("--only_test", "-ot", is_flag=True)
def main(wandb_description, no_sweep, only_test):

    seed_everything(42)

    root_dir = "../../data/"
    dataset = GDADatasetReasoned(root_dir)    
    
    out_dir = "../../models"

    
    model_filepath = os.path.join(out_dir, "opa2vec.model")
    corpus_filepath = os.path.join(out_dir, "corpus.txt")
    model = SyntacticPlusW2VModel(dataset, model_filepath=model_filepath, corpus_filepath=corpus_filepath)
    model.set_w2v_model(vector_size= 200, workers=10, epochs=30, min_count=1)
    model.set_evaluator(GDAEvaluator, "cuda")

    if not only_test:
        model.generate_corpus(save=True, with_annotations=True)
        model.train(epochs=20)
        model.w2v_model.save(model_filepath)
    else:
        model.from_pretrained(model_filepath)
        
    model.evaluate()

    micro_metrics, macro_metrics = model.metrics

    print("Test macro metrics")
    print_as_md(macro_metrics)
    print("\nTest micro metrics")
    print_as_md(micro_metrics)
    


if __name__ == "__main__":
    main()
