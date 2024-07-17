import mowl
mowl.init_jvm("10g")

from mowl.ontology.normalize import ELNormalizer
from mowl.datasets import PathDataset
from mowl.owlapi import OWLAPIAdapter
from org.semanticweb.owlapi.model import IRI

from java.util import HashSet

import os

manager = OWLAPIAdapter().owl_manager


dataset = PathDataset("../data/train.owl")

normalizer = ELNormalizer()
gcis = normalizer.normalize(dataset.ontology)



normalized_ont = manager.createOntology()
for gci_name, gcis in gcis.items():
    j_axioms = HashSet()
    for gci in gcis:
        j_axioms.add(gci.owl_axiom)
    manager.addAxioms(normalized_ont, j_axioms)

ont_file = "../data/train_normalized.owl"
manager.saveOntology(normalized_ont, IRI.create('file:' + os.path.abspath(ont_file)))
