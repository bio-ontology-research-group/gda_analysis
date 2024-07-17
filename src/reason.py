import sys
sys.path.append('../')
import mowl
mowl.init_jvm("10g")
from mowl.owlapi import OWLAPIAdapter
from mowl.reasoning import MOWLReasoner
from org.semanticweb.elk.owlapi import ElkReasonerFactory
from java.util import HashSet
from org.semanticweb.owlapi.model import IRI
from org.semanticweb.owlapi.formats import RDFXMLDocumentFormat
from mowl.datasets import PathDataset
import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


dataset = PathDataset("../data/train.owl")

logger.info("Reasoning with ELK over the ontology")
    
reasoner_factory = ElkReasonerFactory()
reasoner = reasoner_factory.createReasoner(dataset.ontology)
mowl_reasoner = MOWLReasoner(reasoner)

classes = dataset.ontology.getClassesInSignature()
subclass_axioms = mowl_reasoner.infer_subclass_axioms(classes)
equivalent_class_axioms = mowl_reasoner.infer_equivalent_class_axioms(classes)

adapter = OWLAPIAdapter()
manager = adapter.owl_manager

axioms = HashSet()
axioms.addAll(subclass_axioms)
axioms.addAll(equivalent_class_axioms)
manager.addAxioms(dataset.ontology, axioms)

ont_file = "../data/train_reasoned.owl"
manager.saveOntology(dataset.ontology, RDFXMLDocumentFormat() ,IRI.create('file:' + os.path.abspath(ont_file)))



