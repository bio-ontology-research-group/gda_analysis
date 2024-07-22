import mowl
mowl.init_jvm("10g")

from mowl.datasets import PathDataset
from mowl.projection import TaxonomyWithRelationsProjector

ds = PathDataset("../../data/test.owl")

projector = TaxonomyWithRelationsProjector(taxonomy=False, relations=["http://mowl.borg/associated_with"])

edges = projector.project(ds.ontology)

out_file = "../../data/test.csv"
with open(out_file, "w") as f:
    for edge in edges:
        src, rel, dst = edge.src, edge.rel, edge.dst
        f.write(f"{src},{rel},{dst}\n")
    
