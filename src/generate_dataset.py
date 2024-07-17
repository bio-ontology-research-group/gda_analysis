import mowl
mowl.init_jvm("10g")
import click as ck
import pandas as pd
import logging
from jpype import *
import jpype.imports
import os
import wget
import sys
import random

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

from mowl.owlapi import OWLAPIAdapter
from mowl.datasets import OWLClasses
from org.semanticweb.owlapi.model import IRI
import java
from java.util import HashSet

adapter = OWLAPIAdapter()
manager = adapter.owl_manager
factory = adapter.data_factory

random.seed(42)

@ck.command()
@ck.option(
    '--organism', '-org', type=ck.Choice(['mouse', 'human']))
@ck.option(
    '--save_dir', '-s', default='../data', help='Directory to save the data')
def main(organism, save_dir):

    out_dir = os.path.abspath(save_dir)
    logger.info(f'Saving data to {out_dir}')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger.info("Cheking if the data is already downloaded")

    if not os.path.exists(os.path.join(out_dir, 'upheno_all.owl')):
        logger.error("File upheno.owl not found. Downloading it...")
        wget.download("https://purl.obolibrary.org/obo/upheno/v2/upheno.owl", out=out_dir)

    if not os.path.exists(os.path.join(out_dir, 'HMD_HumanPhenotype.rpt')):
        logger.error("File HMD_HumanPhenotype.rpt not found. Downloading it for MGI to Entrez Gene mappings")
        wget.download("https://www.informatics.jax.org/downloads/reports/HMD_HumanPhenotype.rpt", out=out_dir)
        
    if not os.path.exists(os.path.join(out_dir, 'MGI_GenePheno.rpt')):
        logger.error("File MGI_GenePheno.rpt not found. Downloading it for Gene-Phenotype associations")
        wget.download("https://www.informatics.jax.org/downloads/reports/MGI_GenePheno.rpt", out=out_dir)
        
    if not os.path.exists(os.path.join(out_dir, 'phenotype.hpoa')):
        logger.error("File phenotype.hpoa not found. Downloading it for Disease-Phenotype associations")
        wget.download("http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa", out=out_dir)
                
    if not os.path.exists(os.path.join(out_dir, 'MGI_DO.rpt')):
        logger.error("File MGI_DO.rpt not found. Downloading it for Gene-Disease associations")
        wget.download("http://www.informatics.jax.org/downloads/reports/MGI_DO.rpt", out=out_dir)
        
    
    logger.info("Loading ontology")
    ont = manager.loadOntologyFromOntologyDocument(java.io.File(os.path.join(out_dir, 'upheno_all.owl')))
    classes = set(OWLClasses(ont.getClassesInSignature()).as_str)

    existing_mp_phenotypes = set()
    existing_hp_phenotypes = set()
    for cls in classes:
        if "MP_" in cls:
            existing_mp_phenotypes.add(cls)
        elif "HP_" in cls:
            existing_hp_phenotypes.add(cls)
    logger.info(f"Existing MP phenotypes in ontology: {len(existing_mp_phenotypes)}")
    logger.info(f"Existing HP phenotypes in ontology: {len(existing_hp_phenotypes)}")
    
    has_symptom = "http://mowl.borg/has_symptom" # relation between diseases and phenotypes
    has_phenotype = "http://mowl.borg/has_phenotype" # relation between genes and phenotypes
    associated_with = "http://mowl.borg/associated_with" # relation between genes and diseases


    logger.info("Obtaining MGI to Entrez Gene mappings")
    hmd = pd.read_csv(os.path.join(out_dir, 'HMD_HumanPhenotype.rpt'), sep='\t')
    hmd.columns = ["HGene", "EntrezID", "MGene", "MGI ID", "Phenotypes", "Extra"]

    mgi_to_entrez = dict()
    for i, row in hmd.iterrows():
        mgi_to_entrez[row["MGI ID"]] = row["EntrezID"]

    
    logger.info("Obtaining Gene-Phenotype associations from MGI_GenePheno.rpt. Genes are represented as MGI IDs and Phenotypes are represented as MP IDs")

    mgi_gene_pheno = pd.read_csv(os.path.join(out_dir, 'MGI_GenePheno.rpt'), sep='\t', header=None)
    mgi_gene_pheno.columns = ["AlleleComp", "AlleleSymb", "AlleleID", "GenBack", "MP Phenotype", "PubMedID", "MGI ID", "MGI Genotype ID"]

    gene_phenotypes = []
    for index, row in mgi_gene_pheno.iterrows():
        genes = row["MGI ID"]
        phenotype = row["MP Phenotype"]
        assert phenotype.startswith("MP:")
        phenotype = "http://purl.obolibrary.org/obo/" + phenotype.replace(":", "_")
        if not phenotype in existing_mp_phenotypes:
            continue

        for gene in genes.split('|'):
            if gene in mgi_to_entrez:
                gene = mgi_to_entrez[gene]
                gene = "http://mowl.borg/" + str(gene)
                gene_phenotypes.append((gene, phenotype))

    logger.info(f"Gene-Phenotype associations: {len(gene_phenotypes)}")
    logger.info(f"\tE.g. {gene_phenotypes[0]}")
    
    logger.info("Obtaining Disease-Phenotype associations from phenotype.hpoa")
    hpoa = pd.read_csv(os.path.join(out_dir, 'phenotype.hpoa'), sep='\t', comment='#')

    disease_phenotypes = []
    for index, row in hpoa.iterrows():
        disease = row["database_id"]
        phenotype = row["hpo_id"]
        if not disease.startswith("OMIM:"):
            continue
        assert phenotype.startswith("HP:")
        disease = "http://mowl.borg/" + disease.replace(":", "_")
        phenotype = "http://purl.obolibrary.org/obo/" + phenotype.replace(":", "_")
        if not phenotype in existing_hp_phenotypes:
            continue
        disease_phenotypes.append((disease, phenotype))
    logger.info(f"Disease-Phenotype associations: {len(disease_phenotypes)}")
    logger.info(f"\tE.g. {disease_phenotypes[0]}")
    
    logger.info("Obtaining Gene-Disease associations from MGI_DO.rpt. Genes are represented as MGI IDs and Diseases are represented as OMIM IDs")
    mgi_do = pd.read_csv(os.path.join(out_dir, 'MGI_DO.rpt'), sep='\t')
    mgi_human = mgi_do[mgi_do["Common Organism Name"] == "human"]
    gene_disease = []
    for index, row in mgi_human.iterrows():
        gene = int(row["EntrezGene ID"])
        if pd.isna(gene):
            logger.warning(f"Gene not found for {row}")
            continue
        diseases = row["OMIM IDs"]
        if pd.isna(diseases):
            # logger.warning(f"Disease not found for {row}")
            continue
        assert diseases.startswith("OMIM:")
        
        diseases = diseases.split("|")
        gene = "http://mowl.borg/" + str(gene)
        for disease in diseases:
            assert disease.startswith("OMIM:")
            disease = "http://mowl.borg/" + disease.replace(":", "_")
            gene_disease.append((gene, disease))

    logger.info(f"Gene-Disease associations: {len(gene_disease)}")
    logger.info(f"\tE.g.: {gene_disease[0]}")

    logger.info("Filtering out x--phenotype pairs with phenotypes not in the ontology")

    gene_phenotypes = [(gene, phenotype) for gene, phenotype in gene_phenotypes if phenotype in classes]
    disease_phenotypes = [(disease, phenotype) for disease, phenotype in disease_phenotypes if phenotype in classes]
    logger.info(f"Gene-Phenotype associations: {len(gene_phenotypes)}")
    logger.info(f"Disease-Phenotype associations: {len(disease_phenotypes)}")
    
    gene_set = set([gene for gene, _ in gene_phenotypes])
    disease_set = set([disease for disease, _ in disease_phenotypes])

    logger.info(f"Updating gene-disease associations")
    gene_disease = [(gene, disease) for gene, disease in gene_disease if gene in gene_set and disease in disease_set]
    logger.info(f"Gene-Disease associations: {len(gene_disease)}")

    logger.info("Splitting the data into training, validation and test sets")
    train, valid, test = split_pairs(gene_disease, "7,1,2", split_by="tail")
    

    gene_phenotype_axioms = HashSet([create_axiom(gene, has_phenotype, phenotype) for gene, phenotype in gene_phenotypes])
    disease_phenotype_axioms = HashSet([create_axiom(disease, has_symptom, phenotype) for disease, phenotype in disease_phenotypes])

    gene_disease_train_axioms = HashSet([create_axiom(gene, associated_with, disease) for gene, disease in train])
    gene_disease_valid_axioms = HashSet([create_axiom(gene, associated_with, disease) for gene, disease in valid])
    gene_disease_test_axioms = HashSet([create_axiom(gene, associated_with, disease) for gene, disease in test])


    manager.addAxioms(ont, gene_phenotype_axioms)
    manager.addAxioms(ont, disease_phenotype_axioms)
    manager.addAxioms(ont, gene_disease_train_axioms)

    ont_file = os.path.join(out_dir, 'train.owl')
    manager.saveOntology(ont, IRI.create('file:' + os.path.abspath(ont_file)))

    valid_ont = manager.createOntology()
    manager.addAxioms(valid_ont, gene_disease_valid_axioms)
    ont_file = os.path.join(out_dir, 'valid.owl')
    manager.saveOntology(valid_ont, IRI.create('file:' + os.path.abspath(ont_file)))

    test_ont = manager.createOntology()
    manager.addAxioms(test_ont, gene_disease_test_axioms)
    ont_file = os.path.join(out_dir, 'test.owl')
    manager.saveOntology(test_ont, IRI.create('file:' + os.path.abspath(ont_file)))


    logger.info("Done")
        

def create_axiom(subclass, property_, filler):
    subclass = adapter.create_class(subclass)
    property_ = adapter.create_object_property(property_)
    filler = adapter.create_class(filler)

    existential_restriction = adapter.create_object_some_values_from(property_, filler)
    subclass_axiom = adapter.create_subclass_of(subclass, existential_restriction)
    return subclass_axiom

#    train, valid, test = load_and_split_interactions(gene_disease, "7,1,2", split_by="tail")    
def split_pairs(pairs, split, split_by="pair"):
    logger.info(f"Splitting {len(pairs)} pairs into training, validation and test sets with split {split}")
    
    train_ratio, valid_ratio, test_ratio = [int(x) for x in split.split(",")]
    assert train_ratio + valid_ratio + test_ratio == 10

    if split_by == "pair":
        raise NotImplementedError

    if split_by == "head":
        raise NotImplementedError

    if split_by == "tail":
        tail_to_heads = dict()
        for head, tail in pairs:
            if tail not in tail_to_heads:
                tail_to_heads[tail] = []
            tail_to_heads[tail].append(head)

        tails = list(tail_to_heads.keys())
        random.shuffle(tails)

        num_train = int(len(tails) * train_ratio / 10)
        num_valid = int(len(tails) * valid_ratio / 10)
        num_test = int(len(tails) * test_ratio / 10)
        
        train_tails = tails[:num_train]
        valid_tails = tails[num_train:num_train+num_valid]
        test_tails = tails[num_train+num_valid:]

        assert len(set(train_tails) & set(valid_tails)) == 0
        assert len(set(train_tails) & set(test_tails)) == 0
        assert len(set(valid_tails) & set(test_tails)) == 0

        train = [(head, tail) for tail in train_tails for head in tail_to_heads[tail]]
        valid = [(head, tail) for tail in valid_tails for head in tail_to_heads[tail]]
        test = [(head, tail) for tail in test_tails for head in tail_to_heads[tail]]

        assert len(set(train) & set(valid)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(valid) & set(test)) == 0

        assert len(train) + len(valid) + len(test) == len(pairs)
        logger.info(f"Train: {len(train)}. Ratio: {len(train)/len(pairs)}")
        logger.info(f"Valid: {len(valid)}. Ratio: {len(valid)/len(pairs)}")
        logger.info(f"Test: {len(test)}. Ratio: {len(test)/len(pairs)}")
        
        return train, valid, test
                                                                                                                                                

if __name__ == '__main__':
    main()
    shutdownJVM()
