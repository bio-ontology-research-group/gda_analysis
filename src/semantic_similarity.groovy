@Grab(group='com.github.sharispe', module='slib-sml', version='0.9.1')
@Grab(group='net.sourceforge.owlapi', module='owlapi-api', version='4.2.5')
@Grab(group='net.sourceforge.owlapi', module='owlapi-apibinding', version='4.2.5')
@Grab(group='net.sourceforge.owlapi', module='owlapi-impl', version='4.2.5')
@Grab(group='ch.qos.logback', module='logback-classic', version='1.2.3')
@Grab(group='org.slf4j', module='slf4j-api', version='1.7.30')
@Grab(group='org.codehaus.gpars', module='gpars', version='1.1.0')
@Grab('me.tongfei:progressbar:0.9.3')


import org.semanticweb.owlapi.model.*
import  org.semanticweb.owlapi.apibinding.OWLManager

import slib.sml.sm.core.engine.SM_Engine
import slib.sml.sm.core.measures.Measure_Groupwise
import slib.sml.sm.core.metrics.ic.utils.*
import slib.sml.sm.core.utils.SMConstants
import slib.utils.ex.SLIB_Exception
import slib.sml.sm.core.utils.SMconf
import slib.graph.model.impl.graph.memory.GraphMemory
import slib.graph.io.conf.GDataConf
import slib.graph.io.util.GFormat
import slib.graph.io.loader.GraphLoaderGeneric
import slib.graph.model.impl.repo.URIFactoryMemory
import slib.graph.model.impl.graph.elements.*
import slib.sml.sm.core.metrics.ic.utils.*


import org.openrdf.model.vocabulary.RDF

import groovyx.gpars.GParsPool

import java.util.HashSet

import groovy.cli.commons.CliBuilder
import java.nio.file.Paths


import org.slf4j.Logger
import org.slf4j.LoggerFactory

// Initialize the logger

import java.util.logging.Logger
import java.util.logging.Level
import java.util.logging.ConsoleHandler
import java.util.logging.SimpleFormatter

// Initialize the logger
Logger logger = Logger.getLogger(this.class.name)

// Configure the logger
ConsoleHandler handler = new ConsoleHandler()
handler.setLevel(Level.ALL)
handler.setFormatter(new SimpleFormatter())
logger.addHandler(handler)
logger.setLevel(Level.ALL)
logger.setUseParentHandlers(false)




def cli = new CliBuilder(usage: 'semantic_similarity.groovy -r <root_dir> -ic <ic_measure> -pw <pairwise_measure> -gw <groupwise_measure>')
cli.r(longOpt: 'root_dir', args: 1, defaultValue: '../data', 'Root directory')
cli.ic(longOpt: 'ic_measure', args: 1, defaultValue: 'resnik', 'Information Content measure')
cli.pw(longOpt: 'pairwise_measure', args: 1, defaultValue: 'resnik', 'Pairwise measure')
cli.gw(longOpt: 'groupwise_measure', args: 1, defaultValue: 'bma', 'Groupwise measure')

def options = cli.parse(args)
if (!options) return

String rootDir = options.r
String icMeasure = options.ic
String pairwiseMeasure = options.pw
String groupwiseMeasure = options.gw

def manager = OWLManager.createOWLOntologyManager()
def ontology = manager.loadOntologyFromOntologyDocument(new File(rootDir + "/train.owl"))

def classes = ontology.getClassesInSignature().collect { it.toStringID() }

def evalGenes = classes.findAll {
    def parts = it.split("/")
    def lastPart = parts[-1]
    lastPart.isNumber() && it.contains("mowl.borg")
}.sort()

logger.info("Total evaluation genes: ${evalGenes.size()}")

def existingMpPhenotypes = new HashSet()
def existingHpPhenotypes = new HashSet()
classes.each { cls ->
    if (cls.contains("MP_")) {
        existingMpPhenotypes.add(cls)
    } else if (cls.contains("HP_")) {
        existingHpPhenotypes.add(cls)
    }
}



logger.info("Obtaining Gene-Phenotype associations from HMD_HumanPhenotype.rpt. Genes are represented as Entrez Gene IDs and Phenotypes are represented as MP IDs")
logger.info("Obtaining MGI to Entrez Gene ID mapping from HMD_HumanPhenotype.rpt")
def file = new File(rootDir + "/HMD_HumanPhenotype.rpt")
def hmdHuman = file.readLines()*.split('\t')

def mgiToEntrez = new HashMap()

hmdHuman.each { line ->
	def mgi = line[3]
	def entrez = line[1]
	mgiToEntrez[mgi] = "http://mowl.borg/" + entrez
}

if (mgiToEntrez.containsKey("MGI:1927290")) {
    logger.info("MGI:1927290 -> ${mgiToEntrez["MGI:1927290"]}")
}else {
    logger.info("MGI:1927290 not found")
}


logger.info("Obtaining Gene-Phenotype associations from MGI_GenePheno.rpt. Genes are represented as MGI IDs and Phenotypes are represented as MP IDs")
def gene2pheno = new HashMap()

// MP_0002169
// MGI:1927290 
def mgiGenePhenoFile = new File(rootDir + "/MGI_GenePheno.rpt")
def mgiGenePheno = mgiGenePhenoFile.readLines()*.split('\t')

mgiGenePheno.each { line ->
    def genes = line[6].split("\\|")
    def phenotype = "http://purl.obolibrary.org/obo/" + line[4].replace(":", "_")
    
    if (phenotype in existingMpPhenotypes) {

	genes.each { gene ->
	    def entrez = mgiToEntrez[gene]
	    if (entrez != null) {
		if (!gene2pheno.containsKey(entrez)) {
		    gene2pheno[entrez] = new HashSet()
		}
		gene2pheno[entrez].add(phenotype)
	    }
	}
    }else {
	genes.each { gene ->
	    if (mgiToEntrez[gene] == "http://mowl.borg/10009") {
		logger.info("Phenotype $phenotype not found in MP for gene $gene with entrez id 10009")
	    }
	}
	
    }
    
}

logger.info("gene2pheno size: ${gene2pheno.size()}")

// new URL ("file:../data/MGI_GenePheno.rpt").getText().splitEachLine("\t") { line ->
  // def geneid = line[6]
  // def idUri = factory.getURI("http://phenomebrowser.net/ismb-tutorial/gene/"+geneid)
  // def pheno = line[4].replaceAll(":","_")
  // def phenoUri = factory.getURI("http://purl.obolibrary.org/obo/"+pheno)
  // Edge e = new Edge(idUri, RDF.TYPE, phenoUri)
  // graph.addE(e)
// }

    
// hmdHuman.each { line ->
    // def parts = line
    // def entrezGeneId = "http://mowl.borg/" + parts[1]
    // logger.debug("line: ${line}")
    // def phenotype_string = parts[4]
    // def phenotypes = phenotype_string.split(", ").collect { "http://purl.obolibrary.org/obo/" + it.replace(":", "_") }
    // def existingPhenotypes = phenotypes.findAll { existingMpPhenotypes.contains(it) }
    
    // if (! gene2pheno.containsKey(entrezGeneId)) {
		// gene2pheno[entrezGeneId] = []
    // }
    // gene2pheno[entrezGeneId].addAll(existingPhenotypes)
// }


def test_ontology = manager.loadOntologyFromOntologyDocument(new File(rootDir + "/test.owl"))

def testPairs = axiomsToPairs(test_ontology, logger)
def testGenes = testPairs.collect { it[0] }.unique().collect { it.split("/").last() }
def testDiseases = testPairs.collect { it[1] }.unique().collect { it.split("/").last().replace("_", ":") }

logger.info("Obtaining Disease-Phenotype associations from phenotype.hpoa")
def hpoaFile = new File(rootDir + "/phenotype.hpoa")
def hpoa = hpoaFile.readLines().tail().tail().tail().tail().tail()*.split('\t')

def disease2pheno = new HashMap()

hpoa.each { line ->
    def parts = line
    def disease = "http://mowl.borg/" + parts[0].replace(":", "_")
    def phenotype = "http://purl.obolibrary.org/obo/" + parts[3].replace(":", "_")

    if (existingHpPhenotypes.contains(phenotype)) {
	if (! disease2pheno.containsKey(disease)) {
	    disease2pheno[disease] = []
	}
	disease2pheno[disease].add(phenotype)
    }
}

logger.info("Preparing Semantic Similarity Engine")
def factory = URIFactoryMemory.getSingleton()
def graphUri = factory.getURI("http://purl.obolibrary.org/obo/GDA_")
factory.loadNamespacePrefix("GDA", graphUri.toString())
def graph = new GraphMemory(graphUri)

def withAnnotations = true

if (withAnnotations) {
    gene2pheno.each { gene, phenotypes ->
	phenotypes.each { phenotype ->
	    // logger.info("Adding edge: $gene -> $phenotype")
	    def geneId = factory.getURI(gene)
	    def phenotypeId = factory.getURI(phenotype)
	    Edge e = new Edge(geneId, RDF.TYPE, phenotypeId)
	    graph.addE(e)
	}
    }
}

// new URL ("file:../data/MGI_GenePheno.rpt").getText().splitEachLine("\t") { line ->
  // def geneid = line[6]
  // def idUri = factory.getURI("http://phenomebrowser.net/ismb-tutorial/gene/"+geneid)
  // def pheno = line[4].replaceAll(":","_")
  // def phenoUri = factory.getURI("http://purl.obolibrary.org/obo/"+pheno)
  // Edge e = new Edge(idUri, RDF.TYPE, phenoUri)
  // graph.addE(e)
// }


def goConf = new GDataConf(GFormat.RDF_XML, Paths.get(rootDir, "upheno_all.owl").toString())
GraphLoaderGeneric.populate(goConf, graph)

def engine = new SM_Engine(graph)

def icConf = null

if (withAnnotations) {
    icConf = new IC_Conf_Corpus(icMeasureResolver(icMeasure))
}
else {
    icConf = new IC_Conf_Topo(icMeasureResolver(icMeasure))
}
    

def smConfPairwise = new SMconf(pairwiseMeasureResolver(pairwiseMeasure))
smConfPairwise.setICconf(icConf)
def smConfGroupwise = new SMconf(groupwiseMeasureResolver(groupwiseMeasure))

def mr = 0
def mrr = 0
def hitsK = [1: 0, 3: 0, 10: 0, 100: 0]
def ranks = [:]

// testPairs = testPairs.collect { [it[0], disease2pheno[it[1]]] }

//get 100 first pairs
// testPairs = testPairs[0..99]

logger.info("Computing Semantic Similarity for ${testPairs.size()} Gene-Disease pairs")
logger.info("Starting Pool ")
 
    def allRanks = GParsPool.withPool {
    testPairs.collectParallel { pair ->

	def test_gene = pair[0]
	def test_disease = pair[1]
	def disease_phenotypes = disease2pheno[test_disease].collect { factory.getURI(it) }.toSet()

	scores = evalGenes.collect { gene ->
	    def phenotypes = gene2pheno[gene]
	    def gene_phenotypes = phenotypes.collect { factory.getURI(it) }.toSet()
	    def sim_score = engine.compare(smConfGroupwise, smConfPairwise, gene_phenotypes, disease_phenotypes)
	    sim_score
	}
	def test_gene_index = evalGenes.indexOf(test_gene)
	
	[test_gene, test_disease, test_gene_index, scores]
	}

}


def out_file = rootDir + "/results_${icMeasure}_${pairwiseMeasure}_${groupwiseMeasure}.txt"
def out = new File(out_file)
out.withWriter { writer ->
	allRanks.each { r ->
	def gene = r[0]
	def disease = r[1]
	def gene_index = r[2]
	def scores = r[3]
	writer.write("${gene}\t${disease}\t${gene_index}\t${scores.join("\t")}\n")
	}
}


logger.info("Done")
// out.close()


logger.info("Results written to ${rootDir}/results.txt")



// logger.info("Pool finished. Analyzing results")


def axiomsToPairs(ontology, logger) {
    def pairs = []
    ontology.getAxioms().each { axiom ->
        if (axiom.getAxiomType() == AxiomType.SUBCLASS_OF) {
            def superclass = axiom.getSuperClass()
            if (superclass.getClassExpressionType() == ClassExpressionType.OBJECT_SOME_VALUES_FROM) {
                def subclass = axiom.getSubClass()
                def prop = superclass.getProperty()
                def filler = superclass.getFiller()
		
                def gene = subclass.toStringID()
                def disease = filler.toStringID()

                pairs.add([gene, disease])
            }
        }
    }

    logger.info("Total evaluation pairs: ${pairs.size()}")
    return pairs
}

static computeRankRoc(ranks, numEntities) {
    def nTails = numEntities

    def aucX = ranks.keySet().sort()
    def aucY = []
    def tpr = 0
    def sumRank = ranks.values().sum()
    aucX.each { x ->
        tpr += ranks[x]
        aucY.add(tpr / sumRank)
    }
    aucX.add(nTails)
    aucY.add(1)
    def auc = 0
        for (int i = 1; i < aucX.size(); i++) {
        auc += (aucX[i] - aucX[i-1]) * (aucY[i] + aucY[i-1]) / 2
    }
    return auc / nTails
}

static icMeasureResolver(measure) {
    if (measure.toLowerCase() == "sanchez") {
        return SMConstants.FLAG_ICI_SANCHEZ_2011
    } else if (measure.toLowerCase() == "resnik") {
	return SMConstants.FLAG_IC_ANNOT_RESNIK_1995_NORMALIZED

    } else {
        throw new IllegalArgumentException("Invalid IC measure: $measure")
    }
}

static pairwiseMeasureResolver(measure) {
    if (measure.toLowerCase() == "lin") {
        return SMConstants.FLAG_SIM_PAIRWISE_DAG_NODE_LIN_1998
    } else if (measure.toLowerCase() == "resnik") {
	return SMConstants.FLAG_SIM_PAIRWISE_DAG_NODE_RESNIK_1995
    } else {
        throw new IllegalArgumentException("Invalid pairwise measure: $measure")
    }
}

static groupwiseMeasureResolver(measure) {
    if (measure.toLowerCase() == "bma") {
        return SMConstants.FLAG_SIM_GROUPWISE_BMA
    } else {
        throw new IllegalArgumentException("Invalid groupwise measure: $measure")
    }
}

