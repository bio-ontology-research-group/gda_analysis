from mowl.projection import TaxonomyProjector, TaxonomyWithRelationsProjector, Edge
from mowl.utils.data import FastTensorDataLoader
import torch as th
from tqdm import tqdm
import numpy as np
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.model import AxiomType as Ax
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class Evaluator:
    def __init__(self, dataset, device, batch_size=64, evaluate_with_deductive_closure=False, filter_deductive_closure=False):

        if evaluate_with_deductive_closure and filter_deductive_closure:
            raise ValueError("Cannot evaluate with deductive closure and filter it at the same time. Set either evaluate_with_deductive_closure or filter_deductive_closure to False.")

        logger.info(f"Evaluating with deductive closure: {evaluate_with_deductive_closure}")
        logger.info(f"Filtering deductive closure: {filter_deductive_closure}")
        
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.train_tuples = self.create_tuples(dataset.ontology)
        self.valid_tuples = self.create_tuples(dataset.validation)
        self.test_tuples = self.create_tuples(dataset.testing)
        self._deductive_closure_tuples = None

        self.evaluate_with_deductive_closure = evaluate_with_deductive_closure
        self.filter_deductive_closure = filter_deductive_closure
        
        self.class_to_id = {c: i for i, c in enumerate(self.dataset.classes.as_str)}
        self.id_to_class = {i: c for c, i in self.class_to_id.items()}

        self.relation_to_id = {r: i for i, r in enumerate(self.dataset.object_properties.as_str)}
        self.id_to_relation = {i: r for r, i in self.relation_to_id.items()}

        eval_heads, eval_tails = self.dataset.evaluation_classes
        self.class_id_to_head_id = {self.class_to_id[c]: i for i, c in enumerate(eval_heads.as_str)}
        self.class_id_to_tail_id = {self.class_to_id[c]: i for i, c in enumerate(eval_tails.as_str)}
                
        print(f"Number of evaluation classes: {len(eval_heads)} and {len(eval_tails)}")
        self.evaluation_heads = th.tensor([self.class_to_id[c] for c in eval_heads.as_str], dtype=th.long, device=device)
        self.evaluation_tails = th.tensor([self.class_to_id[c] for c in eval_tails.as_str], dtype=th.long, device=device)


    @property
    def deductive_closure_tuples(self):
        if self._deductive_closure_tuples is None:
            self._deductive_closure_tuples = self.create_tuples(self.dataset.deductive_closure_ontology)
        return self._deductive_closure_tuples
        
    def create_tuples(self, ontology):
        raise NotImplementedError

    def get_logits(self, batch):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        model = args[0]
        mode = kwargs.get("mode")
        
        if mode == "valid":
            eval_tuples = self.valid_tuples
        else:
            eval_tuples = self.test_tuples

        return self.evaluate_base(model, eval_tuples, **kwargs)

    
    def evaluate_base(self, model, eval_tuples, mode="test", **kwargs):
        num_heads, num_tails = len(self.evaluation_heads), len(self.evaluation_tails)
        model.eval()
        if not mode in ["valid", "test"]:
            raise ValueError(f"Mode must be either 'valid' or 'test', not {mode}")


        if self.evaluate_with_deductive_closure:
            mask1 = (self.deductive_closure_tuples.unsqueeze(1) == self.train_tuples).all(dim=-1).any(dim=-1)
            mask2 = (self.deductive_closure_tuples.unsqueeze(1) == self.valid_tuples).all(dim=-1).any(dim=-1)
            mask = mask1 | mask2
            deductive_closure_tuples = self.deductive_closure_tuples[~mask]
            
            eval_tuples = th.cat([eval_tuples, deductive_closure_tuples], dim=0)
            # eval_tuples = deductive_closure_tuples
        dataloader = FastTensorDataLoader(eval_tuples, batch_size=self.batch_size, shuffle=False)

        macro_metrics = dict()
        mrr, fmrr = 0, 0
        mr, fmr = 0, 0
        ranks, franks = dict(), dict()

        micro_metrics = dict()
        micro_mr, micro_fmr = dict(), dict()
        micro_mrr, micro_fmrr = dict(), dict()
        micro_hits_k, micro_f_hits_k = dict(), dict()
        micro_ranks, micro_franks = dict(), dict()
        num_results_per_tail = dict()

        
        if mode == "test":
            hits_k = dict({1: 0, 3: 0, 10: 0, 50: 0, 100: 0})
            f_hits_k = dict({1: 0, 3: 0, 10: 0, 50: 0, 100: 0})
        
            filtering_labels = self.get_filtering_labels(num_heads, num_tails, self.class_id_to_head_id, self.class_id_to_tail_id, **kwargs)
            if self.evaluate_with_deductive_closure:
                deductive_labels = self.get_deductive_labels(num_heads, num_tails, **kwargs)
            
        with th.no_grad():
            for batch, in dataloader:
                if batch.shape[1] == 2:
                    heads, tails = batch[:, 0], batch[:, 1]
                elif batch.shape[1] == 3:
                    heads, tails = batch[:, 0], batch[:, 2]
                else:
                    raise ValueError("Batch shape must be either (n, 2) or (n, 3)")
                aux_heads = heads.clone()
                aux_tails = tails.clone()
        
                batch = batch.to(self.device)
                logits_heads, logits_tails = self.get_logits(model, batch)

                if not logits_heads is None:
                    raise NotImplementedError("Logits heads must be None for this task")

                for i, tail in enumerate(aux_tails):
                    head = aux_heads[i]
                    head = th.where(self.evaluation_heads == head)[0].item()
                    tail = th.where(self.evaluation_tails == tail)[0].item()
                    preds = logits_tails[i]

                    if self.evaluate_with_deductive_closure:
                        ded_labels = deductive_labels[:, tail].to(preds.device)
                        ded_labels[head] = 1
                        preds = preds * ded_labels
                    
                    order = th.argsort(preds, descending=False)
                    # rank = order[head].item() + 1
                    rank = th.where(order == head)[0].item() + 1
                    mr += rank
                    mrr += 1 / rank

                    if not tail in micro_mr:
                        micro_mr[tail] = 0
                        micro_mrr[tail] = 0
                        micro_hits_k[tail] = {1: 0, 3: 0, 10: 0, 50: 0, 100: 0}
                        micro_ranks[tail] = dict()
                        num_results_per_tail[tail] = 0

                    num_results_per_tail[tail] += 1
                    micro_mr[tail] += rank
                    micro_mrr[tail] += 1 / rank
                    for k in micro_hits_k[tail]:
                        if rank <= int(k):
                            micro_hits_k[tail][k] += 1

                    if rank not in micro_ranks[tail]:
                        micro_ranks[tail][rank] = 0
                    micro_ranks[tail][rank] += 1
                        

                        
                    if mode == "test":
                        
                        # tail_id = self.class_id_to_tail_id[tail]
                        f_preds = preds * filtering_labels[:, tail].to(preds.device)

                        if self.evaluate_with_deductive_closure:
                            ded_labels = deductive_labels[:, tail].to(preds.device)
                            ded_labels[head] = 1
                            f_preds = f_preds * ded_labels

                        
                        f_order = th.argsort(f_preds, descending=False)
                        f_rank = th.where(f_order == head)[0].item() + 1
                        fmr += f_rank
                        fmrr += 1 / f_rank

                        if not tail in micro_fmr:
                            micro_fmr[tail] = 0
                            micro_fmrr[tail] = 0
                            micro_f_hits_k[tail] = {1: 0, 3: 0, 10: 0, 50: 0, 100: 0}
                            micro_franks[tail] = dict()

                        micro_fmr[tail] += f_rank
                        micro_fmrr[tail] += 1 / f_rank
                        for k in micro_f_hits_k[tail]:
                            if f_rank <= int(k):
                                micro_f_hits_k[tail][k] += 1

                        if f_rank not in micro_franks[tail]:
                            micro_franks[tail][f_rank] = 0
                        micro_franks[tail][f_rank] += 1
                        

                        for k in hits_k:
                            if rank <= int(k):
                                hits_k[k] += 1

                        for k in f_hits_k:
                            if f_rank <= int(k):
                                f_hits_k[k] += 1

                        if rank not in ranks:
                            ranks[rank] = 0
                        ranks[rank] += 1
                                
                        if f_rank not in franks:
                            franks[f_rank] = 0
                        franks[f_rank] += 1

            divisor = 2
            if logits_heads is None:
                divisor -= 1
            if logits_tails is None:
                divisor -= 1

            assert divisor > 0, "Both logits_heads and logits_tails cannot be None"

            divisor  = divisor * len(eval_tuples)
            
            mr = mr / divisor
            mrr = mrr / divisor
            micro_mr = {k: v / num_results_per_tail[k] for k, v in micro_mr.items()}
            micro_mrr = {k: v / num_results_per_tail[k] for k, v in micro_mrr.items()}

            macro_metrics["mr"] = mr
            macro_metrics["mrr"] = mrr
            mean_micro_mr = np.mean(list(micro_mr.values()))
            mean_micro_mrr = np.mean(list(micro_mrr.values()))
            
            
            if mode == "test":
                fmr = fmr / divisor
                fmrr = fmrr / divisor
                auc = compute_rank_roc(ranks, num_heads)
                f_auc = compute_rank_roc(franks, num_heads)

                macro_metrics["f_mr"] = fmr
                macro_metrics["f_mrr"] = fmrr
                macro_metrics["auc"] = auc
                macro_metrics["f_auc"] = f_auc
                
                for k in hits_k:
                    hits_k[k] = hits_k[k] / divisor
                    macro_metrics[f"hits@{k}"] = hits_k[k]
                    
                for k in f_hits_k:
                    f_hits_k[k] = f_hits_k[k] / divisor
                    macro_metrics[f"f_hits@{k}"] = f_hits_k[k]

                

                micro_auc = {k: compute_rank_roc(micro_ranks[k], num_heads) for k in micro_ranks}
                for k in micro_hits_k:
                    micro_hits_k[k] = {k2: v / num_results_per_tail[k] for k2, v in micro_hits_k[k].items()}

                mean_micro_auc = np.mean(list(micro_auc.values()))
                mean_micro_hits_k = dict()
                for k in hits_k:
                    mean_micro_hits_k[k] = np.mean([x[k] for x in micro_hits_k.values()])


            micro_fmr = {k: v / num_results_per_tail[k] for k, v in micro_fmr.items()}
            micro_fmrr = {k: v / num_results_per_tail[k] for k, v in micro_fmrr.items()}
            micro_metrics["mr"] = mean_micro_mr
            micro_metrics["mrr"] = mean_micro_mrr

            if mode == "test":
                micro_f_auc = {k: compute_rank_roc(micro_franks[k], num_heads) for k in micro_franks}
                for k in micro_f_hits_k:
                    micro_f_hits_k[k] = {k2: v / num_results_per_tail[k] for k2, v in micro_f_hits_k[k].items()}

                mean_micro_fmr = np.mean(list(micro_fmr.values()))
                mean_micro_fmrr = np.mean(list(micro_fmrr.values()))
                mean_micro_f_auc = np.mean(list(micro_f_auc.values()))
                mean_micro_f_hits_k = dict()
                for k in hits_k:
                    mean_micro_f_hits_k[k] = np.mean([x[k] for x in micro_f_hits_k.values()])
                

                micro_metrics["mr"] = mean_micro_mr
                micro_metrics["mrr"] = mean_micro_mrr
                micro_metrics["auc"] = mean_micro_auc
                for k in mean_micro_hits_k:
                    micro_metrics[f"hits@{k}"] = mean_micro_hits_k[k]

                micro_metrics["f_mr"] = mean_micro_fmr
                micro_metrics["f_mrr"] = mean_micro_fmrr
                micro_metrics["f_auc"] = mean_micro_f_auc
                for k in mean_micro_f_hits_k:
                    micro_metrics[f"f_hits@{k}"] = mean_micro_f_hits_k[k]

            micro_metrics = {f"{mode}_{k}": v for k, v in micro_metrics.items()}
            macro_metrics = {f"{mode}_{k}": v for k, v in macro_metrics.items()}
            
            return micro_metrics, macro_metrics

        
class GDAEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_tuples(self, ontology):
        projector = TaxonomyWithRelationsProjector(relations=["http://mowl.borg/associated_with"])
        edges = projector.project(ontology)

        classes, relations = Edge.get_entities_and_relations(edges)

        class_str2owl = self.dataset.classes.to_dict()
        class_owl2idx = self.dataset.classes.to_index_dict()
        relation_str2owl = self.dataset.object_properties.to_dict()
        relation_owl2idx = self.dataset.object_properties.to_index_dict()
        
        edges_indexed = []
        
        for e in edges:
            head = class_owl2idx[class_str2owl[e.src]]
            relation = relation_owl2idx[relation_str2owl[e.rel]]
            tail = class_owl2idx[class_str2owl[e.dst]]
            edges_indexed.append((head, relation, tail))
        
        return th.tensor(edges_indexed, dtype=th.long)

    def get_logits(self, model, batch):
        heads, rels, tails = batch[:, 0], batch[:, 1], batch[:, 2]
        num_heads, num_tails = len(heads), len(tails)

        rels = rels.repeat_interleave(len(self.evaluation_heads)).unsqueeze(1)
        tails = tails.repeat_interleave(len(self.evaluation_heads)).unsqueeze(1)
        eval_heads = self.evaluation_heads.repeat(num_tails).unsqueeze(1)
        assert rels.shape == tails.shape == eval_heads.shape, f"{rels.shape} != {tails.shape} != {eval_heads.shape}"
        # eval_heads = th.arange(len(self.evaluation_heads), device=tails.device).repeat(num_tails).unsqueeze(1)

        logits_tails = model(th.cat([eval_heads, rels, tails], dim=-1), "gci2")
        logits_tails = logits_tails.view(-1, len(self.evaluation_heads))
        # print(logits_heads, logits_tails)
        return None, logits_tails

    
    def get_filtering_labels(self, num_heads, num_tails, class_id_to_head_id, class_id_to_tail_id, **kwargs):
        filtering_tuples = th.cat([self.train_tuples, self.valid_tuples], dim=0)
        filtering_labels = th.ones((num_heads, num_tails), dtype=th.float)

        for head, rel, tail in filtering_tuples:
            head = class_id_to_head_id[head.item()]
            tail = class_id_to_tail_id[tail.item()]
            filtering_labels[head, tail] = 10000
        
        return filtering_labels
    

def compute_rank_roc(ranks, num_entities):
    n_tails = num_entities
                    
    auc_x = list(ranks.keys())
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(n_tails)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x) / n_tails
    return auc
