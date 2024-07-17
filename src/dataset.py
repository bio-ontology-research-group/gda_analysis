from mowl.datasets import PathDataset, OWLClasses
import os

class GDADatasetEL(PathDataset):

    def __init__(self, root_dir):
        root = os.path.abspath(root_dir)
        train_path = os.path.join(root, "train_normalized.owl")
        valid_path = os.path.join(root, "valid.owl")
        test_path = os.path.join(root, "test.owl")

        super(GDADatasetEL, self).__init__(train_path, valid_path, test_path)

    @property
    def evaluation_classes(self):
        
        if self._evaluation_classes is None:
            genes = set()
            diseases = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                
                if "mowl.borg" in owl_name and owl_name.split("/")[-1].isnumeric():
                    genes.add(owl_cls)
                if "OMIM_" in owl_name:
                    diseases.add(owl_cls)

            genes = OWLClasses(genes)
            diseases = OWLClasses(diseases)
            self._evaluation_classes = (genes, diseases)

        return self._evaluation_classes

    @property
    def evaluation_object_property(self):
        return "http://mowl.borg/associated_with"

class GDADataset(PathDataset):

    def __init__(self, root_dir):
        root = os.path.abspath(root_dir)
        train_path = os.path.join(root, "train.owl")
        valid_path = os.path.join(root, "valid.owl")
        test_path = os.path.join(root, "test.owl")

        super(GDADataset, self).__init__(train_path, valid_path, test_path)

    @property
    def evaluation_classes(self):
        
        if self._evaluation_classes is None:
            genes = set()
            diseases = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                
                if "mowl.borg" in owl_name and owl_name.split("/")[-1].isnumeric():
                    genes.add(owl_cls)
                if "OMIM_" in owl_name:
                    diseases.add(owl_cls)

            genes = OWLClasses(genes)
            diseases = OWLClasses(diseases)
            self._evaluation_classes = (genes, diseases)

        return self._evaluation_classes

    @property
    def evaluation_object_property(self):
        return "http://mowl.borg/associated_with"

    
class GDADatasetReasoned(PathDataset):

    def __init__(self, root_dir):
        root = os.path.abspath(root_dir)
        train_path = os.path.join(root, "train_reasoned.owl")
        valid_path = os.path.join(root, "valid.owl")
        test_path = os.path.join(root, "test.owl")

        super(GDADatasetReasoned, self).__init__(train_path, valid_path, test_path)

    @property
    def evaluation_classes(self):
        
        if self._evaluation_classes is None:
            genes = set()
            diseases = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                
                if "mowl.borg" in owl_name and owl_name.split("/")[-1].isnumeric():
                    genes.add(owl_cls)
                if "OMIM_" in owl_name:
                    diseases.add(owl_cls)

            genes = OWLClasses(genes)
            diseases = OWLClasses(diseases)
            self._evaluation_classes = (genes, diseases)

        return self._evaluation_classes

    @property
    def evaluation_object_property(self):
        return "http://mowl.borg/associated_with"

