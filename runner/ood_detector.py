import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

from runner.predictor import Predictor


class OodDetector(Predictor):
    def __init__(self, args, loader_in, loader_out, model):
        super().__init__(args, loader_in, model)
        self.loader_out = loader_out
        self.loader_in = loader_in

    def infer(self, is_gbs, is_odin, with_acc=False, seed=0):
        self.loader = self.loader_in
        output_id, _ = super().infer(is_gbs, is_odin, with_acc, seed)
        self.loader = self.loader_out
        output_od, _ = super().infer(is_gbs, is_odin, with_acc=False, seed=seed)
        self.output_id = output_id
        self.output_od = output_od
        return output_id, output_od

    def auroc(self, temp):
        label = np.r_[np.zeros([self.output_id.shape[-2]]),
                      np.ones([self.output_od.shape[-2]])]
        mean_id = self.predictive_mean(self.output_id, temp)
        mean_od = self.predictive_mean(self.output_od, temp)
        mean = np.r_[mean_id, mean_od]
        tpr, fpr, ths = roc_curve(label, mean.max(1))
        auc_ = auc(fpr, tpr)
        return max(auc_, 1 - auc_)
