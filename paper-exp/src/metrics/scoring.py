import numpy as np


class Metrics:
    def __init__(self, classes):
        self.total = np.zeros(classes, dtype=np.int64)
        self.correct = np.zeros_like(self.total)
        self.bins = np.zeros(15)
        self.confidence = np.zeros(15)
        self.bin_correct = np.zeros(15)

    def add(self, logits, labels):
        p, y = logits.detach().softmax(1).max(1)
        p, y, labels = p.cpu().numpy(), y.cpu().numpy(), labels.numpy()
        ok = y == labels
        ids = np.minimum((p*15).astype(int), 14)
        self.total += np.bincount(labels, minlength=len(self.total))
        self.correct += np.bincount(labels[ok], minlength=len(self.total))
        self.bins += np.bincount(ids, minlength=15)
        self.confidence += np.bincount(ids, weights=p, minlength=15)
        self.bin_correct += np.bincount(ids, weights=ok.astype(float), minlength=15)

    def result(self):
        n = int(self.total.sum())
        mask = self.total > 0
        return dict(samples=n, top1=float(100*self.correct.sum()/n),
                    macro_accuracy=float(100*np.mean(self.correct[mask]/self.total[mask])),
                    ece=float(100*np.abs(self.confidence-self.bin_correct).sum()/n),
                    observed_classes=int(mask.sum()))
