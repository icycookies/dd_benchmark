import logging
import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

logger = logging.getLogger(__name__)
def train_svm(train_dataset, eval_dataset, test_dataset, args):
    logger.info("training SVM classifier")
    feats = train_dataset[:][0].numpy()
    preds = train_dataset[:][8].numpy()
    model = svm.SVC()
    model.fit(feats, preds)
    feats = test_dataset[:][0].numpy()
    labels = test_dataset[:][8].numpy()
    preds = model.predict(feats)
    f1 = f1_score(preds, labels, average="micro")
    print("Micro F1 = %s" % (str(f1)))
    if "precision" in args.eval_metric:
        print("Precision = %s", precision_score(labels, preds))
    if "recall" in args.eval_metric:
        print("Recall = %s", recall_score(labels, preds))
    if "accuracy" in args.eval_metric:
        print("Accuracy = %s", accuracy_score(labels, preds))
    