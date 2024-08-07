import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class roc_auc:
    def __init__(self, y_true, y_scores):
        self.y_true = y_true
        self.y_scores = y_scores

        # 计算ROC曲线

        # 计算正样本数
        positive_cnt = sum(self.y_true)
        # 计算负样本数
        negative_cnt = len(self.y_true) - positive_cnt

        # 进行排序
        sorted_index = np.argsort(self.y_scores)[::-1]
        y_true_sorted = np.array(self.y_true)[sorted_index]
        y_scores_sorted = np.array(self.y_scores)[sorted_index]

        # 计算TPR和FPR
        self.tpr = []
        self.fpr = []

        tp = 0
        fp = 0

        for i in range(len(y_scores_sorted)):
            if y_true_sorted[i] == 1:
                tp += 1
            
            else:
                fp += 1

            self.tpr.append(tp / positive_cnt)
            self.fpr.append(fp / negative_cnt)
            
    
    def get_roc(self):
        
        
        plt.figure()
        plt.plot(self.fpr, self.tpr, marker='.', label='ROC curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

    def get_auc(self):

        auc = np.trapz(self.tpr, self.fpr)
        return auc

        