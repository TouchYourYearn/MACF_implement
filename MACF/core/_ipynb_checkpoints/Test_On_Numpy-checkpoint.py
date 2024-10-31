import numpy as np
from sklearn.metrics import accuracy_score, f1_score
a = np.array([5,-12,2])
b = np.clip(a,-3,3)
print(b)

a = np.array([1,1,2,3])
b = np.array([0.9,0.8,2,2.9])
c = np.corrcoef(a,b)
print(c)

a = np.array([1.2,1.6,1.8,2.1,0.9])
b = np.array([1,1,2,2,1])
print(np.sum(np.round(a) == np.round(b))/float(len(a)))

label = np.array([-1,-2,-2.5,0,0,1.2,3])
pre = np.array([-0.9,-1.8,-1.9,0.2,0.2,1.2,2.9])
non_zeros = np.array([i for i, e in enumerate(label) if e != 0])
non_zeros_binary_truth = (label[non_zeros] > 0)
non_zeros_binary_preds = (pre[non_zeros] > 0)
print(non_zeros_binary_truth)
print(non_zeros_binary_preds)

non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
print(non_zeros_acc2)
non_zeros_f1_score = f1_score(non_zeros_binary_preds, non_zeros_binary_truth, average='weighted')
print(non_zeros_f1_score)

binary_truth = (label >= 0)
print(binary_truth)
binary_preds = (pre >= 0 )
print(binary_preds)
