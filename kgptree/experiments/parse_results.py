import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score


vectors = []
for file in os.listdir('output'):
	if not file.endswith('.p'): continue
	dataset = file.split('_')[0]
	results = pickle.load(open('output/{}'.format(file), 'rb'))

	ground_truth = results['ground_truth']

	tree_time = results['tree_fit_time']
	tree_preds = results['tree_preds']
	tree_acc = accuracy_score(ground_truth, tree_preds)

	forest_time = results['forest_fit_time']
	forest_preds = results['forest_preds']
	forest_acc = accuracy_score(ground_truth, forest_preds)

	transf_time = results['transform_fit_time']
	transf_lr_preds = results['transform_lr_preds']
	transf_lr_acc = accuracy_score(ground_truth, transf_lr_preds)
	transf_rf_preds = results['transform_rf_preds']
	transf_rf_acc = accuracy_score(ground_truth, transf_rf_preds)

	vectors.append([dataset, tree_time, tree_acc, forest_time, forest_acc, 
				    transf_time, transf_lr_acc, transf_rf_acc])

df = pd.DataFrame(vectors, columns=['Dataset', 'Tree Time', 'Tree Accuracy',
									'Forest Time', 'Forest Accuracy', 'Transform Time',
									'Transform + LR Accuracy', 'Transform + RF Accuracy'])
print(df.groupby('Dataset').agg(['mean', 'std']))
