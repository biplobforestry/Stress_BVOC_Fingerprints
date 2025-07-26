# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 22:49:07 2025
@author: b.dey
email: biplobforestry@gmail.com
project: Multi-stress interaction effects on BVOC emission fingerprints from oak and beech:
         A cross-investigation using Machine Learning and Positive Matrix Factorization 
"""

"""
This script implements a complete Random Forest classification pipeline to identify abiotic stress conditions 
(heat, ozone, and  ozone+heat) in Oak/beech trees based on VOC (volatile organic compound) emission data. 
It includes data preprocessing, model training with hyperparameter tuning, SHAP-based feature importance analysis, 
entropy uncertainty estimation, and multiple evaluation visualizations including confusion matrices and ROC curves.

Inputs:
- Preprocessed VOC data with stress labels
- Compound identity sheet with ion names
- Optional list of features to drop (e.g., highly correlated compounds to avoid redundant features)


Outputs:
- Classification metrics (F1, accuracy, etc.)
- SHAP global and class-wise plots
- Entropy distributions and time series
- ROC-AUC curves
- Exported Excel file with top SHAP features

Adapt the file paths before running. 
License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
"""
import os
import tkinter as tk
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import entropy
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    make_scorer, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import label_binarize
import io
from PIL import Image
from matplotlib_venn import venn3
from upsetplot import UpSet, from_contents
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from upsetplot import UpSet, from_contents
from itertools import combinations

import random
root = tk.Tk()
root.withdraw()
%matplotlib tk
mpl.rcParams['font.family'] = 'Arial'
#%%
identity = pd.read_excel("..../PMF_master_file.xlsx", sheet_name="Identity_mz")
ion_to_name = dict(zip(identity['ion'], identity['CompoundName']))
df = pd.read_excel("..../VOC_Data.xlsx", sheet_name='Oak')
drop_cols = pd.read_excel("..../VOC_Data.xlsx", sheet_name='Drop_oak')['Drop'].dropna().unique().tolist()
df.drop(columns=drop_cols, inplace=True)

df['UTCTime'] = pd.to_datetime(df['UTCTime'])
df.sort_values('UTCTime', inplace=True)
df.reset_index(drop=True, inplace=True)

le = LabelEncoder()
df['label'] = le.fit_transform(df['stress'])

#%%
train_df, test_df = [], []

for label in df['label'].unique():
    class_df = df[df['label'] == label].sort_values('UTCTime')
    total_test_target = int(len(class_df) * 0.30)

    split_idx = int(len(class_df) * 0.7)
    train_part = class_df.iloc[:split_idx]
    full_test_part = class_df.iloc[split_idx:]

    test_part = full_test_part.sample(n=min(total_test_target, len(full_test_part)))
    train_part = pd.concat([train_part, full_test_part.drop(test_part.index)])

    train_df.append(train_part)
    test_df.append(test_part)

train_df = pd.concat(train_df).sort_values('UTCTime').reset_index(drop=True)
test_df = pd.concat(test_df).sort_values('UTCTime').reset_index(drop=True)


drop_cols_common = ['UTCTime', 'stress', 'label']
X_train = train_df.drop(columns=drop_cols_common)
y_train = train_df['label']
X_test = test_df.drop(columns=drop_cols_common)
y_test = test_df['label']


selector = VarianceThreshold(threshold=1e-7)
X_train = selector.fit_transform(X_train)
X_test = selector.transform(X_test)

X_train = np.log1p(np.clip(X_train, a_min=0, a_max=None))
X_test = np.log1p(np.clip(X_test, a_min=0, a_max=None))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest with Grid Search 
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier( class_weight='balanced')
grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=2,
                    scoring=make_scorer(f1_score, average='weighted'))
grid.fit(X_train, y_train)
results = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")
results[['params', 'mean_test_score', 'rank_test_score']].to_excel('......./rf_hyperparameter_results_oak.xlsx', index=False)

print("Best Parameters:", grid.best_params_)
print("Best F1 Score:", grid.best_score_)

final_model = RandomForestClassifier(class_weight='balanced', **best_params)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(7, 5))
disp.plot(cmap="PRGn", values_format='d', ax=ax)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        value = cm[i, j]
        text_obj = disp.text_[i, j]
        norm_value = value / cm.max()
        text_color = "white" if norm_value < 0.4 else "black"
        text_obj.set_fontsize(14)
        text_obj.set_fontweight("bold")
        text_obj.set_color(text_color)

ax.set_title("Confusion Matrix (Oak)", fontsize=14, fontweight='bold')
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_xlabel("")
ax.set_ylabel("")
cbar = fig.axes[-1]
cbar.tick_params(labelsize=14)
plt.tight_layout()
plt.show()

y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_proba = final_model.predict_proba(X_test)
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{le.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title("Multiclass ROC-AUC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()


report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report_dict).T.loc[le.classes_, ['precision', 'recall', 'f1-score']].round(2)
supports = pd.DataFrame(report_dict).T.loc[le.classes_, 'support'].astype(int)
report_df.index = [f"{cls} ({n})" for cls, n in zip(le.classes_, supports)]

plt.figure(figsize=(7.48, 3.94))
sns.heatmap(report_df, annot=True, cmap='RdBu_r', vmin=0, vmax=1,annot_kws={"size": 15})
plt.title("(b)",fontsize=17,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=16)  
plt.tight_layout()
plt.show()
#%%
#SHAP Feature extraction
feature_names = selector.get_feature_names_out()
feature_names_with_compound = [
    f"{ion} ({ion_to_name.get(ion, 'Unknown')})" for ion in feature_names
]

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)  
shap_values = np.array(shap_values)


global_shap_matrix = np.mean(np.abs(shap_values), axis=2)

shap.summary_plot(
    global_shap_matrix,
    features=X_test,
    feature_names=feature_names_with_compound,
    plot_type='dot',
    max_display=20,
    show=False  # Prevent immediate show so we can edit
)

plt.gcf().set_size_inches((8.48, 7))
plt.title("Global SHAP Feature Importance", fontsize=15, fontweight='bold')
plt.xlabel("Mean SHAP value", fontsize=15)
#plt.ylabel("VOC Compounds", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
cbar = plt.gcf().axes[-1]  
cbar.tick_params(labelsize=15)  
cbar.set_ylabel("Feature value", fontsize=15)  
plt.tight_layout()
plt.tight_layout()
plt.savefig(r".....\shap global_Oak.png", dpi=500, bbox_inches='tight')
plt.savefig(r"...\shap global_Oak.svg", dpi=500, bbox_inches='tight')
plt.show()

# CLASS-WISE SHAP SUMMARY PLOTS
class_names = le.classes_

for i, class_name in enumerate(class_names):
    plt.figure(figsize=(7.48, 5.12))
    shap.summary_plot(
        shap_values[:, :, i],  
        features=X_test,
        feature_names=feature_names_with_compound,
        plot_type='dot',
        max_display=15,
        show=False
    )
    fig = plt.gcf()
    if len(fig.axes) > 1:
        fig.delaxes(fig.axes[-1]) 
    plt.title(f"{class_name}", fontsize=17,fontweight='bold')
    #plt.xlabel("Mean SHAP value", fontsize=15)
    #plt.ylabel("VOC Compounds", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(r'......\Figures\ML', f"shap_summary_Oak_{class_name}.png"), dpi=500, bbox_inches='tight',pad_inches=0)
    plt.show()
#%%
import re
def extract_ion_name(label):
    return re.match(r'^[^\s(]+', label).group(0)
top_features_dict = {}
stress_classes = [c for c in le.classes_ if c != 'Pre-stress']
for i, class_name in enumerate(le.classes_):
    if class_name == 'Pre-stress':
        continue  

    shap_vals = shap_values[:, :, i]
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    top_idx = np.argsort(mean_abs_shap)[-15:][::-1]
    ions = [extract_ion_name(feature_names_with_compound[j]) for j in top_idx]
    importances = mean_abs_shap[top_idx]

    top_features_dict[class_name] = pd.DataFrame({
        'Feature': ions,
        'Importance': importances
    })
df_combined = pd.concat(
    [df.set_index('Feature')['Importance'] for df in top_features_dict.values()],
    axis=1,
    keys=top_features_dict.keys()
).fillna(0).reset_index()

with pd.ExcelWriter(r'.......\shap_top15_stress_Oak.xlsx') as writer:
    df_combined.to_excel(writer, sheet_name='Combined_Stress_Top15', index=False)
    for class_name, df in top_features_dict.items():
        sheet_name = class_name.replace(' ', '_') + '_Top15'
        df.to_excel(writer, sheet_name=sheet_name, index=False)
#%%


probs = final_model.predict_proba(X_test)
entropy_scores = entropy(probs, axis=1)  

plt.figure(figsize=(7.48, 3.94))
sns.histplot(entropy_scores, bins=30, kde=True)
plt.title("(d)",fontsize=17, fontweight='bold')
plt.xlabel("Entropy",fontsize=14)
plt.ylabel("Number of samples",fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()
#%%
correct_mask = y_pred == y_test.to_numpy()
plt.figure(figsize=(7.48, 3.94))

sns.histplot(entropy_scores[correct_mask], color='green', label='Correct', kde=True, stat='density', bins=30)
sns.histplot(entropy_scores[~correct_mask], color='red', label='Incorrect', kde=True, stat='density', bins=30)

plt.title("(f)", fontsize=17, fontweight='bold')
plt.xlabel("Entropy", fontsize=16)
plt.ylabel("Density", fontsize=16)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(fontsize=16)
plt.tight_layout()
#plt.savefig(r"..........\entropy_distribution_Oak.png", dpi=500)
#plt.savefig(r"..........\entropy_distribution_Oak.svg", dpi=500)
plt.show()
#%%
test_df = test_df.copy()
test_df['Entropy'] = entropy_scores

plt.figure(figsize=(7.48, 3.94))
plt.plot(test_df['UTCTime'], test_df['Entropy'], marker='o', linestyle='-', alpha=0.7)
plt.title("Prediction Entropy Over Time:Oak")
plt.xlabel("Time")
plt.ylabel("Entropy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#%%

test_df = test_df.copy()
test_df['Entropy'] = entropy_scores
plt.figure(figsize=(7.48, 3.94))

for stress_class in test_df['stress'].unique():
    subset = test_df[test_df['stress'] == stress_class]
    plt.plot(subset['UTCTime'], subset['Entropy'], marker='o', linestyle='-', label=stress_class, alpha=0.8)

plt.title("Entropy over time by stress class:Oak", fontsize=14, fontweight='bold')
plt.xlabel("Date", fontsize=14)
plt.ylabel("Entropy", fontsize=14)

plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=14)
plt.legend(fontsize=12, ncol=3, loc='upper left', frameon=False)
plt.tight_layout()
#plt.savefig(r".......\entropy_timeseries_Oak.png", dpi=500)
#plt.savefig(r"................\entropy_timeseries_Oak.svg", dpi=500)
plt.show()
#%%
from sklearn.metrics import accuracy_score, f1_score

n_bootstrap = 1000
boot_acc = []
boot_f1 = []

for _ in tqdm(range(n_bootstrap)):
    indices = np.random.choice(len(y_test), size=len(y_test), replace=True)
    y_test_boot = y_test.iloc[indices]
    y_pred_boot = y_pred[indices]

    boot_acc.append(accuracy_score(y_test_boot, y_pred_boot))
    boot_f1.append(f1_score(y_test_boot, y_pred_boot, average='weighted'))

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 5))

sns.histplot(boot_acc, kde=True, bins=30, ax=axes[0], color='steelblue')
axes[0].set_title("Bootstrapped accuracy :Oak ", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Accuracy", fontsize=14)
axes[0].set_ylabel("Frequency", fontsize=14)
axes[0].tick_params(axis='x', labelsize=14)
axes[0].tick_params(axis='y', labelsize=14)

sns.histplot(boot_f1, kde=True, bins=30, ax=axes[1], color='orange')
axes[1].set_title("Bootstrapped F1-score :Oak", fontsize=14, fontweight='bold')
axes[1].set_xlabel("F1 Score", fontsize=14)
axes[1].set_ylabel("Frequency", fontsize=14)
axes[1].tick_params(axis='x', labelsize=14)
axes[1].tick_params(axis='y', labelsize=14)

plt.tight_layout()
#plt.savefig(r"..........\bootstrap_matric_Oak.png", dpi=500)
#plt.savefig(r"..........\bootstrap_matric_Oak.svg", dpi=500)
plt.show()