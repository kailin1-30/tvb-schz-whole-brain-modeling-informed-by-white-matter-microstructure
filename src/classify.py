# classification using SGD
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from tools import load

PATH_DATA = "/Users/lin/Documents/Bachelor thesis /"
J_i, wFFI, wLRE, SC_all, region, region_labels = load(PATH_DATA)

# temp_results = results
# temp_labels = labels

metric = 'ADC'
AMPLITUDE = 0.01
SPLITS = 20
DURATION = 500 #in ms
SEED = 100
PATH = f"/Users/lin/Documents/Bachelor thesis /hallucinations/simulations/results/S_i_S_e/10subj_{DURATION}_{AMPLITUDE}_{SEED}"
PATH_LABEL = f"/Users/lin/Documents/Bachelor thesis /hallucinations/simulations/results/S_i_S_e/10subj_{DURATION}_{AMPLITUDE}_{SEED}_labels"
PATH_ACCURACY = f"/Users/lin/Documents/Bachelor thesis /hallucinations/simulations/results/S_i_S_e/all_{DURATION}_{AMPLITUDE}_{SEED}_accuracy"
SUBJECTS = SC_all[metric].keys()
# Load the results and labels
results = np.load(PATH, allow_pickle=True).item()
labels = np.load(PATH_LABEL)


subjects = results['ADC'].keys()
print(subjects)
# Prepare the data for classification
nsplits = SPLITS


accuracies_ctrl = []
accuracies_schz = []
accuracies_all  = []
FP_per_subj = {}
FP = []
FN_per_subj = {}
for subj in subjects:
    X = []
    y = []
    FP_per_subj[subj] = []
    FN_per_subj[subj] = []

    print(subj)
    for seed in range(SEED):
        if seed > (SEED // 2 - 1):
            X.append(np.array(results[metric][subj][seed][(AMPLITUDE, DURATION)]))
            y.append(1)
        else:
            X.append(np.array(results[metric][subj][seed][(0.0, DURATION)]))
            y.append(0)
    X = np.array(X)
    X = X.reshape(100, -1)

    for ii_split in range(nsplits):
                    ctr = 0
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=ii_split, stratify=y)
                    clf = make_pipeline(StandardScaler(),
                                SGDClassifier(loss = 'perceptron', penalty='l2', max_iter=5000, tol=5e-3))
                    clf.fit(X_train, y_train)

                    y_pred = clf.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred)
                    TN, FP, FN, TP = cm.ravel()
                    FP_per_subj[subj].append(FP)
                    FN_per_subj[subj].append(FN)

                    print(ii_split, f' Accuracy: {accuracy}')
                    accuracies_all.append(accuracy)

                    if 'schz' in subj:
                        accuracies_schz.append(accuracy)
                        print("Schz")
                    else:
                        accuracies_ctrl.append(accuracy)
                        print("Ctrl")
                    ctr+= 1

print(f"Overall accuracy:       {np.mean(accuracies_all):.7f} ± {np.std(accuracies_all):.3f}")
print(f"Control accuracy:       {np.mean(accuracies_ctrl):.7f} ± {np.std(accuracies_ctrl):.3f}")
print(f"Schizophrenia accuracy: {np.mean(accuracies_schz):.7f} ± {np.std(accuracies_schz):.3f}")

# save data
np.save(f"{PATH_ACCURACY}_all.npy", accuracies_all)
np.save(f"{PATH_ACCURACY}_ctrl.npy", accuracies_ctrl)
np.save(f"{PATH_ACCURACY}_schz.npy", accuracies_schz)
np.save(f"{PATH_ACCURACY}_FP.npy", FP_per_subj)
np.save(f"{PATH_ACCURACY}_FN.npy", FN_per_subj)
