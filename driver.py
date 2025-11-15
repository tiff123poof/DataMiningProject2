from ucimlrepo import fetch_ucirepo 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA



# ------------------------- Part A ------------------------- #

def fetch_dataset():
    wine_quality = fetch_ucirepo(id=186) 
    x = wine_quality.data.features 
    x['color'] = ['red'] * 1599 + ['white'] * 4898 
    y = wine_quality.data.targets 
    return x, y


def plot_distribution(df, feature, bins = 20):
    plt.figure(figsize=(6,4))
    sns.histplot(df[feature], bins=bins, kde=True, edgecolor="black")
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.savefig(f"plots/{feature}_distribution.png", dpi=300)
    plt.close()


def plot_color(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x='color', data=df, palette=['#B22222', '#F5DEB3'], edgecolor='black')
    plt.title('Distribution of Wine Color')
    plt.xlabel('Color')
    plt.ylabel('Count')
    plt.savefig('plots/color_bar_chart.png', dpi=300)
    plt.close()

X, Y = fetch_dataset()
numeric_features = X.select_dtypes(include=["float64"]).columns
for feature in numeric_features:
    plot_distribution(X, feature)
plot_color(X)

DATA_MULTI = pd.concat([X, Y], axis=1)
T = 6   # threshold
Y['quality_binary'] = [1 if q >= T else 0 for q in Y['quality']]
DATA_BINARY = pd.concat([X, Y.drop(columns=['quality'])], axis=1)

DATA_BINARY.to_csv("wine_quality_binary.csv", index=False)
DATA_MULTI.to_csv("wine_quality_multi.csv", index=False)

# ------------------------- Part B ------------------------- #

def cv_SVM(X, y, thresholds, kernels, k=5):
    hp_values = [(t, kernel) for t in thresholds for kernel in kernels]
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    subsets = list(kf.split(X))
    accuracy_fold = []
    for train_idx, test_idx in subsets:
        accuracy_hp =[]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        for (t, kernel) in hp_values:
            y_tr = (y_train >= t).astype(int)
            y_te = (y_test >= t).astype(int)

            means = X_train.mean(axis=0)
            stds = X_train.std(axis=0)
            X_tr = (X_train - means) /stds
            X_te = (X_test - means) /stds

            model = SVC(kernel=kernel, random_state=42)
            model.fit(X_tr, y_tr)

            preds = model.predict(X_te)
            test_accuracy = np.mean(preds == y_te)
            accuracy_hp.append(test_accuracy)
    
        accuracy_fold.append(accuracy_hp)

    avg_scores = np.mean(np.array(accuracy_fold), axis=0)
    return hp_values, avg_scores


def cv_RF(X, y, hp_values, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    subsets = list(kf.split(X))
    accuracy_fold = []
    for train_idx, test_idx in subsets:
        accuracy_hp =[]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        for hp in hp_values:
            model = RandomForestClassifier(n_estimators=hp, random_state=42)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            test_accuracy = np.mean(preds == y_test)
            accuracy_hp.append(test_accuracy)
    
        accuracy_fold.append(accuracy_hp)

    avg_scores = np.mean(np.array(accuracy_fold), axis=0)
    return hp_values, avg_scores


def visualize_hp_SVM(hp_values, avg_scores, title, filename):
    hp_values = np.array(hp_values)
    avg_scores = np.array(avg_scores)

    thresholds = np.unique(hp_values[:,0])
    kernels = np.unique(hp_values[:,1])

    plt.figure(figsize=(8,6))
    
    x_positions = np.arange(len(kernels))

    for i, t in enumerate(thresholds):
        idx = hp_values[:,0] == t
        scores_t = avg_scores[idx]

        plt.bar(x_positions + i*(0.25), scores_t, 0.25, label = f"t = {t}")

    plt.xticks(x_positions + (0.125), kernels)
    plt.ylim(0.6,0.9)
    plt.title(title)
    plt.xlabel("Kernel Type")
    plt.ylabel("Mean 5-Fold Cross Validation Accuracy")
    plt.legend(title="Threshold")
    plt.savefig(f"plots/{filename}", dpi=300)
    plt.close()


def visualize_hp_RF(hp_values, avg_scores, title, filename):
    plt.figure(figsize=(6,4))
    sns.barplot(x=hp_values, y=avg_scores)
    plt.ylim(0.6,0.7)
    plt.title(title)
    plt.xlabel("Number of Trees")
    plt.ylabel("Mean 5-Fold Cross Validation Accuracy")
    plt.savefig(f"plots/{filename}", dpi=300)
    plt.close()


def train_svm(X_train, y_train, t, kernel):
    y_train_final = (y_train >= t).astype(int)

    # scale
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    X_scaled = (X_train - means) /stds

    # train
    start_time = time.time()
    model = SVC(kernel=kernel, random_state=42)
    model.fit(X_scaled, y_train_final)

    preds = model.predict(X_scaled)
    train_accuracy = np.mean(preds == y_train_final)

    return model, means, stds, train_accuracy, time.time()-start_time


def test_svm(model, X_val, y_val, means, stds, t):
    y_val_final = (y_val >= t).astype(int)
    X_scaled = (X_val - means) /stds

    # test
    start_time = time.time()
    preds = model.predict(X_scaled)
    val_accuracy = np.mean(preds == y_val_final)

    return val_accuracy, time.time()-start_time


def train_rf(X_train, y_train, trees):
    # train
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=trees, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_train)
    train_accuracy = np.mean(preds == y_train)

    return model, train_accuracy, time.time()-start_time


def test_rf(model, X_val, y_val):
    # test
    start_time = time.time()
    preds = model.predict(X_val)
    val_accuracy = np.mean(preds == y_val)

    return val_accuracy, time.time()-start_time


def print_svm_cv_results(hp_values, avg_scores):
    print("\n================= SVM CROSS-VALIDATION RESULTS =================")
    print(" Index | Threshold | Kernel   | Mean Accuracy")
    print("---------------------------------------------------------------")

    for i, ((t, kernel), score) in enumerate(zip(hp_values, avg_scores)):
        print(f" {i:5d} | {t:9d} | {kernel:7s} | {score:.4f}")

    best_idx = np.argmax(avg_scores)
    best_t, best_kernel = hp_values[best_idx]

    print("---------------------------------------------------------------")
    print(f"Best Threshold: {best_t}")
    print(f"Best Kernel:    {best_kernel}")
    print(f"Best Accuracy:  {avg_scores[best_idx]:.4f}")
    print("===============================================================\n")


def print_rf_cv_results(hp_values, avg_scores):
    print("\n============= RANDOM FOREST CROSS-VALIDATION RESULTS =============")
    print(" Trees | Mean Accuracy")
    print("------------------------")

    for n, score in zip(hp_values, avg_scores):
        print(f" {n:5d} | {score:.4f}")

    best_idx = np.argmax(avg_scores)
    best_trees = hp_values[best_idx]

    print("------------------------")
    print(f"Best n_estimators: {best_trees}")
    print(f"Best Accuracy:     {avg_scores[best_idx]:.4f}")
    print("===============================================================\n")


# features are all numeric columns except for the target label
FEATURES = DATA_MULTI.select_dtypes(include=["float64", "int64"]).columns.tolist()
FEATURES.remove("quality")

# 90/10 split
TRAIN_BIN, VAL_BIN = train_test_split(DATA_MULTI, test_size=0.1, random_state=42)

X = TRAIN_BIN[FEATURES].values
Y = TRAIN_BIN["quality"].values

THRESHOLDS = [6,7]
KERNELS = ["linear", "poly", "rbf"]

HP_VALUES_BI, AVG_SCORES_BI = cv_SVM(X, Y, THRESHOLDS, KERNELS)

print_svm_cv_results(HP_VALUES_BI, AVG_SCORES_BI)

visualize_hp_SVM(HP_VALUES_BI, AVG_SCORES_BI, "SVM Hyperparameter Tuning (Kernel x Threshold)", "svm_cv.png")

best_index = np.argmax(AVG_SCORES_BI)
BEST_T, BEST_KERNEL = HP_VALUES_BI[best_index]

SVM, MEANS_BI, STDS_BI, TRAIN_ACCURACY_BI, TRAIN_TIME_BI = train_svm(X, Y, BEST_T, BEST_KERNEL)

print("\n----- Q3.4 Training Results -----")
print(f"Training Accuracy: {TRAIN_ACCURACY_BI:.4f}")
print(f"Training Runtime: {TRAIN_TIME_BI:.4f} sec")
print("---------------------------------")

X_VAL = VAL_BIN[FEATURES].values
Y_VAL= VAL_BIN["quality"].values

VAL_ACCURACY_BI, VAL_TIME_BI = test_svm(SVM, X_VAL, Y_VAL, MEANS_BI, STDS_BI, BEST_T)

print("\n---- Q3.4 Validating Results ----")
print(f"Validating Accuracy: {VAL_ACCURACY_BI:.4f}")
print(f"Validating Runtime: {VAL_TIME_BI:.4f} sec")
print("---------------------------------")

TRAIN_BIN.to_csv("train_multi.csv", index=False)
VAL_BIN.to_csv("validate_multi.csv", index=False)

TREES = [50, 100, 150, 200, 250]
HP_VALUES_MULTI, AVG_SCORES_MULTI = cv_RF(X, Y, TREES)

print_rf_cv_results(HP_VALUES_MULTI, AVG_SCORES_MULTI)

visualize_hp_RF(HP_VALUES_MULTI, AVG_SCORES_MULTI, "Random Forest Hyperparameter Tuning", "rf_cv.png")

BEST_TREES = HP_VALUES_MULTI[np.argmax(AVG_SCORES_MULTI)]

RF, TRAIN_ACCURACY_MULTI, TRAIN_TIME_MULTI = train_rf(X, Y, BEST_TREES)

print("\n----- Q4.4 Training Results -----")
print(f"Training Accuracy: {TRAIN_ACCURACY_MULTI:.4f}")
print(f"Training Runtime: {TRAIN_TIME_MULTI:.4f} sec")
print("---------------------------------")

VAL_ACCURACY_MULTI, VAL_TIME_MULTI = test_rf(RF, X_VAL, Y_VAL)

print("\n---- Q4.4 Validating Results ----")
print(f"Validating Accuracy: {VAL_ACCURACY_MULTI:.4f}")
print(f"Validating Runtime: {VAL_TIME_MULTI:.4f} sec")
print("---------------------------------")

# ------------------------- Part C ------------------------- #

def apply_pca(X_train, X_test, n_components):
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

def train_svm_pca(X_train_pca, y_train, t, kernel):
    y_train_bin = (y_train >= t).astype(int)
    start = time.time()
    model = SVC(kernel=kernel, random_state=42)
    model.fit(X_train_pca, y_train_bin)
    preds = model.predict(X_train_pca)
    acc = np.mean(preds == y_train_bin)
    return model, acc, time.time() - start


def test_svm_pca(model, X_val_pca, y_val, t):
    y_val_bin = (y_val >= t).astype(int)
    start = time.time()
    preds = model.predict(X_val_pca)
    acc = np.mean(preds == y_val_bin)
    return acc, time.time() - start

def train_rf_pca(X_train_pca, y_train, trees):
    # train
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=trees, random_state=42)
    model.fit(X_train_pca, y_train)

    preds = model.predict(X_train_pca)
    train_accuracy = np.mean(preds == y_train)

    return model, train_accuracy, time.time()-start_time


def test_rf_pca(model, X_val_pca, y_val):
    # test
    start_time = time.time()
    preds = model.predict(X_val_pca)
    val_accuracy = np.mean(preds == y_val)

    return val_accuracy, time.time()-start_time

# ---- Scale before PCA ----
means = X.mean(axis=0)
stds = X.std(axis=0)

X_train_scaled = (X - means) / stds
X_val_scaled   = (X_VAL   - means) / stds

# ---- PCA reduces feature count by half ----
n_components = len(FEATURES) // 2
X_train_pca, X_val_pca, PCA_MODEL = apply_pca(X_train_scaled, X_val_scaled, n_components)

train_pca_df = pd.DataFrame(X_train_pca, columns=[f"PC{i+1}" for i in range(n_components)])
val_pca_df   = pd.DataFrame(X_val_pca, columns=[f"PC{i+1}" for i in range(n_components)])

# Convert to binary AFTER PCA
y_train_binary = (Y >= T).astype(int)
y_val_binary   = (Y_VAL >= T).astype(int)

# Create PCA-reduced train/validation DataFrames
train_pca_bin_df = pd.DataFrame(X_train_pca, columns=[f"PC{i+1}" for i in range(n_components)])
val_pca_bin_df   = pd.DataFrame(X_val_pca, columns=[f"PC{i+1}" for i in range(n_components)])

# Attach binary label
train_pca_bin_df["quality_binary"] = y_train_binary
val_pca_bin_df["quality_binary"]   = y_val_binary

# Save files for Q4.1
train_pca_bin_df.to_csv("train_pca_svm.csv", index=False)
val_pca_bin_df.to_csv("validate_pca_svm.csv", index=False)

# Add the target variable (multiclass quality)
train_pca_df["quality"] = Y
val_pca_df["quality"]   = Y_VAL

train_pca_df.to_csv("train_pca_rf.csv", index=False)
val_pca_df.to_csv("validate_pca_rf.csv", index=False)

HP_VALUES_BI_PCA, AVG_SCORES_BI_PCA = cv_SVM(X_train_pca, Y, THRESHOLDS, KERNELS)

print_svm_cv_results(HP_VALUES_BI_PCA, AVG_SCORES_BI_PCA)

visualize_hp_SVM(
    HP_VALUES_BI_PCA,
    AVG_SCORES_BI_PCA,
    "SVM Hyperparameter Tuning with PCA (Threshold x Kernel)",
    "svm_cv_pca.png"
)

best_index = np.argmax(AVG_SCORES_BI_PCA)
BEST_T_PCA, BEST_KERNEL_PCA = HP_VALUES_BI_PCA[best_index]

SVM_PCA, TRAIN_ACC_PCA, TRAIN_TIME_PCA = train_svm_pca(X_train_pca, Y, BEST_T_PCA, BEST_KERNEL_PCA)

VAL_ACC_PCA, VAL_TIME_PCA = test_svm_pca(SVM_PCA, X_val_pca, Y_VAL, BEST_T_PCA)

print("\n---- Q3.4 PCA SVM Training Results ----")
print(f"Training Accuracy: {TRAIN_ACC_PCA:.4f}")
print(f"Training Runtime: {TRAIN_TIME_PCA:.4f} sec")

print("\n---- Q3.4 PCA SVM Validation Results ----")
print(f"Validation Accuracy: {VAL_ACC_PCA:.4f}")
print(f"Validation Runtime: {VAL_TIME_PCA:.4f} sec")
print("----------------------------------------")

HP_VALUES_MULTI_PCA, AVG_SCORES_MULTI_PCA = cv_RF(X_train_pca, Y, TREES)

print_rf_cv_results(HP_VALUES_MULTI_PCA, AVG_SCORES_MULTI_PCA)

visualize_hp_RF(
    HP_VALUES_MULTI_PCA,
    AVG_SCORES_MULTI_PCA,
    "Random Forest Hyperparameter Tuning with PCA",
    "rf_cv_pca.png"
)

best_index = np.argmax(AVG_SCORES_MULTI_PCA)
BEST_TREES_PCA = HP_VALUES_MULTI_PCA[best_index]

RF_PCA, TRAIN_ACC_PCA, TRAIN_TIME_PCA = train_rf_pca(X_train_pca, Y, BEST_TREES_PCA)

VAL_ACC_PCA, VAL_TIME_PCA = test_rf_pca(RF_PCA, X_val_pca, Y_VAL)

print("\n---- Q4.4 PCA RF Training Results ----")
print(f"Training Accuracy: {TRAIN_ACC_PCA:.4f}")
print(f"Training Runtime: {TRAIN_TIME_PCA:.4f} sec")

print("\n---- Q4.4 PCA RF Validation Results ----")
print(f"Validation Accuracy: {VAL_ACC_PCA:.4f}")
print(f"Validation Runtime: {VAL_TIME_PCA:.4f} sec")
print("----------------------------------------")
