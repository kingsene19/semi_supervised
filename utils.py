import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelSpreading

# Charger et découper la base
def load_data(filepath):
    column_names = [f"V{i}" for i in range(40)]
    column_names.extend(["Label"])
    df = pd.read_csv(filepath, sep=' ', names=column_names)
    feats = df.drop(columns=["Label"]).to_numpy()
    labels = df["Label"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(feats, labels, stratify=labels, test_size=.5)
    print("Chargement des données effectué")
    print("Données d'apprentissage X: ", X_train.shape, " Y: ", y_train.shape)
    print("Données test X: ", X_test.shape, " Y: ", y_test.shape)
    return (X_train, y_train), (X_test, y_test)

# Rendre la base semi supervisé
def make_partially_labelled(y_train, pct=0.3):
    np.random.seed(42)
    if pct > 1:
        pct = pct / 100
    if pct < 0 or pct > 1:
        raise ValueError("pct must be between 0 and 1")
    pct = 1 - pct
    
    y_train_partial = np.copy(y_train).astype(float)
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    class_mask_counts = {}
    for c, count in zip(unique_classes, class_counts):
        class_mask_counts[c] = int(np.round(count * pct))
    
    for c in unique_classes:
        class_indices = np.where(y_train == c)[0]        
        num_class_mask = class_mask_counts[c]
        mask_indices = np.random.choice(class_indices, num_class_mask, replace=False)        
        y_train_partial[mask_indices] = -1
    return y_train_partial


# Calculer le score de pertinence
def calculate_pertinence(X_train, y_train_partial):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    indices_labelled = np.where(y_train_partial != -1)[0]
    indices_unlabelled = np.where(y_train_partial == -1)[0]

    ficher_scores = fisher_score(X_train[indices_labelled], y_train_partial[indices_labelled])
    laplacian_scores = laplacian_score(X_train[indices_unlabelled])

    pertinence_scores = {f"V{i}": ficher_scores[i]/laplacian_scores[i] for i  in range(40)}

    return pertinence_scores

# Calculer le score Fisher
def fisher_score(X_train, y_train):
    X = np.array(X_train)
    y = np.array(y_train)
    n_features = X.shape[1]
    classes = np.unique(y)    
    fisher_scores = np.zeros(n_features)
    
    for feature in range(n_features):
        mu = np.mean(X[:, feature])
        n_i = []
        mu_i = []
        sigma_i = []
        
        for c in classes:
            mask = (y == c)
            n_i.append(np.sum(mask))            
            mu_i.append(np.mean(X[mask, feature]))            
            sigma_i.append(np.std(X[mask, feature], ddof=1))        
        n_i = np.array(n_i)
        mu_i = np.array(mu_i)
        sigma_i = np.array(sigma_i)        
        numerateur = np.sum(n_i * (mu_i - mu)**2)
        denominateur = np.sum(n_i * sigma_i**2)
        fisher_scores[feature] = numerateur / denominateur if denominateur != 0 else 0
    return fisher_scores

# Calculer le score laplacien
def laplacian_score(X, t=10):
    n_samples, n_features = X.shape
    scores = []
    pairwise_distances = cdist(X, X, metric='euclidean')
    similarity_matrix = np.exp(-pairwise_distances ** 2 / t)
    for i in range(n_features):
        v = X[:, i]
        v_diff = v[:, np.newaxis] - v[np.newaxis, :]

        numerator = np.sum(similarity_matrix * (v_diff ** 2))
        denominator = np.var(v)

        scores.append(numerator / (denominator + 1e-8))
    return np.array(scores)

# Tracer l'histogramme des pertinences
def plot_relevance_histogram(scores_dict):
    features, values = zip(*scores_dict.items())
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(values)), values)
    plt.title('Histogramme des scores de pertinence des variables')
    plt.xlabel('Variables')
    plt.ylabel('Score de pertinence')
    plt.xticks(range(len(features)), [features[i] for i in range(0, len(features))], rotation=45)
    plt.tight_layout()
    plt.show()

# Fonction pour l'évaluation du MLP
def evaluate_mlp(X_train, X_test, y_train_partial, y_test, selected_features):
    mlp = MLPClassifier(hidden_layer_sizes=(128,64,32) ,max_iter=500, learning_rate='adaptive', random_state=42, solver="adam")
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    labeled_mask = y_train_partial != -1
    X_labeled = X_train_selected[labeled_mask]
    y_labeled = y_train_partial[labeled_mask]
    X_unlabeled = X_train_selected[~labeled_mask]
    mlp = MLPClassifier(random_state=42)
    mlp.fit(X_labeled, y_labeled)
    pseudo_labels = mlp.predict(X_unlabeled)
    X_combined = np.vstack((X_labeled, X_unlabeled))
    y_combined = np.concatenate((y_labeled, pseudo_labels))
    mlp.fit(X_combined, y_combined)
    y_pred = mlp.predict(X_test_selected)
    return accuracy_score(y_test, y_pred)

# Fonction pour tracer la courbe d'efficacité en fonction de la sélection de variables
def plot_efficiency_curve(X_train, X_test, y_train_partial, y_test, scores_dict, normalised=True, feature_select=True, reverse=True):
    sorted_features = sorted(scores_dict.items(), key=lambda x: x[1], reverse=reverse)
    
    sorted_feature_names = [item[0] for item in sorted_features]
    
    accuracies = []

    if reverse:
        feature_counts = list(range(5, len(sorted_feature_names) + 1, 5))
    else:
        feature_counts = list(range(5,  21, 5))

    if normalised:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    for num_features in feature_counts:
        if feature_select:
            selected_features = [int(name[1:]) - 1 for name in sorted_feature_names[:num_features]]
        else:
            selected_features = [i for i in range(num_features)]
        accuracy = evaluate_mlp(X_train, X_test, y_train_partial, y_test, selected_features)
        accuracies.append(accuracy)
    
    plt.figure(figsize=(10, 6))
    plt.plot(feature_counts, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
    if not feature_select:
        plt.title('Performance du MLP en fonction du nombre de variables (sans sélection)', fontsize=14)
    elif not reverse:
        plt.title('Performance du MLP en fonction du nombre de variables non pertinentes', fontsize=14)
    else:
        plt.title('Performance du MLP en fonction du nombre de variables pertinentes', fontsize=14)
    if not feature_select:
        plt.xlabel('Nombre de variables', fontsize=12)
    elif not reverse:
        plt.xlabel('Nombre de variables non pertinentes', fontsize=12)
    else:
        plt.xlabel('Nombre de variables pertinentes', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(feature_counts)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Fonction pour tracer la courbe d'efficacité en fonction du pourcentage de données labélisées
def plot_efficiency_by_pct(X_train, X_test, y_train, y_test, n_features=20, step=10):

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    percentages = range(10, 100, step)
    accuracies = []
    
    for pct in percentages:
        y_train_partial = make_partially_labelled(y_train, pct=pct/100)
        
        scores = calculate_pertinence(X_train_norm, y_train_partial)
        
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [int(name[1:]) - 1 for name in [item[0] for item in sorted_features[:n_features]]]
        
        accuracy = evaluate_mlp(X_train_norm, X_test_norm, y_train_partial, y_test, selected_features)
        accuracies.append(accuracy)
    
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, accuracies, marker='o', linestyle='-', color='b')
    plt.title(f'Performance du MLP en fonction du % de données labélisées\n({n_features} variables sélectionnées)', 
              fontsize=14)
    plt.xlabel('Pourcentage de données labélisées (%)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True)
    plt.xticks(percentages)
    plt.tight_layout()
    plt.show()
