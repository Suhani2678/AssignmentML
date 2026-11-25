import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from torch.utils.data import DataLoader, Subset

from cifar10_setup import (
    BATCH_SIZE,
    SEED,
    make_train_validation_loaders,
    seed_everything,
)
from models import ConvAutoencoder, SmallCNN
from training import collect_embeddings


RESULTS = Path("results")
EMBEDDINGS = RESULTS / "embeddings"
LINEAR_PROBE_TRAIN_SIZE = 10_000
SVM_TRAIN_SIZE = 5_000
SVM_C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
SVM_KERNELS = ("linear", "rbf", "poly")
CLUSTER_COUNTS = (8, 10, 12)


def _ordered_loader(subset_loader, batch_size):
    subset = subset_loader.dataset
    source_indices = np.asarray(subset.indices)
    loader = DataLoader(
        Subset(subset.dataset, source_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return loader, source_indices


def _save_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _extract_cnn_embeddings(device, batch_size):
    train_subset_loader, validation_subset_loader = make_train_validation_loaders(
        batch_size=batch_size, condition="clean"
    )
    train_loader, source_train_indices = _ordered_loader(
        train_subset_loader, batch_size
    )
    validation_loader, source_validation_indices = _ordered_loader(
        validation_subset_loader, batch_size
    )
    model = SmallCNN(embedding_dim=240).to(device)
    checkpoint = torch.load(
        RESULTS / "checkpoints" / "clean_cnn" / "best.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["state_dict"])
    train_features, train_labels = collect_embeddings(model, train_loader, device)
    validation_features, validation_labels = collect_embeddings(
        model, validation_loader, device
    )
    del model, train_subset_loader, validation_subset_loader
    return (
        train_features,
        train_labels,
        source_train_indices,
        validation_features,
        validation_labels,
        source_validation_indices,
    )


def _select_linear_probes(device, batch_size, train_labels, validation_labels):
    rows = []
    selection = {}
    train_positions, _ = train_test_split(
        np.arange(len(train_labels)),
        train_size=LINEAR_PROBE_TRAIN_SIZE,
        random_state=SEED,
        stratify=train_labels,
    )
    for bottleneck_dim in (32, 128, 512):
        train_subset_loader, validation_subset_loader = make_train_validation_loaders(
            batch_size=batch_size, condition="clean", normalize=False
        )
        train_loader, _ = _ordered_loader(train_subset_loader, batch_size)
        validation_loader, _ = _ordered_loader(validation_subset_loader, batch_size)
        model = ConvAutoencoder(bottleneck_dim).to(device)
        checkpoint = torch.load(
            RESULTS / "checkpoints" / f"clean_ae_{bottleneck_dim}" / "best.pt",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["state_dict"])
        train_features, labels = collect_embeddings(
            model, train_loader, device, autoencoder=True
        )
        validation_features, val_labels = collect_embeddings(
            model, validation_loader, device, autoencoder=True
        )
        if not np.array_equal(labels, train_labels):
            raise RuntimeError("Training labels do not align with frozen AE features.")
        if not np.array_equal(val_labels, validation_labels):
            raise RuntimeError("Validation labels do not align with frozen AE features.")

        best_row = None
        best_model = None
        for c_value in (0.01, 0.1, 1.0, 10.0):
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_features[train_positions])
            validation_x = scaler.transform(validation_features)
            probe = LinearSVC(
                C=c_value,
                dual="auto",
                max_iter=5_000,
                random_state=SEED,
            )
            probe.fit(train_x, train_labels[train_positions])
            train_accuracy = accuracy_score(
                train_labels[train_positions], probe.predict(train_x)
            )
            validation_accuracy = accuracy_score(
                validation_labels, probe.predict(validation_x)
            )
            row = {
                "bottleneck_dim": bottleneck_dim,
                "C": c_value,
                "train_samples": LINEAR_PROBE_TRAIN_SIZE,
                "training_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
            }
            rows.append(row)
            if best_row is None or validation_accuracy > best_row["validation_accuracy"]:
                best_row = row
                best_model = (scaler, probe)
        selection[str(bottleneck_dim)] = {
            "C": best_row["C"],
            "validation_accuracy": best_row["validation_accuracy"],
            "train_samples": LINEAR_PROBE_TRAIN_SIZE,
        }
        joblib.dump(
            best_model,
            RESULTS / f"linear_probe_{bottleneck_dim}.joblib",
        )
        del (
            train_subset_loader,
            validation_subset_loader,
            train_loader,
            validation_loader,
            model,
            train_features,
            validation_features,
        )
    _save_csv(RESULTS / "linear_probe_sweep.csv", rows)
    return selection


def _select_svm(
    train_features,
    train_labels,
    train_source_indices,
    validation_features,
    validation_labels,
):
    train_positions, _ = train_test_split(
        np.arange(len(train_labels)),
        train_size=SVM_TRAIN_SIZE,
        random_state=SEED,
        stratify=train_labels,
    )
    x_train = train_features[train_positions]
    y_train = train_labels[train_positions]
    rows = []
    estimators = {}
    for kernel in SVM_KERNELS:
        for c_value in SVM_C_VALUES:
            print(f"Fitting SVM kernel={kernel} C={c_value}", flush=True)
            estimator = SVC(
                C=c_value,
                kernel=kernel,
                degree=3,
                gamma="scale",
                decision_function_shape="ovo",
                cache_size=512,
            )
            estimator.fit(x_train, y_train)
            train_accuracy = accuracy_score(y_train, estimator.predict(x_train))
            validation_accuracy = accuracy_score(
                validation_labels, estimator.predict(validation_features)
            )
            row = {
                "kernel": kernel,
                "C": c_value,
                "degree": 3 if kernel == "poly" else "",
                "gamma": "scale" if kernel != "linear" else "",
                "training_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
                "support_vector_fraction": len(estimator.support_) / len(y_train),
            }
            rows.append(row)
            estimators[(kernel, c_value)] = estimator

    best_row = max(
        rows,
        key=lambda row: (
            row["validation_accuracy"],
            -row["C"],
            -SVM_KERNELS.index(row["kernel"]),
        ),
    )
    best_estimator = estimators[(best_row["kernel"], best_row["C"])]
    joblib.dump(best_estimator, RESULTS / "selected_svm.joblib")
    np.save(RESULTS / "svm_train_positions.npy", train_positions)
    np.save(RESULTS / "svm_train_source_indices.npy", train_source_indices[train_positions])
    _save_csv(RESULTS / "svm_sweep.csv", rows)
    return {
        "kernel": best_row["kernel"],
        "C": best_row["C"],
        "degree": 3,
        "gamma": "scale",
        "validation_accuracy": best_row["validation_accuracy"],
        "training_accuracy": best_row["training_accuracy"],
        "support_vector_fraction": best_row["support_vector_fraction"],
        "training_samples": SVM_TRAIN_SIZE,
        "selection_rule": "highest validation accuracy; ties prefer lower C, then linear/RBF/poly order",
    }


def _select_clustering(train_features, validation_features):
    candidates = []
    validation_positions = np.linspace(
        0, len(validation_features) - 1, min(2_000, len(validation_features)), dtype=int
    )
    for n_clusters in CLUSTER_COUNTS:
        print(f"Scoring k-means with k={n_clusters}", flush=True)
        candidate = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=SEED,
            n_init=3,
            batch_size=1_024,
            max_iter=60,
            reassignment_ratio=0.01,
        )
        candidate.fit(train_features)
        labels = candidate.predict(validation_features[validation_positions])
        silhouette = silhouette_score(
            validation_features[validation_positions],
            labels,
            metric="euclidean",
        )
        candidates.append(
            {"n_clusters": n_clusters, "validation_silhouette": silhouette}
        )
    best = max(candidates, key=lambda row: row["validation_silhouette"])
    final_model = MiniBatchKMeans(
        n_clusters=best["n_clusters"],
        random_state=SEED,
        n_init=5,
        batch_size=1_024,
        max_iter=100,
        reassignment_ratio=0.01,
    )
    final_model.fit(train_features)
    joblib.dump(final_model, RESULTS / "selected_kmeans.joblib")
    _save_csv(RESULTS / "cluster_sweep.csv", candidates)
    return {
        "method": "MiniBatchKMeans",
        "n_clusters": best["n_clusters"],
        "validation_silhouette": best["validation_silhouette"],
        "selection_rule": "highest validation silhouette; labels not used",
        "final_fit_samples": len(train_features),
    }


def run_selection(batch_size):
    RESULTS.mkdir(exist_ok=True)
    EMBEDDINGS.mkdir(exist_ok=True)
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        train_features,
        train_labels,
        train_source_indices,
        validation_features,
        validation_labels,
        validation_source_indices,
    ) = _extract_cnn_embeddings(device, batch_size)
    for name, array in (
        ("cnn_train_features.npy", train_features),
        ("cnn_train_labels.npy", train_labels),
        ("cnn_train_source_indices.npy", train_source_indices),
        ("cnn_validation_features.npy", validation_features),
        ("cnn_validation_labels.npy", validation_labels),
        ("cnn_validation_source_indices.npy", validation_source_indices),
    ):
        np.save(EMBEDDINGS / name, array)

    print("\n=== Frozen autoencoder linear probes ===", flush=True)
    linear_probe_selection = _select_linear_probes(
        device, batch_size, train_labels, validation_labels
    )
    print("\n=== SVM validation sweep ===", flush=True)
    svm_selection = _select_svm(
        train_features,
        train_labels,
        train_source_indices,
        validation_features,
        validation_labels,
    )
    print("\n=== Unsupervised k-means selection ===", flush=True)
    cluster_selection = _select_clustering(train_features, validation_features)

    choices_path = RESULTS / "selection.json"
    choices = {
        "official_test_opened": False,
        "cnn_representation": "240-dimensional hidden embedding from clean CNN",
        "svm": svm_selection,
        "clustering": cluster_selection,
        "linear_probes": linear_probe_selection,
    }
    choices_path.write_text(json.dumps(choices, indent=2), encoding="utf-8")
    print(f"\nValidation selections saved to {choices_path.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Select SVM, clustering, and linear-probe settings using only train/validation data."
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    run_selection(args.batch_size)


if __name__ == "__main__":
    main()
