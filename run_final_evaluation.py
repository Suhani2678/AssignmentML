import argparse
import csv
import itertools
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
)
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from cifar10_setup import (
    BATCH_SIZE,
    DATA_DIR,
    SEED,
    make_test_loader,
    make_train_validation_loaders,
    seed_everything,
)
from models import ConvAutoencoder, MLP, SmallCNN
from training import add_gaussian_noise, collect_embeddings, collect_predictions


RESULTS = Path("results")
FIGURES = RESULTS / "figures"
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)


def _load_state(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _classification_metrics(probabilities, labels):
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = predictions == labels
    bin_edges = np.linspace(0.0, 1.0, 16)
    ece = 0.0
    calibration_rows = []
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lower) & (
            confidences < upper if upper < 1.0 else confidences <= upper
        )
        if not np.any(mask):
            continue
        bin_accuracy = float(correct[mask].mean())
        bin_confidence = float(confidences[mask].mean())
        ece += float(mask.mean()) * abs(bin_accuracy - bin_confidence)
        calibration_rows.append(
            {
                "lower_confidence": lower,
                "upper_confidence": upper,
                "count": int(mask.sum()),
                "accuracy": bin_accuracy,
                "mean_confidence": bin_confidence,
            }
        )
    per_class = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_mask = labels == class_index
        per_class[class_name] = {
            "accuracy": float((predictions[class_mask] == labels[class_mask]).mean()),
            "count": int(class_mask.sum()),
        }
    return (
        {
            "accuracy": float(accuracy_score(labels, predictions)),
            "error_rate": float(1.0 - accuracy_score(labels, predictions)),
            "mean_confidence": float(confidences.mean()),
            "mean_confidence_correct": float(confidences[correct].mean()),
            "mean_confidence_incorrect": float(confidences[~correct].mean()),
            "expected_calibration_error_15_bins": float(ece),
            "per_class_accuracy": per_class,
        },
        predictions,
        calibration_rows,
    )


def _evaluate_autoencoders(device, batch_size, classifier):
    with (RESULTS / "autoencoder_selection.json").open(encoding="utf-8") as file:
        selected_dim = json.load(file)["selected_bottleneck"]
    selected_ae = _load_state(
        ConvAutoencoder(selected_dim).to(device),
        RESULTS / "checkpoints" / f"clean_ae_{selected_dim}" / "best.pt",
        device,
    )
    denoising_ae = _load_state(
        ConvAutoencoder(selected_dim).to(device),
        RESULTS / "checkpoints" / "denoising_ae" / "best.pt",
        device,
    )
    test_loader = make_test_loader(batch_size=batch_size, condition="clean", normalize=False)
    generator = torch.Generator(device=device).manual_seed(SEED + 20_000)
    sums = {
        "clean_reconstruction_mse": 0.0,
        "corrupted_input_mse": 0.0,
        "clean_ae_reconstruction_from_corruption_mse": 0.0,
        "denoising_ae_reconstruction_mse": 0.0,
    }
    clean_mse_by_dim = {dim: 0.0 for dim in (32, 128, 512)}
    count = 0
    probabilities = {
        "corrupted_images": [],
        "clean_ae_reconstructions": [],
        "denoising_ae_reconstructions": [],
    }
    labels = []
    representations = {}
    models = {}
    for bottleneck_dim in (32, 128, 512):
        models[bottleneck_dim] = _load_state(
            ConvAutoencoder(bottleneck_dim).to(device),
            RESULTS / "checkpoints" / f"clean_ae_{bottleneck_dim}" / "best.pt",
            device,
        )
        representations[bottleneck_dim] = []

    classifier.eval()
    with torch.no_grad():
        for images, batch_labels in test_loader:
            images = images.to(device)
            noisy = add_gaussian_noise(images, 0.15, generator)
            clean_reconstruction = selected_ae(images)
            clean_ae_from_noise = selected_ae(noisy)
            denoised = denoising_ae(noisy)
            batch_count = images.size(0)
            count += batch_count
            sums["clean_reconstruction_mse"] += (
                torch.nn.functional.mse_loss(
                    clean_reconstruction, images, reduction="sum"
                ).item()
            )
            sums["corrupted_input_mse"] += (
                torch.nn.functional.mse_loss(noisy, images, reduction="sum").item()
            )
            sums["clean_ae_reconstruction_from_corruption_mse"] += (
                torch.nn.functional.mse_loss(
                    clean_ae_from_noise, images, reduction="sum"
                ).item()
            )
            sums["denoising_ae_reconstruction_mse"] += (
                torch.nn.functional.mse_loss(denoised, images, reduction="sum").item()
            )
            for name, model_input in (
                ("corrupted_images", noisy),
                ("clean_ae_reconstructions", clean_ae_from_noise),
                ("denoising_ae_reconstructions", denoised),
            ):
                normalized = (model_input - MEAN.to(device)) / STD.to(device)
                probabilities[name].append(
                    torch.softmax(classifier(normalized), dim=1).cpu().numpy()
                )
            for bottleneck_dim, model in models.items():
                reconstruction = model(images)
                clean_mse_by_dim[bottleneck_dim] += torch.nn.functional.mse_loss(
                    reconstruction, images, reduction="sum"
                ).item()
                representations[bottleneck_dim].append(
                    model.encode(images).cpu().numpy()
                )
            labels.append(batch_labels.numpy())

    labels = np.concatenate(labels)
    result = {
        "selected_clean_autoencoder_bottleneck": selected_dim,
        "denoising_corruption": "independent additive Gaussian noise, sigma=0.15, clipped to [0,1]",
        "test_reconstruction_mse": {
            key: value / (count * 3 * 32 * 32) for key, value in sums.items()
        },
        "clean_reconstruction_mse_by_bottleneck": {
            str(dim): value / (count * 3 * 32 * 32)
            for dim, value in clean_mse_by_dim.items()
        },
        "classification": {},
        "frozen_encoder_linear_probe_test_accuracy": {},
    }
    calibration = {}
    for name, batch_probabilities in probabilities.items():
        metrics, _, bins = _classification_metrics(
            np.concatenate(batch_probabilities), labels
        )
        result["classification"][name] = metrics
        calibration[name] = bins

    for bottleneck_dim, batches in representations.items():
        test_features = np.concatenate(batches)
        scaler, probe = joblib.load(RESULTS / f"linear_probe_{bottleneck_dim}.joblib")
        prediction = probe.predict(scaler.transform(test_features))
        result["frozen_encoder_linear_probe_test_accuracy"][str(bottleneck_dim)] = (
            float(accuracy_score(labels, prediction))
        )
    with (RESULTS / "autoencoder_test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    with (RESULTS / "autoencoder_calibration_bins.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(calibration, file, indent=2)
    return result


def _save_boundary_figure(pair_records, train_dataset, test_dataset):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(pair_records), 4, figsize=(7, max(3, 2.4 * len(pair_records)))
    )
    if len(pair_records) == 1:
        axes = np.expand_dims(axes, 0)
    for row_index, record in enumerate(pair_records):
        examples = record["support_vector_examples"][:2] + record["near_boundary_test_examples"][:2]
        for column_index, example in enumerate(examples):
            dataset = train_dataset if example["split"] == "train" else test_dataset
            image = dataset.data[example["source_index"]]
            axis = axes[row_index, column_index]
            axis.imshow(image)
            axis.set_title(
                f"{example['split']} {CLASS_NAMES[example['true_class']]}\n"
                "score={:.3f}".format(example["pair_score"]),
                fontsize=7,
            )
            axis.axis("off")
        axes[row_index, 0].set_ylabel(
            f"{CLASS_NAMES[record['class_a']]} / {CLASS_NAMES[record['class_b']]}",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(FIGURES / "svm_difficult_pairs.png", dpi=180)
    plt.close(fig)


def _analyze_svm_pairs(
    svm,
    train_features,
    train_labels,
    train_source_indices,
    test_features,
    test_labels,
    test_predictions,
):
    matrix = confusion_matrix(test_labels, test_predictions, labels=np.arange(10))
    pair_errors = []
    for class_a in range(10):
        for class_b in range(class_a + 1, 10):
            count = int(matrix[class_a, class_b] + matrix[class_b, class_a])
            pair_errors.append((count, class_a, class_b))
    selected_pairs = [
        (class_a, class_b)
        for _, class_a, class_b in sorted(pair_errors, reverse=True)[:3]
    ]

    source_positions = np.load(RESULTS / "svm_train_positions.npy")
    svm_train_features = train_features[source_positions]
    svm_train_labels = train_labels[source_positions]
    train_scores = svm.decision_function(svm_train_features)
    test_scores = svm.decision_function(test_features)
    class_pairs = list(itertools.combinations(range(10), 2))
    pair_records = []
    for class_a, class_b in selected_pairs:
        score_index = class_pairs.index((class_a, class_b))
        train_mask = np.isin(svm_train_labels, (class_a, class_b))
        train_candidates = np.flatnonzero(train_mask)
        support_mask = np.isin(svm.support_, train_candidates)
        support_positions = svm.support_[support_mask]
        support_positions = support_positions[
            np.argsort(np.abs(train_scores[support_positions, score_index]))
        ][:4]

        test_mask = np.isin(test_labels, (class_a, class_b))
        test_candidates = np.flatnonzero(test_mask)
        test_positions = test_candidates[
            np.argsort(np.abs(test_scores[test_candidates, score_index]))
        ][:4]
        support_examples = [
            {
                "split": "train",
                "source_index": int(train_source_indices[source_positions[position]]),
                "true_class": int(svm_train_labels[position]),
                "predicted_class": int(
                    svm.predict(svm_train_features[position : position + 1])[0]
                ),
                "pair_score": float(train_scores[position, score_index]),
            }
            for position in support_positions
        ]
        test_examples = [
            {
                "split": "test",
                "source_index": int(position),
                "true_class": int(test_labels[position]),
                "predicted_class": int(test_predictions[position]),
                "pair_score": float(test_scores[position, score_index]),
            }
            for position in test_positions
        ]
        pair_records.append(
            {
                "class_a": class_a,
                "class_b": class_b,
                "test_confusion_count": int(
                    matrix[class_a, class_b] + matrix[class_b, class_a]
                ),
                "support_vector_examples": support_examples,
                "near_boundary_test_examples": test_examples,
            }
        )
    train_dataset = CIFAR10(root=DATA_DIR, train=True, download=False)
    test_dataset = CIFAR10(root=DATA_DIR, train=False, download=False)
    _save_boundary_figure(pair_records, train_dataset, test_dataset)
    return pair_records


def run_final_evaluation(batch_size):
    if (RESULTS / "final_test.json").exists():
        raise RuntimeError(
            "Final test evaluation already exists. Do not rerun or tune against the test set."
        )
    with (RESULTS / "selection.json").open(encoding="utf-8") as file:
        selection = json.load(file)
    if selection.get("official_test_opened") is not False:
        raise RuntimeError("Validation selection metadata is missing or already modified.")

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(SEED)
    final_results = {
        "seed": SEED,
        "test_evaluation_policy": "all choices fixed using train/validation only before this run",
        "classification": {},
        "autoencoder": {},
        "svm": {},
        "clustering": {},
    }
    calibration = {}
    test_logits_by_condition = {}

    print("Beginning the one-time final test evaluation.", flush=True)
    for condition in ("clean", "scrambled", "grayscale"):
        test_loader = make_test_loader(
            batch_size=batch_size, condition=condition, normalize=True
        )
        for architecture in ("mlp", "cnn"):
            model = (
                MLP(hidden_dim=326).to(device)
                if architecture == "mlp"
                else SmallCNN(embedding_dim=240).to(device)
            )
            model = _load_state(
                model,
                RESULTS / "checkpoints" / f"{condition}_{architecture}" / "best.pt",
                device,
            )
            logits, probabilities, labels = collect_predictions(model, test_loader, device)
            metrics, predictions, bins = _classification_metrics(probabilities, labels)
            key = f"{condition}_{architecture}"
            final_results["classification"][key] = metrics
            calibration[key] = bins
            test_logits_by_condition[key] = logits
            np.save(RESULTS / f"predictions_{key}.npy", predictions)
            if condition == "clean" and architecture == "cnn":
                clean_classifier = model
            del model

    # Keep one confidence/error summary and a confusion plot directly in the results.
    clean_logits = test_logits_by_condition["clean_cnn"]
    clean_predictions = np.load(RESULTS / "predictions_clean_cnn.npy")
    confusion = confusion_matrix(labels, clean_predictions, labels=np.arange(10))
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(confusion, cmap="Blues")
    axis.set_xticks(np.arange(10), CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticks(np.arange(10), CLASS_NAMES)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(FIGURES / "clean_cnn_confusion.png", dpi=180)
    plt.close(fig)
    with (RESULTS / "calibration_bins.json").open("w", encoding="utf-8") as file:
        json.dump(calibration, file, indent=2)

    print("\n=== Final autoencoder evaluation ===", flush=True)
    final_results["autoencoder"] = _evaluate_autoencoders(
        device, batch_size, clean_classifier
    )

    print("\n=== Final SVM and clustering evaluation ===", flush=True)
    test_loader = make_test_loader(
        batch_size=batch_size, condition="clean", normalize=True
    )
    test_features, test_labels = collect_embeddings(
        clean_classifier, test_loader, device
    )
    selected_svm = joblib.load(RESULTS / "selected_svm.joblib")
    svm_predictions = selected_svm.predict(test_features)
    svm_accuracy = float(accuracy_score(test_labels, svm_predictions))
    decision_scores = selected_svm.decision_function(test_features)
    final_results["svm"] = {
        "selected_configuration": selection["svm"],
        "test_accuracy": svm_accuracy,
        "test_error_rate": 1.0 - svm_accuracy,
        "confusion_matrix": confusion_matrix(
            test_labels, svm_predictions, labels=np.arange(10)
        ).tolist(),
    }
    np.save(RESULTS / "svm_test_decision_scores.npy", decision_scores)
    svm_pairs = _analyze_svm_pairs(
        selected_svm,
        np.load(RESULTS / "embeddings" / "cnn_train_features.npy"),
        np.load(RESULTS / "embeddings" / "cnn_train_labels.npy"),
        np.load(RESULTS / "embeddings" / "cnn_train_source_indices.npy"),
        test_features,
        test_labels,
        svm_predictions,
    )
    final_results["svm"]["difficult_pairs"] = svm_pairs

    kmeans = joblib.load(RESULTS / "selected_kmeans.joblib")
    test_clusters = kmeans.predict(test_features)
    train_features = np.load(RESULTS / "embeddings" / "cnn_train_features.npy")
    train_labels = np.load(RESULTS / "embeddings" / "cnn_train_labels.npy")
    train_clusters = kmeans.predict(train_features)
    cluster_to_class = {}
    for cluster in np.unique(train_clusters):
        members = train_labels[train_clusters == cluster]
        cluster_to_class[int(cluster)] = int(np.bincount(members, minlength=10).argmax())
    mapped_cluster_predictions = np.asarray(
        [cluster_to_class[int(cluster)] for cluster in test_clusters]
    )
    contingency = np.zeros((len(np.unique(test_clusters)), 10), dtype=int)
    for cluster_row, cluster in enumerate(np.unique(test_clusters)):
        contingency[cluster_row] = np.bincount(
            test_labels[test_clusters == cluster], minlength=10
        )
    final_results["clustering"] = {
        "selected_configuration": selection["clustering"],
        "test_ari": float(adjusted_rand_score(test_labels, test_clusters)),
        "test_nmi": float(normalized_mutual_info_score(test_labels, test_clusters)),
        "test_cluster_to_class_contingency": contingency.tolist(),
        "test_cluster_ids": [int(cluster) for cluster in np.unique(test_clusters)],
        "training_majority_mapping": cluster_to_class,
        "majority_mapped_test_accuracy": float(
            accuracy_score(test_labels, mapped_cluster_predictions)
        ),
        "svm_vs_mapped_clustering_disagreement_rate": float(
            np.mean(svm_predictions != mapped_cluster_predictions)
        ),
    }
    disagreement_by_class = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        mask = test_labels == class_index
        disagreement_by_class.append(
            {
                "class": class_name,
                "count": int(mask.sum()),
                "disagreement_rate": float(
                    np.mean(svm_predictions[mask] != mapped_cluster_predictions[mask])
                ),
                "svm_accuracy": float(
                    np.mean(svm_predictions[mask] == test_labels[mask])
                ),
                "mapped_cluster_accuracy": float(
                    np.mean(mapped_cluster_predictions[mask] == test_labels[mask])
                ),
            }
        )
    final_results["clustering"]["disagreement_by_class"] = disagreement_by_class

    (RESULTS / "final_test.json").write_text(
        json.dumps(final_results, indent=2), encoding="utf-8"
    )
    print(f"\nFinal test results saved to {(RESULTS / 'final_test.json').resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="One-time final test evaluation after validation-only model selection."
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    run_final_evaluation(args.batch_size)


if __name__ == "__main__":
    main()
