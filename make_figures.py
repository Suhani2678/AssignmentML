import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _read_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def plot_q1_learning():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model_name, label in (("clean_mlp", "MLP"), ("clean_cnn", "CNN")):
        rows = _read_csv(RESULTS / "checkpoints" / model_name / "history.csv")
        epochs = [int(row["epoch"]) for row in rows]
        axes[0].plot(
            epochs,
            [float(row["train_accuracy"]) for row in rows],
            marker="o",
            label=f"{label} train",
        )
        axes[0].plot(
            epochs,
            [float(row["validation_accuracy"]) for row in rows],
            marker="s",
            linestyle="--",
            label=f"{label} validation",
        )
    axes[0].set_title("Parameter-matched clean models")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=8)
    rows = _read_csv(RESULTS / "checkpoints" / "overfit_mlp_1000" / "history.csv")
    epochs = [int(row["epoch"]) for row in rows]
    axes[1].plot(
        epochs,
        [float(row["train_accuracy"]) for row in rows],
        marker="o",
        label="1,000-image train subset",
    )
    axes[1].plot(
        epochs,
        [float(row["validation_accuracy"]) for row in rows],
        marker="s",
        linestyle="--",
        label="5,000-image validation",
    )
    axes[1].set_title("High-capacity MLP trajectory")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "q1_learning_curves.png", dpi=180)
    plt.close(fig)


def plot_q2_conditions():
    results = _read_json(RESULTS / "final_test.json")["classification"]
    conditions = ("clean", "scrambled", "grayscale")
    models = ("mlp", "cnn")
    x = np.arange(len(conditions))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for offset, model in enumerate(models):
        metrics = [results[f"{condition}_{model}"] for condition in conditions]
        axes[0].bar(
            x + (offset - 0.5) * width,
            [item["accuracy"] for item in metrics],
            width,
            label=model.upper(),
        )
        axes[1].plot(
            conditions,
            [item["mean_confidence_incorrect"] for item in metrics],
            marker="o",
            label=model.upper(),
        )
    axes[0].set_xticks(x, conditions)
    axes[0].set_ylabel("Official test accuracy")
    axes[0].set_title("Clean, scrambled, and grayscale")
    axes[0].legend()
    axes[1].set_ylabel("Mean confidence on errors")
    axes[1].set_title("Error confidence")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "q2_test_conditions.png", dpi=180)
    plt.close(fig)


def plot_q3_autoencoders():
    dims = (32, 128, 512)
    validation_mse = []
    for dim in dims:
        rows = _read_csv(RESULTS / "checkpoints" / f"clean_ae_{dim}" / "history.csv")
        validation_mse.append(
            min(float(row["validation_mse"]) for row in rows)
        )
    test_metrics = _read_json(RESULTS / "autoencoder_test_metrics.json")
    test_mse = [
        test_metrics["clean_reconstruction_mse_by_bottleneck"][str(dim)]
        for dim in dims
    ]
    probes = test_metrics["frozen_encoder_linear_probe_test_accuracy"]
    probe_accuracy = [probes[str(dim)] for dim in dims]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(dims, validation_mse, marker="o", label="Validation MSE")
    axes[0].plot(dims, test_mse, marker="s", linestyle="--", label="Test MSE")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Bottleneck dimension")
    axes[0].set_ylabel("Pixel MSE")
    axes[0].set_title("Reconstruction")
    axes[0].legend()
    axes[1].plot(dims, probe_accuracy, marker="o")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Frozen bottleneck dimension")
    axes[1].set_ylabel("Linear-probe test accuracy")
    axes[1].set_title("Class information")
    fig.tight_layout()
    fig.savefig(FIGURES / "q3_reconstruction_and_class_info.png", dpi=180)
    plt.close(fig)


def plot_q4_svm():
    rows = _read_csv(RESULTS / "svm_sweep.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, kernel in zip(axes, ("linear", "rbf", "poly")):
        selected = sorted(
            (row for row in rows if row["kernel"] == kernel),
            key=lambda row: float(row["C"]),
        )
        c_values = [float(row["C"]) for row in selected]
        validation_accuracy = [float(row["validation_accuracy"]) for row in selected]
        support_fraction = [
            float(row["support_vector_fraction"]) for row in selected
        ]
        axis.plot(c_values, validation_accuracy, marker="o", label="Val. accuracy")
        axis.set_xscale("log")
        axis.set_xlabel("C")
        axis.set_title(f"{kernel.upper()} kernel")
        axis.set_ylabel("Validation accuracy")
        secondary = axis.twinx()
        secondary.plot(
            c_values,
            support_fraction,
            marker="s",
            linestyle="--",
            color="tab:orange",
            label="Support-vector fraction",
        )
        secondary.set_ylabel("Support-vector fraction")
        lines, labels = axis.get_legend_handles_labels()
        second_lines, second_labels = secondary.get_legend_handles_labels()
        axis.legend(lines + second_lines, labels + second_labels, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "q4_svm_validation_sweep.png", dpi=180)
    plt.close(fig)


def plot_q4_contingency():
    metrics = _read_json(RESULTS / "final_test.json")["clustering"]
    matrix = np.asarray(metrics["test_cluster_to_class_contingency"])
    clusters = metrics["test_cluster_ids"]
    fig, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(matrix, aspect="auto", cmap="Blues")
    axis.set_xticks(np.arange(10), CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(clusters)), [str(c) for c in clusters])
    axis.set_xlabel("Test class")
    axis.set_ylabel("Cluster")
    axis.set_title("Test cluster-to-class contingency")
    fig.colorbar(image, ax=axis, label="Number of images")
    fig.tight_layout()
    fig.savefig(FIGURES / "q4_cluster_contingency.png", dpi=180)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_q1_learning()
    plot_q2_conditions()
    plot_q3_autoencoders()
    plot_q4_svm()
    plot_q4_contingency()
    print(f"Figures saved under {FIGURES.resolve()}")


if __name__ == "__main__":
    main()
