import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from cifar10_setup import (
    BATCH_SIZE,
    DATA_DIR,
    SEED,
    make_split_indices,
    make_train_validation_loaders,
    seed_everything,
)
from models import ConvAutoencoder, MLP, SmallCNN
from training import count_parameters, fit_autoencoder, fit_classifier


RESULTS = Path("results")
CHECKPOINTS = RESULTS / "checkpoints"
CLASSIFIER_EPOCHS = 5
OVERFIT_EPOCHS = 20
AUTOENCODER_EPOCHS = 5
NOISE_SIGMA = 0.15


def _save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _record_spec(rows, name, model, condition, epochs, samples):
    rows.append(
        {
            "experiment": name,
            "condition": condition,
            "parameters": count_parameters(model),
            "epochs": epochs,
            "training_samples": samples,
        }
    )


def _make_model(architecture):
    if architecture == "mlp":
        return MLP(hidden_dim=326)
    if architecture == "cnn":
        return SmallCNN(embedding_dim=240)
    raise ValueError(f"Unknown architecture: {architecture}")


def run_training(batch_size, classifier_epochs, overfit_epochs, autoencoder_epochs):
    RESULTS.mkdir(exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configuration = {
        "seed": SEED,
        "data_dir": DATA_DIR,
        "batch_size": batch_size,
        "classifier_epochs": classifier_epochs,
        "overfit_epochs": overfit_epochs,
        "autoencoder_epochs": autoencoder_epochs,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "classifier_loss": "cross_entropy",
        "autoencoder_loss": "pixel_mean_squared_error",
        "noise_sigma": NOISE_SIGMA,
        "device": str(device),
        "parameter_budget": "approximately 1.005 million parameters",
        "q1a_reason_tested": (
            "The flattened MLP lacks a locality/weight-sharing prior; parameter matching controls "
            "for raw parameter count as an alternative explanation."
        ),
        "predeclared_hypotheses": {
            "spatial_scramble": (
                "The CNN will lose more clean-test accuracy than the MLP because convolution "
                "relies on local spatial structure."
            ),
            "grayscale": (
                "If CIFAR-10 color cues matter to this CNN, grayscale training/testing will lower "
                "its accuracy; a small change suggests shape/texture dominates."
            ),
        },
    }
    _save_json(RESULTS / "configuration.json", configuration)
    model_specs = []

    for condition in ("clean", "scrambled", "grayscale"):
        print(f"\n=== Classification condition: {condition} ===", flush=True)
        train_loader, validation_loader = make_train_validation_loaders(
            batch_size=batch_size, condition=condition
        )
        for offset, architecture in enumerate(("mlp", "cnn")):
            seed_everything(SEED + offset + 10 * ("clean", "scrambled", "grayscale").index(condition))
            model = _make_model(architecture).to(device)
            name = f"{condition}_{architecture}"
            output_dir = CHECKPOINTS / name
            _record_spec(
                model_specs,
                name,
                model,
                condition,
                classifier_epochs,
                len(train_loader.dataset),
            )
            model, _, checkpoint = fit_classifier(
                model,
                train_loader,
                validation_loader,
                classifier_epochs,
                device,
                output_dir,
            )
            print(
                f"selected epoch={checkpoint['epoch']} "
                f"validation accuracy={checkpoint['metrics']['validation_accuracy']:.4f}",
                flush=True,
            )
        del train_loader, validation_loader, model

    print("\n=== MLP memorization trajectory ===", flush=True)
    train_loader, validation_loader = make_train_validation_loaders(
        batch_size=batch_size, condition="clean"
    )
    train_indices, _ = make_split_indices(train_loader.dataset.dataset.targets)
    labels = np.asarray(train_loader.dataset.dataset.targets)
    subset_indices, _ = train_test_split(
        train_indices,
        train_size=1_000,
        random_state=SEED,
        stratify=labels[train_indices],
    )
    subset_loader = DataLoader(
        Subset(train_loader.dataset.dataset, subset_indices.tolist()),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )
    seed_everything(SEED + 100)
    memorization_model = MLP(hidden_dim=1024).to(device)
    _record_spec(
        model_specs,
        "overfit_mlp_1000",
        memorization_model,
        "clean",
        overfit_epochs,
        len(subset_loader.dataset),
    )
    fit_classifier(
        memorization_model,
        subset_loader,
        validation_loader,
        overfit_epochs,
        device,
        CHECKPOINTS / "overfit_mlp_1000",
    )
    del train_loader, validation_loader, subset_loader, memorization_model

    print("\n=== Clean convolutional autoencoders ===", flush=True)
    train_loader, validation_loader = make_train_validation_loaders(
        batch_size=batch_size, condition="clean", normalize=False
    )
    autoencoder_dims = (32, 128, 512)
    clean_autoencoder_results = {}
    for bottleneck_dim in autoencoder_dims:
        seed_everything(SEED + bottleneck_dim)
        model = ConvAutoencoder(bottleneck_dim).to(device)
        name = f"clean_ae_{bottleneck_dim}"
        _record_spec(
            model_specs,
            name,
            model,
            "clean",
            autoencoder_epochs,
            len(train_loader.dataset),
        )
        _, _, checkpoint = fit_autoencoder(
            model,
            train_loader,
            validation_loader,
            autoencoder_epochs,
            device,
            CHECKPOINTS / name,
        )
        clean_autoencoder_results[str(bottleneck_dim)] = {
            "validation_mse": checkpoint["metrics"]["validation_mse"],
            "selected_epoch": checkpoint["epoch"],
        }

    selected_bottleneck = min(
        clean_autoencoder_results,
        key=lambda dim: clean_autoencoder_results[dim]["validation_mse"],
    )
    print(f"Selected clean-AE bottleneck: {selected_bottleneck}", flush=True)
    _save_json(
        RESULTS / "autoencoder_selection.json",
        {
            "bottleneck_validation": clean_autoencoder_results,
            "selected_bottleneck": int(selected_bottleneck),
            "selection_rule": "lowest validation reconstruction MSE",
        },
    )

    print("\n=== Denoising autoencoder ===", flush=True)
    seed_everything(SEED + 200)
    denoising_model = ConvAutoencoder(int(selected_bottleneck)).to(device)
    _record_spec(
        model_specs,
        "denoising_ae",
        denoising_model,
        "gaussian_noise_sigma_0.15",
        autoencoder_epochs,
        len(train_loader.dataset),
    )
    fit_autoencoder(
        denoising_model,
        train_loader,
        validation_loader,
        autoencoder_epochs,
        device,
        CHECKPOINTS / "denoising_ae",
        train_noise_sigma=NOISE_SIGMA,
        validation_noise_sigma=NOISE_SIGMA,
    )
    del train_loader, validation_loader, denoising_model

    with (RESULTS / "model_specs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(model_specs[0]))
        writer.writeheader()
        writer.writerows(model_specs)
    print(f"\nTraining complete on {device}. Results are in {RESULTS.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train the CIFAR-10 models; this phase does not load the official test set."
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--classifier-epochs", type=int, default=CLASSIFIER_EPOCHS)
    parser.add_argument("--overfit-epochs", type=int, default=OVERFIT_EPOCHS)
    parser.add_argument("--autoencoder-epochs", type=int, default=AUTOENCODER_EPOCHS)
    args = parser.parse_args()
    run_training(
        args.batch_size,
        args.classifier_epochs,
        args.overfit_epochs,
        args.autoencoder_epochs,
    )


if __name__ == "__main__":
    main()
