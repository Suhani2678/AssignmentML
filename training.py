import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _classification_pass(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += batch_size
    return {"loss": total_loss / count, "accuracy": correct / count}


def fit_classifier(
    model,
    train_loader,
    validation_loader,
    epochs,
    device,
    output_dir,
    learning_rate=0.001,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    best_accuracy = -1.0
    best_path = output_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        train_metrics = _classification_pass(model, train_loader, device, optimizer)
        validation_metrics = _classification_pass(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"epoch {epoch}/{epochs} "
            f"train_acc={row['train_accuracy']:.4f} "
            f"val_acc={row['validation_accuracy']:.4f}",
            flush=True,
        )
        if row["validation_accuracy"] > best_accuracy:
            best_accuracy = row["validation_accuracy"]
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "metrics": row},
                best_path,
            )

    _write_history(output_dir / "history.csv", history)
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return model, history, checkpoint


def add_gaussian_noise(images, sigma, generator=None):
    noise = torch.randn(
        images.shape,
        dtype=images.dtype,
        device=images.device,
        generator=generator,
    )
    return (images + sigma * noise).clamp(0.0, 1.0)


def _autoencoder_pass(model, loader, device, optimizer=None, noise_sigma=0.0, seed=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    count = 0
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            inputs = (
                add_gaussian_noise(images, noise_sigma, generator)
                if noise_sigma > 0
                else images
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            reconstruction = model(inputs)
            loss = F.mse_loss(reconstruction, images)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            count += images.size(0)
    return total_loss / count


def fit_autoencoder(
    model,
    train_loader,
    validation_loader,
    epochs,
    device,
    output_dir,
    learning_rate=0.001,
    train_noise_sigma=0.0,
    validation_noise_sigma=0.0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    best_loss = float("inf")
    best_path = output_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        train_loss = _autoencoder_pass(
            model, train_loader, device, optimizer, train_noise_sigma
        )
        validation_loss = _autoencoder_pass(
            model,
            validation_loader,
            device,
            noise_sigma=validation_noise_sigma,
            seed=9100 + epoch,
        )
        row = {
            "epoch": epoch,
            "train_mse": train_loss,
            "validation_mse": validation_loss,
        }
        history.append(row)
        print(
            f"epoch {epoch}/{epochs} train_mse={train_loss:.6f} "
            f"val_mse={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "metrics": row},
                best_path,
            )

    _write_history(output_dir / "history.csv", history)
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return model, history, checkpoint


def collect_predictions(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probabilities = torch.softmax(logits, dim=1).numpy()
    return logits.numpy(), probabilities, labels


def collect_embeddings(model, loader, device, autoencoder=False):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            if autoencoder:
                representations = model.encode(images)
            else:
                representations = model.encode(images)
            embeddings.append(representations.cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


def _write_history(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
