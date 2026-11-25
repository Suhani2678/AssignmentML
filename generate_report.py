import csv
import html
import json
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from scipy.stats import spearmanr


ROOT = Path(".")
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
OUTPUT = ROOT / "CIFAR10_Assignment_3_Report.pdf"
STUDENT_NAME = "ML Assignment"
ROLL_NUMBER = "M26CSA001"
CLASS_SHORT_NAMES = (
    "air",
    "auto",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

PAGE_W, PAGE_H = letter
LEFT = RIGHT = 42
TOP = 40
BOTTOM = 38
GUTTER = 18
COLUMN_W = (PAGE_W - LEFT - RIGHT - GUTTER) / 2
CONTENT_H = PAGE_H - TOP - BOTTOM

styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=18,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="AuthorLine",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        alignment=TA_CENTER,
        spaceAfter=5,
    )
)
styles.add(
    ParagraphStyle(
        name="Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="Subsection",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=10,
        spaceBefore=4,
        spaceAfter=1,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="BodySmall",
        parent=styles["BodyText"],
        fontName="Times-Roman",
        fontSize=8.2,
        leading=10,
        spaceAfter=3,
        alignment=4,
    )
)
styles.add(
    ParagraphStyle(
        name="CaptionSmall",
        parent=styles["BodyText"],
        fontName="Times-Italic",
        fontSize=7,
        leading=8.2,
        spaceBefore=1,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="TableCell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=6.8,
        leading=7.8,
    )
)
styles.add(
    ParagraphStyle(
        name="TableHeader",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=6.8,
        leading=7.8,
        textColor=colors.white,
    )
)


def read_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def para(text, style="BodySmall"):
    return Paragraph(text, styles[style])


def escaped(text):
    return html.escape(str(text))


def percent(value):
    return f"{100 * float(value):.1f}%"


def number(value):
    return f"{float(value):.4f}"


def make_table(rows, widths, font_size=6.8):
    converted = []
    cell_style = ParagraphStyle(
        name="DynamicTableCell",
        parent=styles["TableCell"],
        fontSize=font_size,
        leading=font_size + 1,
    )
    header_style = ParagraphStyle(
        name="DynamicTableHeader",
        parent=styles["TableHeader"],
        fontSize=font_size,
        leading=font_size + 1,
    )
    for row_index, row in enumerate(rows):
        style = header_style if row_index == 0 else cell_style
        converted.append([Paragraph(str(cell), style) for cell in row])
    table = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9aa4ad")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def image_flowable(path, width=COLUMN_W - 4, max_height=2.2 * inch):
    from PIL import Image as PILImage

    path = Path(path)
    if not path.exists():
        return para(f"[Figure not generated: {escaped(path.name)}]", "CaptionSmall")
    with PILImage.open(path) as image:
        image_width, image_height = image.size
    scale = min(width / image_width, max_height / image_height)
    image = Image(str(path), width=image_width * scale, height=image_height * scale)
    image.hAlign = "CENTER"
    return image


def page_chrome(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#59636e"))
    canvas.drawString(LEFT, PAGE_H - 24, "CIFAR-10 | Assignment 3")
    canvas.drawRightString(PAGE_W - RIGHT, PAGE_H - 24, ROLL_NUMBER)
    canvas.setStrokeColor(colors.HexColor("#b7c0c8"))
    canvas.line(LEFT, PAGE_H - 29, PAGE_W - RIGHT, PAGE_H - 29)
    canvas.drawCentredString(PAGE_W / 2, 22, str(doc.page))
    canvas.restoreState()


def build_report():
    required = (
        RESULTS / "final_test.json",
        RESULTS / "selection.json",
        RESULTS / "autoencoder_selection.json",
        RESULTS / "svm_sweep.csv",
        RESULTS / "linear_probe_sweep.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run training, validation-only selection, final test evaluation, and figure generation first: "
            + ", ".join(missing)
        )

    test = read_json(RESULTS / "final_test.json")
    selection = read_json(RESULTS / "selection.json")
    ae_selection = read_json(RESULTS / "autoencoder_selection.json")
    ae_test = read_json(RESULTS / "autoencoder_test_metrics.json")
    model_specs = {
        row["experiment"]: int(row["parameters"])
        for row in read_csv(RESULTS / "model_specs.csv")
    }
    classification = test["classification"]
    q1_mlp = classification["clean_mlp"]
    q1_cnn = classification["clean_cnn"]
    q1_history_mlp = read_csv(RESULTS / "checkpoints" / "clean_mlp" / "history.csv")
    q1_history_cnn = read_csv(RESULTS / "checkpoints" / "clean_cnn" / "history.csv")
    overfit_history = read_csv(
        RESULTS / "checkpoints" / "overfit_mlp_1000" / "history.csv"
    )
    q1_gap = q1_cnn["accuracy"] - q1_mlp["accuracy"]
    if q1_gap > 0.01:
        q1_interpretation = (
            f"The CNN exceeds the matched MLP by {100 * q1_gap:.1f} percentage points on the "
            "official test set, supporting the locality/weight-sharing explanation over raw "
            "parameter count as the sole cause."
        )
    elif q1_gap < -0.01:
        q1_interpretation = (
            f"The MLP exceeds the matched CNN by {100 * -q1_gap:.1f} percentage points on the "
            "official test set; this run does not support a CNN advantage from locality alone."
        )
    else:
        q1_interpretation = (
            f"The test gap is only {100 * q1_gap:.1f} percentage points, so this run provides "
            "little evidence that locality alone explains an MLP deficit."
        )
    overfit_best_index = max(
        range(len(overfit_history)),
        key=lambda index: float(overfit_history[index]["validation_accuracy"]),
    )
    overfit_best = overfit_history[overfit_best_index]
    late_train_gain = float(overfit_history[-1]["train_accuracy"]) - float(
        overfit_best["train_accuracy"]
    )
    late_val_change = float(overfit_history[-1]["validation_accuracy"]) - float(
        overfit_best["validation_accuracy"]
    )
    if late_train_gain > 0.02 and late_val_change < -0.01:
        memorization_interpretation = (
            "After the validation peak, subset training accuracy continues to rise while "
            "validation accuracy falls; this trajectory supports useful structure being learned "
            "before later memorization."
        )
    elif late_train_gain > 0.02:
        memorization_interpretation = (
            "Training accuracy continues to rise after the validation peak, but validation does "
            "not materially fall; the run shows growing fit but gives weaker evidence for a clear "
            "memorization onset."
        )
    else:
        memorization_interpretation = (
            "Training does not materially improve after the validation peak, so this run does not "
            "clearly isolate a later memorization phase."
        )
    q1a_rows = [
        ["Model", "Params", "Best val", "Test acc."],
    ]
    for name, metrics, history in (
        ("MLP", q1_mlp, q1_history_mlp),
        ("CNN", q1_cnn, q1_history_cnn),
    ):
        best_row = max(history, key=lambda row: float(row["validation_accuracy"]))
        q1a_rows.append(
            [
                name,
                f"{model_specs['clean_' + name.lower()]:,}",
                percent(best_row["validation_accuracy"]),
                percent(metrics["accuracy"]),
            ]
        )

    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=LEFT,
        rightMargin=RIGHT,
        topMargin=TOP,
        bottomMargin=BOTTOM,
        title="MLPs, CNNs, Autoencoders, SVMs and Clustering on CIFAR-10",
        author=STUDENT_NAME,
    )
    frames = [
        Frame(LEFT, BOTTOM, COLUMN_W, CONTENT_H, id="left", leftPadding=2, rightPadding=2),
        Frame(
            LEFT + COLUMN_W + GUTTER,
            BOTTOM,
            COLUMN_W,
            CONTENT_H,
            id="right",
            leftPadding=2,
            rightPadding=2,
        ),
    ]
    doc.addPageTemplates([PageTemplate(id="two-column", frames=frames, onPage=page_chrome)])
    story = [
        para("MLPs, CNNs, Autoencoders, SVMs and Clustering on CIFAR-10", "ReportTitle"),
        para(
            f"{escaped(STUDENT_NAME)} &nbsp;|&nbsp; Roll: {ROLL_NUMBER}<br/>"
            "Repository: local-only; not pushed to GitHub &nbsp;|&nbsp; "
            "Commit hash: not created",
            "AuthorLine",
        ),
        para(
            "<b>Abstract.</b> We compare parameter-matched dense and convolutional classifiers, "
            "test the effect of spatial scrambling and grayscale conversion, study reconstruction "
            "and frozen representations from three convolutional autoencoders, and evaluate "
            "kernel SVMs and unsupervised clustering on a fixed CNN representation. All model "
            "selection uses the validation split; final results below use the official test set.",
        ),
        para("Dataset and common protocol", "Section"),
        para(
            "CIFAR-10 contains 50,000 training images and a separate official 10,000-image test set. "
            "We stratified the training portion into 45,000 train and 5,000 validation examples "
            "using seed 26001 (digits of M26CSA001). Each class contributes 4,500 train and 500 "
            "validation examples. The saved split is "
            "<font name='Courier'>data/split_seed_26001.npz</font>. Inputs are 32x32 RGB images, "
            "converted to tensors and channel-normalized with the standard CIFAR-10 training "
            "statistics. No pretrained models or external image data are used. The test loader is "
            "constructed only in the final evaluation phase.",
        ),
        para("Q1. Why MLPs can underperform", "Section"),
        para("Q1(a): spatial locality versus parameter count", "Subsection"),
        para(
            "A flattened MLP does not encode that nearby pixels form local edges and textures; "
            "it must learn those relationships independently at each pixel location. To separate "
            "this inductive-bias explanation from a simple parameter-count explanation, the "
            "one-hidden-layer MLP and two-stage CNN were matched to approximately 1.005M "
            "trainable parameters: 3072-326-10 for the MLP and 3x32/32x64 convolutions, "
            "pooling, 4096-240-10 for the CNN. Both saw the same 45,000 images, Adam "
            "(learning rate 0.001), batch size 128, five epochs, and validation-based checkpoint "
            "selection. This is a "
            "parameter- and update-budget comparison, not a wall-clock comparison.",
        ),
        make_table(q1a_rows, [42, 62, 58, 60]),
        Spacer(1, 3),
        para(
            q1_interpretation
            + " Matching parameters and update count reduces, but does not eliminate, optimization "
            "and compute differences.",
        ),
        image_flowable(FIGURES / "q1_learning_curves.png", max_height=1.65 * inch),
        para(
            "Fig. 1. Training/validation trajectories for the clean parameter-matched models and "
            "the high-capacity memorization run.",
            "CaptionSmall",
        ),
        para("Q1(b): learning before memorization", "Subsection"),
        para(
            f"The overfit MLP (3072-1024-10, {model_specs['overfit_mlp_1000']:,} parameters) trains "
            "for 20 epochs on a fixed, class-balanced 1,000-image subset of the training split; all epochs are shown "
            "against the untouched validation split. At the final epoch, subset train accuracy is "
            f"{percent(overfit_history[-1]['train_accuracy'])} and validation accuracy is "
            f"{percent(overfit_history[-1]['validation_accuracy'])}. The trajectory, rather than "
            f"the final checkpoint alone, is used to distinguish useful early learning from later "
            f"memorization. Validation peaks at epoch {int(overfit_best['epoch'])} "
            f"({percent(overfit_best['validation_accuracy'])}); from that point to the final epoch, "
            f"training changes by {100 * late_train_gain:+.1f} points and validation by "
            f"{100 * late_val_change:+.1f} points. {memorization_interpretation}",
        ),
        para("Q2. CNN versus MLP and input structure", "Section"),
        para(
            "The predeclared fairness criterion is equal trainable parameters (about 1.005M), "
            "identical optimizer, batch size 128, five epochs, examples, and checkpoint rule. For spatial "
            "disturbance, the same fixed permutation of 1,024 RGB pixel-vectors is applied to every "
            "image at train and evaluation time; the pixel values are preserved. The third "
            "condition converts images to grayscale before training and testing. Before training, "
            "we expected scrambling to hurt the CNN more than the MLP if locality explains the "
            "clean gap, and grayscale to hurt if color carries useful class information.",
        ),
        para(
            "The clean MLP and CNN checkpoints are reused from Q1(a); scrambled and grayscale "
            "conditions are retrained from scratch with the same architecture and budget.",
        ),
    ]

    cnn_scramble_change = (
        classification["scrambled_cnn"]["accuracy"]
        - classification["clean_cnn"]["accuracy"]
    )
    mlp_scramble_change = (
        classification["scrambled_mlp"]["accuracy"]
        - classification["clean_mlp"]["accuracy"]
    )
    cnn_gray_change = (
        classification["grayscale_cnn"]["accuracy"]
        - classification["clean_cnn"]["accuracy"]
    )
    q2_locality_interpretation = (
        "The CNN loses more accuracy than the MLP under scrambling, as predicted by the locality "
        "hypothesis."
        if cnn_scramble_change < mlp_scramble_change - 0.01
        else "The CNN does not lose more accuracy than the MLP under scrambling, so the prediction "
        "that local structure is the dominant explanation is not supported."
    )
    q2_color_interpretation = (
        "The grayscale change is substantial, consistent with useful color information."
        if cnn_gray_change < -0.02
        else "The grayscale change is modest or positive, suggesting that color is not a dominant "
        "cue for this CNN under the tested setup."
    )
    q2_rows = [
        [
            "Input condition",
            f"MLP ({model_specs['clean_mlp']:,})",
            f"CNN ({model_specs['clean_cnn']:,})",
            "CNN error conf.",
        ]
    ]
    for condition in ("clean", "scrambled", "grayscale"):
        q2_rows.append(
            [
                condition,
                percent(classification[f"{condition}_mlp"]["accuracy"]),
                percent(classification[f"{condition}_cnn"]["accuracy"]),
                number(classification[f"{condition}_cnn"]["mean_confidence_incorrect"]),
            ]
        )
    story.extend(
        [
            make_table(q2_rows, [67, 55, 55, 62]),
            Spacer(1, 3),
            image_flowable(FIGURES / "q2_test_conditions.png", max_height=1.6 * inch),
            para(
                "Fig. 2. Official test accuracy and mean confidence on incorrect CNN/MLP predictions.",
                "CaptionSmall",
            ),
            para(
                f"Scrambling changes CNN accuracy by {100 * cnn_scramble_change:+.1f} percentage "
                f"points and MLP accuracy by {100 * mlp_scramble_change:+.1f} points. "
                f"{q2_locality_interpretation} Grayscale changes CNN accuracy by "
                f"{100 * cnn_gray_change:+.1f} points. {q2_color_interpretation} Error confidence "
                "and per-class accuracy are included in the saved test "
                "metrics; high confidence on incorrect predictions indicates a calibration weakness, "
                "not merely a larger error count.",
            ),
            para("Q3. Autoencoders: reconstruction and class information", "Section"),
            para(
                "All convolutional autoencoders use the same encoder/decoder apart from the "
                "bottleneck dimension (32, 128, or 512): three stride-2 convolutional encoder stages "
                "(3-16-32-64 channels), a vector bottleneck, and mirrored transposed-convolution "
                "decoder. They are trained without labels for five epochs using Adam (learning rate "
                "0.001, batch size 128) with pixel MSE. The clean model is selected solely by the "
                "lowest validation MSE; test reconstruction errors for all three are reported only "
                "after choices are fixed. Standardized LinearSVC probes use 10,000 stratified training "
                "embeddings; C in (0.01, 0.1, 1, 10) is selected by validation accuracy.",
            ),
        ]
    )
    ae_rows = [
        ["Bottleneck", "Params", "Val MSE", "Test MSE", "Probe C", "Probe test"]
    ]
    ae_test_mse = ae_test["clean_reconstruction_mse_by_bottleneck"]
    ae_test_probe = ae_test["frozen_encoder_linear_probe_test_accuracy"]
    probe_selection = selection["linear_probes"]
    q3_spearman = float(
        spearmanr(
            [ae_test_mse[str(dim)] for dim in (32, 128, 512)],
            [ae_test_probe[str(dim)] for dim in (32, 128, 512)],
        ).statistic
    )
    for dim in (32, 128, 512):
        history = read_csv(RESULTS / "checkpoints" / f"clean_ae_{dim}" / "history.csv")
        val_mse = min(float(row["validation_mse"]) for row in history)
        ae_rows.append(
            [
                str(dim),
                f"{model_specs[f'clean_ae_{dim}']:,}",
                number(val_mse),
                number(ae_test_mse[str(dim)]),
                str(probe_selection[str(dim)]["C"]),
                percent(ae_test_probe[str(dim)]),
            ]
        )
    story.append(
        KeepTogether(
            [
                make_table(ae_rows, [40, 51, 43, 43, 35, 42]),
                Spacer(1, 3),
                image_flowable(
                    FIGURES / "q3_reconstruction_and_class_info.png",
                    max_height=1.65 * inch,
                ),
                para(
                    "Fig. 3. Reconstruction errors and test accuracy of a linear probe trained on "
                    "frozen encoder features.",
                    "CaptionSmall",
                ),
            ]
        )
    )
    selected_dim = ae_selection["selected_bottleneck"]
    q3_metrics = ae_test["test_reconstruction_mse"]
    denoise_classification = ae_test["classification"]
    corrupted_accuracy = denoise_classification["corrupted_images"]["accuracy"]
    clean_ae_accuracy = denoise_classification["clean_ae_reconstructions"]["accuracy"]
    denoising_accuracy = denoise_classification["denoising_ae_reconstructions"]["accuracy"]
    denoising_gain = denoising_accuracy - corrupted_accuracy
    if denoising_gain > 0.02:
        denoising_conclusion = (
            "The denoising reconstruction improves the fixed classifier by more than two points, "
            "so it recovers some task-relevant information."
        )
    elif denoising_gain > 0:
        denoising_conclusion = (
            "The denoising reconstruction gives a small positive classifier gain; recovery is "
            "limited rather than complete."
        )
    else:
        denoising_conclusion = (
            "The denoising reconstruction does not improve the fixed classifier, despite optimizing "
            "pixel reconstruction."
        )
    if q3_spearman < -0.5:
        q3_relation = (
            "Across these three capacities, lower test reconstruction MSE tends to accompany "
            "higher probe accuracy, though the three-point trend is descriptive."
        )
    elif q3_spearman > 0.5:
        q3_relation = (
            "Across these three capacities, lower test reconstruction MSE does not correspond to "
            "higher probe accuracy; pixel fidelity and class information diverge in this run."
        )
    else:
        q3_relation = (
            "Across these three capacities, the rank association is weak, so reconstruction quality "
            "does not reliably order class information."
        )
    story.extend(
        [
            para(
                f"Validation selects bottleneck {selected_dim}; its official test reconstruction "
                f"MSE is {number(q3_metrics['clean_reconstruction_mse'])}. For denoising, "
                "independent Gaussian noise (sigma=0.15, clipped to [0,1]) is added to clean "
                "training images, and the clean image is the target; labels are ignored. On the "
                "test set, the corrupted-input MSE is "
                f"{number(q3_metrics['corrupted_input_mse'])}, the selected clean AE reconstruction "
                f"MSE is {number(q3_metrics['clean_ae_reconstruction_from_corruption_mse'])}, and "
                "the denoising AE reconstruction MSE is "
                f"{number(q3_metrics['denoising_ae_reconstruction_mse'])}.",
            ),
            KeepTogether(
                [
                    para(
                        f"Using the clean CNN reused from Q2(a), official-test accuracy is "
                        f"{percent(q1_cnn['accuracy'])} on clean images, "
                        f"{percent(corrupted_accuracy)} on corrupted images, "
                        f"{percent(clean_ae_accuracy)} after the clean AE, and "
                        f"{percent(denoising_accuracy)} after the denoising AE. "
                        f"{denoising_conclusion}",
                    ),
                    para(
                        "The standardized frozen-feature probes in Table 3 achieve test accuracy "
                        f"{percent(ae_test_probe['32'])}, {percent(ae_test_probe['128'])}, and "
                        f"{percent(ae_test_probe['512'])} for bottlenecks 32, 128, and 512; no "
                        f"encoder fine-tuning is used. Spearman rho between test reconstruction "
                        f"MSE and probe accuracy is {q3_spearman:+.2f}. {q3_relation}",
                    ),
                ]
            ),
            para("Q4. Bonus: SVMs and clustering", "Section"),
            para(
                "The fixed representation is the 240-dimensional hidden embedding from the "
                "clean CNN first trained in Q1(a) and reused in Q2(a). The SVM uses a stratified, "
                "fixed 5,000-example subset of the 45,000 "
                "training embeddings, selected before the sweep; all 15 settings use that same "
                "subset. C is swept over 0.01, 0.1, 1, 10, and 100 with linear, RBF (gamma=scale), "
                "and degree-3 polynomial kernels. Hyperparameters are selected by validation "
                "accuracy; ties prefer lower C, then linear/RBF/poly.",
            ),
        ]
    )
    svm_sweep_rows = read_csv(RESULTS / "svm_sweep.csv")
    svm_rows = [["Kernel", "C", "Train acc.", "Val acc.", "SV fraction"]]
    for row in svm_sweep_rows:
        svm_rows.append(
            [
                row["kernel"],
                row["C"],
                percent(row["training_accuracy"]),
                percent(row["validation_accuracy"]),
                percent(row["support_vector_fraction"]),
            ]
        )
    story.extend(
        [
            make_table(svm_rows, [46, 34, 52, 48, 58]),
            Spacer(1, 3),
            image_flowable(FIGURES / "q4_svm_validation_sweep.png", max_height=1.7 * inch),
            para(
                "Fig. 4. Validation accuracy and training support-vector fraction across C for each kernel.",
                "CaptionSmall",
            ),
        ]
    )
    svm_result = test["svm"]
    svm_behavior = []
    for kernel in ("linear", "rbf", "poly"):
        curve = sorted(
            (row for row in svm_sweep_rows if row["kernel"] == kernel),
            key=lambda row: float(row["C"]),
        )
        best_validation = max(curve, key=lambda row: float(row["validation_accuracy"]))
        largest_c = curve[-1]
        generalization_gap = (
            float(largest_c["training_accuracy"])
            - float(largest_c["validation_accuracy"])
        )
        validation_fall = (
            float(best_validation["validation_accuracy"])
            - float(largest_c["validation_accuracy"])
        )
        if validation_fall > 0.01 and generalization_gap > 0.03:
            observation = (
                f"{kernel}: C={largest_c['C']} has a {100 * generalization_gap:.1f}-point "
                f"train-validation gap and is {100 * validation_fall:.1f} points below the "
                "kernel's validation peak (overfitting at high C)."
            )
        elif float(best_validation["C"]) == float(largest_c["C"]):
            observation = (
                f"{kernel}: validation peaks at the largest tested C={largest_c['C']}; "
                "no high-C validation downturn is observed in this sweep."
            )
        else:
            observation = (
                f"{kernel}: validation peaks at C={best_validation['C']} with a "
                f"{100 * generalization_gap:.1f}-point high-C train-validation gap."
            )
        svm_behavior.append(observation)
    story.extend(
        [
            para(
                f"The selected {selection['svm']['kernel']} SVM (C={selection['svm']['C']}) "
                f"has {percent(svm_result['test_accuracy'])} official test accuracy. Increasing C "
                "penalizes training errors more strongly. Observed sweep behavior: "
                + " ".join(svm_behavior),
            ),
            para(
                "Three difficult class pairs were selected by their test confusion counts after "
                "the SVM was fixed. The plot shows two training support vectors and two closest "
                "test examples per pair, with OVO decision scores; the detailed examples and "
                "indices are retained in final_test.json.",
            ),
            image_flowable(FIGURES / "svm_difficult_pairs.png", max_height=2.0 * inch),
            para(
                "Fig. 5. Support vectors and near-boundary test images for the three most confused pairs.",
                "CaptionSmall",
            ),
        ]
    )
    cluster_selection_rows = [["k", "Validation silhouette"]]
    for row in read_csv(RESULTS / "cluster_sweep.csv"):
        cluster_selection_rows.append(
            [row["n_clusters"], number(row["validation_silhouette"])]
        )
    cluster_metrics = test["clustering"]
    contingency_rows = [["ID"] + list(CLASS_SHORT_NAMES)]
    for cluster, counts in zip(
        cluster_metrics["test_cluster_ids"],
        cluster_metrics["test_cluster_to_class_contingency"],
    ):
        contingency_rows.append([str(cluster)] + [str(value) for value in counts])
    largest_class_gap = max(
        cluster_metrics["disagreement_by_class"],
        key=lambda row: abs(row["svm_accuracy"] - row["mapped_cluster_accuracy"]),
    )
    story.extend(
        [
            para("Validation-only choice of cluster count", "Subsection"),
            make_table(cluster_selection_rows, [42, 105]),
            Spacer(1, 3),
            para("Official-test cluster-to-class contingency", "Subsection"),
            make_table(
                contingency_rows,
                [25] + [(COLUMN_W - 25) / 10] * 10,
                font_size=5.8,
            ),
            Spacer(1, 3),
            image_flowable(FIGURES / "q4_cluster_contingency.png", max_height=1.35 * inch),
            para(
                "Fig. 6. Heatmap view of the same official-test contingency table.",
                "CaptionSmall",
            ),
        ]
    )
    story.extend(
        [
            para(
                f"MiniBatchKMeans was selected by validation silhouette among k={(8, 10, 12)}; "
                f"the chosen k={test['clustering']['selected_configuration']['n_clusters']} was then "
                "fit on all training embeddings without class labels. For mapped accuracy, each "
                "cluster is assigned its majority class using training labels only. Test labels are used only for "
                f"evaluation: ARI={number(test['clustering']['test_ari'])}, "
                f"NMI={number(test['clustering']['test_nmi'])}, majority-mapped test accuracy="
                f"{percent(test['clustering']['majority_mapped_test_accuracy'])}. The SVM and "
                "majority-mapped cluster assignments disagree on "
                f"{percent(test['clustering']['svm_vs_mapped_clustering_disagreement_rate'])} of "
                "test images.",
            ),
            KeepTogether(
                [
                    para(
                        f"The largest class-level gap is for "
                        f"{escaped(largest_class_gap['class'])}: SVM accuracy "
                        f"{percent(largest_class_gap['svm_accuracy'])} versus majority-mapped "
                        f"clustering accuracy "
                        f"{percent(largest_class_gap['mapped_cluster_accuracy'])}, with "
                        f"disagreement on {percent(largest_class_gap['disagreement_rate'])} of "
                        "that class's test images.",
                    )
                ]
            ),
            para("Conclusions", "Section"),
            para(
                "The parameter-matched classifier comparison and fixed spatial permutation test "
                "whether spatial inductive bias, rather than parameter count alone, explains the "
                "CNN/MLP gap. Grayscale isolates the role of color cues. Autoencoder reconstruction "
                "and frozen linear-probe results show whether pixel fidelity tracks label information. "
                "Finally, the selected kernel SVM and unsupervised clustering reveal different "
                "properties of the same embedding: classification boundaries optimize labels, while "
                "clusters reflect geometric density. All conclusions should be read alongside the "
                "reported test evidence, not as claims from accuracy alone.",
            ),
            para("Reproducibility and files", "Section"),
            para(
                "The project is local-only (no GitHub push). Use README.md for exact commands. "
                "The saved split, configurations, checkpoints, histories, validation sweeps, "
                "figures, and final test metrics are in data/ and results/. The assignment asks for "
                "a repository URL and final commit hash; neither is supplied because no remote or "
                "commit was requested.",
            ),
            para(
                "References: A. Krizhevsky, Learning Multiple Layers of Features from Tiny Images, "
                "2009. C. Cortes and V. Vapnik, Support-vector networks, Machine Learning, 1995. "
                "J. MacQueen, Some Methods for Classification and Analysis of Multivariate "
                "Observations, 1967.",
                "CaptionSmall",
            ),
        ]
    )

    doc.build(story)
    pages = len(PdfReader(str(OUTPUT)).pages)
    if pages > 10:
        raise RuntimeError(f"Report has {pages} pages; the assignment limit is 10.")
    print(f"Generated {OUTPUT.resolve()} ({pages} pages).")


if __name__ == "__main__":
    build_report()
