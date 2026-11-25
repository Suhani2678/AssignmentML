# CIFAR-10 Assignment 3

Local-only experiment and report files for the MLP, CNN, autoencoder, SVM, and
clustering questions. No GitHub remote or push is configured.

## Reproduction

Use Python 3.12. Install the pinned CPU dependencies from the project directory:

```powershell
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Run the phases in order:

```powershell
python run_training.py
python run_selection.py
python run_final_evaluation.py
python make_figures.py
python generate_report.py
```

The training phase downloads CIFAR-10 into `data/`, creates and saves the fixed
stratified split, and writes model checkpoints and train/validation histories.
The selection phase uses training and validation data only. It fixes the SVM,
clustering, and linear-probe settings and does not construct an official test
loader. Run final evaluation only after the selection outputs are fixed; the
test phase saves results in `results/final_test.json` and refuses to overwrite
an existing final-test report.

All transformations are deterministic and shared between training and the
corresponding validation/test condition. The spatial-scramble permutation uses
the assignment seed, `26001`.
