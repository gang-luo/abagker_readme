<<<<<<< HEAD
# AbAgKer: A Unified Semi-Supervised Framework for Antigen-Antibody Binding Affinity and Kinetics Prediction

## Abstract
To the best of our knowledge, we are the first to achieve full K-value prediction for antigen-antibody interactions (covering Kd, koff and kon), providing a comprehensive tool for antibody screening and analysis.
=======
# AbAgKer: Antibody–Antigen Affinity Prediction with Structured Cross-Representation Learning

## Abstract
AbAgKer is a deep learning framework for antibody–antigen affinity prediction that integrates pretrained protein language representations with task-specific interaction modeling. The pipeline encodes antibody heavy/light chains and antigen sequences, injects structural priors such as CDR-sensitive masking and sequence-derived side features, and performs bidirectional cross-attention to capture inter-molecular dependencies. The training objective is formulated as supervised regression over binding-related endpoints (e.g., affinity/Kd), with optional auxiliary balancing losses for mixture-style routing modules. The codebase is implemented in PyTorch Lightning and supports deterministic training, multi-configuration experiments, and fold-based data splits. Beyond raw prediction, the architecture is designed for analysis-oriented workflows: attention maps, pooled token representations, and modular encoder blocks can be inspected to study where interaction signals emerge. This repository targets reproducible antibody engineering research where transparent data flow, explicit tensor semantics, and configurable training protocols are required for publication-grade experimentation.
>>>>>>> codex/complete-repository-refactor-with-comments-and-readme-s2d5co

## Method Overview
The method follows four stages:
1. **Sequence encoding**: heavy chain, light chain, and antigen sequences are tokenized and converted into contextual embeddings using pretrained encoders.
<<<<<<< HEAD
2. **Antibody modeling**: antibody tokens are refined by Cattention, where part of the attention heads only focus on CDR positions.
3. **Antigen compression**: long antigen token sequences are pooled into fixed-length representations via learned token weighting (convolutional and SSF-guided).
4. **Cross-interaction fusion**: antibody and antigen streams exchange information with symmetric cross-attention before regression heads produce outputs.
=======
2. **Antibody modeling**: antibody tokens are refined by CDR-aware self-attention, where part of the attention heads can focus on CDR positions.
3. **Antigen compression**: long antigen token sequences are pooled into fixed-length representations via learned token weighting (convolutional and SSF-guided).
4. **Cross-interaction fusion**: antibody and antigen streams exchange information with symmetric cross-attention before regression heads produce affinity-related outputs.
>>>>>>> codex/complete-repository-refactor-with-comments-and-readme-s2d5co

## Architecture Description
- **Training entrypoint**: `main_wandb.py` provides config-driven model/data instantiation, deterministic setup, logger/callback registration, and Trainer execution.
- **Lightning wrapper**: `taming/models/AbAgKer_newLLM.py` defines `AbAgKerTrainer`, including `training_step`, `validation_step`, metric tracking, optimizer/scheduler setup, and feature assembly.
<<<<<<< HEAD
- **Antibody encoder**: `taming/modules/autoencoder/AbModule.py` contains Cattention module and token poolers.
=======
- **Antibody encoder**: `taming/modules/autoencoder/AbModule.py` contains CDR-aware attention layers and token poolers.
>>>>>>> codex/complete-repository-refactor-with-comments-and-readme-s2d5co
- **Antigen poolers**: `taming/modules/autoencoder/AgModule.py` provides multiple antigen sequence reduction variants.
- **Cross modules**: `taming/modules/autoencoder/AbAgCross.py` implements bidirectional co-attention blocks.
- **Losses/metrics**: `taming/modules/losses/loss_Ab.py` and `taming/modules/metrics/metrics.py` define optimization targets and evaluation criteria.

## Data Format Explanation
Training/validation data are JSON lists where each sample includes:
<<<<<<< HEAD
- `pdb`: complex name.
- `H`, `L`: antibody heavy/light chain sequences.
- `X`: antigen sequence.
- `AbAgA`: supervised affinity target (used as primary regression signal).
- `AbAgI`, `AbAgAoff`: additional interaction/kinetic fields used by specific settings.

The dataloader utilities are defined in `taming/data/PM_Data.py` and `taming/data/utils.py`, with custom collation in `custom_collate`.

=======
- `pdb`: complex identifier.
- `H`, `L`: antibody heavy/light chain sequences.
- `X`: antigen sequence.
- `AbAgA`: supervised affinity target (used as primary regression signal).
- `AbAgI`, `AbAgAoff`: additional kinetic/auxiliary fields used by specific settings.

The dataloader utilities are defined in `taming/data/PM_Data.py` and `taming/data/utils.py`, with custom collation in `custom_collate`.


## Data Processing Pipeline
The preprocessing workflow under `data_pipline/` converts raw public resources into train-ready JSON splits and auxiliary tensors. The process is notebook-driven and reproducible at the directory level.

### Stage A: Source Extraction
- **SabDab parsing**: `data_pipline/sabdab/SabDab.ipynb`
  - Reads SabDab metadata (`sabdab_summary_all.tsv`) and structure archive (`SabDab.zip`).
  - Extracts antibody/antigen chain sequences from PDB entries.
  - Exports normalized records to `data_pipline/sabdab/result/`.
- **SKEMPI cleaning**: `data_pipline/skempi/skempi_clean_2509.ipynb`
  - Reads mutation/affinity tables and PDB mapping files.
  - Applies chain-role normalization and mutation consistency checks.
  - Produces curated records under `data_pipline/skempi/result/`.

### Stage B: Unified Sample Construction
- **Merge and split orchestration**: `data_pipline/AbAgKer_all/collectall_andsplit.ipynb`
  - Merges SabDab and SKEMPI-derived records.
  - Generates random and PDB-aware 5-fold train/test splits.
  - Writes fold artifacts to `data_pipline/AbAgKer_all/split_fivefold/`.

### Stage C: Auxiliary Feature Generation
- **SSF features**: `data_pipline/AbAgKer_all/ssf/ssf_get.ipynb`
  - Computes per-residue sequence-side features used by antigen pooling modules.
- **CDR masks/features**: `data_pipline/AbAgKer_all/cdrs_ssf/cdrs_compute.ipynb`
  - Uses ANARCI-based numbering to derive CDR region indicators.
  - Exports tensors consumed by the antibody CDR-aware attention pipeline.

### Runtime Linkage
Generated JSON and feature tensors are mirrored into runtime directories (`data/`, `data/data_aug/`) and consumed by `taming/data/PM_Data.py` during training and evaluation.

>>>>>>> codex/complete-repository-refactor-with-comments-and-readme-s2d5co
## Training Instructions
1. Prepare a YAML config (examples are provided in `any_tests/config_1120*` and related folders).
2. Launch training with:
```bash
python main_wandb.py -t True -b path/to/config.yaml --save-dir logs
```
3. For deterministic behavior, set `--seed` explicitly (default is already fixed in the script).
4. Resume from an existing run by passing `--resume path/to/logdir_or_ckpt`.

## Inference Instructions
For checkpoint-based inference/prediction:
1. Load the trained checkpoint through the corresponding config.
2. Reuse the same dataset schema for test JSON input.
3. Execute Lightning test/predict flow (or notebook workflows in `notebook/mutation_prediction/`).

Recommended practice:
- Keep preprocessing consistent with training (tokenization, padding lengths, SSF/CDR feature extraction).
- Validate output scale against the training target normalization policy.

## Configuration Explanation
Configuration is OmegaConf/YAML-based and typically includes:
- `model`: target class path and model hyperparameters.
- `data`: dataset class paths and file locations.
- `lightning.trainer`: accelerator/devices/precision/training schedule.
- `opt_config`: optimizer, gradient clipping, warmup/cosine scheduling.
- `loss_config`: objective weights and optional auxiliary-loss controls.

<<<<<<< HEAD
=======
Command-line arguments can override YAML values via dotlist syntax.

>>>>>>> codex/complete-repository-refactor-with-comments-and-readme-s2d5co
## File Structure Overview
```text
.
├── main_wandb.py                     # Training entrypoint (Lightning + configs)
├── taming/
│   ├── models/                       # LightningModule wrappers
│   ├── modules/
│   │   ├── autoencoder/              # Encoders, poolers, cross-attention blocks
│   │   ├── losses/                   # Task losses
│   │   └── metrics/                  # Evaluation metrics
│   └── data/                         # Dataset classes and collation utilities
├── any_tests/                        # Example experiment configs
├── data/                             # Fold splits and processed JSON data
├── data_pipline/                     # Data preparation artifacts/scripts
└── notebook/                         # Analysis and prediction notebooks
```

## Reproducibility Notes
- Use fixed seeds and deterministic backend settings in `main_wandb.py`.
- Record exact config snapshots saved under each run directory (`logs/.../configs`).
- Keep pretrained model versions and tokenizer files unchanged across runs.
- For fold-based evaluation, do not mix train/test JSON splits across folds.
- Report metrics with the same monitor key used for checkpoint selection.

<<<<<<< HEAD
<!-- ## Citation
=======
## Citation
>>>>>>> codex/complete-repository-refactor-with-comments-and-readme-s2d5co
If you use this repository, please cite the corresponding paper (replace placeholder metadata):

```bibtex
@inproceedings{abagker2026,
  title     = {AbAgKer: Cross-Representation Learning for Antibody--Antigen Affinity Prediction},
  author    = {Author, A. and Author, B. and Author, C.},
  booktitle = {Proceedings of Conference Name},
  year      = {2026},
  publisher = {Publisher}
}
<<<<<<< HEAD
``` -->
=======
```
>>>>>>> codex/complete-repository-refactor-with-comments-and-readme-s2d5co

## License
License information is not yet finalized. Add the appropriate open-source license text (e.g., MIT, Apache-2.0, or custom academic license) before public release.
