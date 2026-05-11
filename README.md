# Lung Cancer Classification — LUNA16

**Seminar 2 · Fuzzy Logic & Neural Networks · UTS · 12 May 2026**

Patient-level lung nodule classification on the LUNA16 challenge dataset, with a focus on
*honest* evaluation methodology — patient-level splits, confidence intervals, and an
ablation study isolating each architectural improvement.

---

## Headline Results

| Metric | v1 (ResNet-50) | v2 (ResNet-50 + CBAM + Focal + MixUp + TTA) | v3 (DenseNet-121 + CBAM + Focal + MixUp + TTA) |
|---|---|---|---|
| Accuracy | **0.9738** | 0.9732 | 0.9692 |
| Precision | **0.9800** | 0.9784 | 0.9791 |
| Recall (Sensitivity) | 0.9866 | **0.9874** | 0.9815 |
| Specificity | **0.9288** | 0.9228 | 0.9258 |
| **F1-Score** | **0.9833** | 0.9829 | 0.9803 |
| **MCC** | **0.9233** | 0.9212 | 0.9103 |
| ROC-AUC | **0.9929** | 0.9917 | 0.9896 |
| PR-AUC | **0.9978** | 0.9966 | 0.9960 |

Evaluated on **subsets 8 + 9** of LUNA16 — 1 528 patches across 176 unseen patients.
At this test-set size, the 95 % CI on F1 is ±0.008, so v1 / v2 / v3 are
**statistically indistinguishable** — but v2 is the chosen deployment model
(see §4 below for why).

---

## 1 · Project Arc

The project follows a clear scientific narrative:

1. **First attempt — IQ-OTH/NCCD dataset.** A ResNet-50 baseline achieved
   **99 %+ accuracy** on the IQ-OTH/NCCD CT slice dataset. *Too good to be true.*
2. **Investigation — patient-level data leakage.** Pixel-correlation analysis
   between train and test splits revealed near-identical adjacent CT slices in
   both sets. The "amazing" results were an artefact of leakage. See
   `leakage_analysis/` for the full evidence.
3. **Pivot to LUNA16.** Switched to LUNA16 (888 patients, candidate-based
   annotations) with a strict patient-level split: subsets 0–6 → train,
   subset 7 → val, subsets 8 + 9 → test.
4. **Three model variants.** Trained v1 (baseline), v2 (with CBAM attention,
   Focal Loss, MixUp augmentation, and Test-Time Augmentation), and v3 (v2 with
   the backbone swapped to DenseNet-121).
5. **Ablation study.** Tutor's suggestion: isolate each v2 improvement on top of
   v1 to measure its individual contribution.

---

## 2 · Dataset & Split

| Split | Source | Patients | Patches |
|---|---|---|---|
| train | subsets 0–6 | 623 | 4 972 |
| val | subset 7 | 89 | 585 |
| test | subsets 8 + 9 | 176 | 1 528 |

Patch extraction: 50 mm physical window → 64 × 64 pixels, HU-windowed to
`[-1000, +400]`. Negatives sampled at 3:1 ratio per patient.

LUNA16 raw data and extracted patches are **not** in this repo (~115 GB and ~31 MB
respectively). See [§ Reproduce](#5--reproduce-from-scratch).

---

## 3 · Methodology

All three variants share: ResNet-50 / DenseNet-121 pretrained on ImageNet,
OneCycleLR (max LR = 5 × 10⁻⁴), AdamW (wd = 1 × 10⁻⁴), mixed precision (AMP),
gradient clipping (norm = 1.0), early stopping on val F1 (patience = 10),
and a weighted random sampler for the mild 1:4 class imbalance.

### Per-version differences

| Component | v1 | v2 | v3 |
|---|---|---|---|
| Backbone | ResNet-50 | ResNet-50 | DenseNet-121 |
| Parameters | 23.5 M | 24.0 M | 7.0 M |
| CBAM attention | ✗ | ✓ | ✓ |
| Loss | Weighted CE | Focal Loss (α = 0.25, γ = 2) | Focal Loss |
| MixUp augmentation | ✗ | ✓ (β = 0.4, 50 % of batches) | ✓ |
| Test-Time Augmentation | ✗ | ✓ (5-view averaged) | ✓ |
| Epochs | 30 | 25 | 25 |

---

## 4 · Ablation Study — Why v2 (Even Though v1 + TTA Wins on Paper)

The tutor's ablation: take v1 and add **one** improvement at a time.

| Variant | F1 | MCC | ROC-AUC | Δ F1 vs v1 |
|---|---|---|---|---|
| v1 (baseline) | 0.9833 | 0.9233 | 0.9929 | — |
| v1 + CBAM | 0.9795 | 0.9063 | 0.9912 | −0.0038 |
| v1 + Focal | 0.9807 | 0.9128 | 0.9907 | −0.0026 |
| v1 + MixUp | 0.9804 | 0.9096 | 0.9884 | −0.0029 |
| **v1 + TTA** | **0.9854** | **0.9329** | **0.9934** | **+0.0021** |
| v2 (all four) | 0.9829 | 0.9212 | 0.9917 | −0.0004 |

**Key finding:** at this dataset size (~5 000 train patches) the regularisation
tricks (CBAM, Focal, MixUp) individually drift inside the noise band. Only TTA
gives a measurable improvement. **But each component addresses a different
ResNet-50 limitation**, and combining them in v2 produces the methodologically
complete model — the one with interpretability (CBAM heatmap), calibration
(TTA), imbalance handling (Focal), and out-of-distribution robustness (MixUp +
Focal + CBAM together).

We recommend **v2** for deployment for these qualitative reasons; v1 + TTA is a
fine fallback if simpler infrastructure is needed.

---

## 5 · Reproduce From Scratch

```bash
# 1. Get the raw LUNA16 dataset (~115 GB) from
#    https://luna16.grand-challenge.org/Download/
#    Place subsets 0–9 + candidates_V2.csv + annotations.csv in ./luna16/
#
# 2. Set up dependencies
pip install torch torchvision SimpleITK scikit-learn numpy pandas pillow \
            matplotlib imageio[ffmpeg] jupyter
#
# 3. Extract patches (~6 minutes)
python _extract_full_optionB.py
#
# 4. Train all three models + 3-way comparison notebook
jupyter nbconvert --to notebook --execute luna16_pipeline_full.ipynb \
        --output luna16_pipeline_full.ipynb --ExecutePreprocessor.timeout=10800
#
# 5. Run the ablation study
python _run_ablation.py
#
# 6. Re-render the 60-second demo video
python _make_demo_video.py
#
# 7. Re-build the PowerPoint deck (Node.js + pptxgenjs)
npm install -g pptxgenjs
node _build_seminar2_pptx.js
```

Total reproduction time on a single RTX 4060 (laptop): **~3 hours**.

---

## 6 · Repo Layout

```
.
├── README.md                          ← this file
├── .gitignore
│
├── Lung_Cancer_Classification_Seminar2.pptx   ← the deck
├── Lung_Cancer_Classification_Seminar2.pdf    ← PDF mirror
├── demo.mp4                                   ← 60-second live-inference video
├── uts logo.png
│
├── Topic_ Information about Seminar 2 (12 May 2026).pdf  ← the brief
│
├── luna16_pipeline_full.ipynb         ← MAIN notebook (all three variants trained, graphs inline)
├── lung_cancer_cnn.ipynb              ← early IQ-OTH/NCCD work (leakage discovery)
│
├── _extract_full_optionB.py           ← LUNA16 → patches extraction
├── _run_ablation.py                   ← isolated single-component experiments
├── _resume_ablation.py                ← recovery after a cuDNN crash
├── _retrain_v1_mixup.py
├── _leakage_analysis.py               ← pixel-correlation evidence
├── _make_demo_video.py                ← generates demo.mp4
├── _build_seminar2_pptx.js            ← generates the deck
├── _build_full_notebook.py            ← builds luna16_pipeline_full.ipynb
│
├── luna16_train.py                    ← canonical training script
├── luna16_evaluate.py                 ← evaluation with 9-metric suite
├── luna16_extract_patches.py          ← earlier extractor (3-subset)
│
├── luna16_runs_full/
│   ├── comparison_report.json         ← v1 / v2 / v3 final test metrics
│   ├── v1_full/history.json
│   ├── v2_full/
│   │   ├── best.pth                   ← THE deployment checkpoint
│   │   └── history.json
│   └── v3_full/history.json
│
├── luna16_runs_ablation/
│   ├── ablation_report.json           ← v1 + each component, full metric suite
│   ├── v1_cbam/history.json
│   ├── v1_focal/history.json
│   └── v1_mixup/history.json
│
├── luna16_output/                     ← early evaluation artefacts (PNGs + JSON)
├── luna16_output_v2/                  (subsets 7/8/9 only)
├── luna16_output_v3/
│
└── leakage_analysis/                  ← the leakage discovery on IQ-OTH/NCCD
    ├── leakage_report.json
    ├── leakage_summary.txt
    ├── fig1_similarity_distributions.png
    ├── fig2_near_duplicate_frames.png
    ├── fig3_leakage_mechanism.png
    └── fig4_impact_metrics.png
```

---

## 7 · Team

| Member | ID | Role | Contribution |
|---|---|---|---|
| **Huu Thuc Tran** | 26164741 | Leader | Dataset acquisition + extraction; patient-leakage analysis on IQ-OTH/NCCD; LUNA16 patch pipeline; implementation, training, evaluation of v1, v2, v3; ablation study; 9-metric comparison; report and slide content. |
| Alex | 26131779 | Member | Slide design (Seminar 1) and initial Seminar 2 draft; early literature review on lung cancer detection. |
| Rabiul | TBC | Member | Literature review on lung-nodule classification methods. |

---

## 8 · Acknowledgements

- LUNA16 challenge organisers (luna16.grand-challenge.org) for the dataset.
- Woo et al. (ECCV 2018) for CBAM.
- Lin et al. (ICCV 2017) for Focal Loss.
- Zhang et al. (ICLR 2018) for MixUp.

---

*Prepared for Seminar 2, 12 May 2026 — University of Technology Sydney.*
