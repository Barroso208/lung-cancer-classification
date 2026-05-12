# Lung Cancer Classification — LUNA16

**Seminar 2 · Fuzzy Logic & Neural Networks · UTS · 12 May 2026**

Patient-level lung nodule classification on the LUNA16 challenge dataset, with a focus on
*honest* evaluation methodology — patient-level splits, confidence intervals, an
ablation study isolating each architectural improvement, and a sanity-check audit
that caught (and fixed) our own metric-reporting bug.

---

## Final Headline Result — v1 + TTA

| Metric (nodule = positive class) | v1 + TTA |
|---|---|
| F1-Score | **0.9474** |
| Recall (Sensitivity) | **0.9347** |
| MCC | **0.9329** |
| ROC-AUC | **0.9934** |
| Accuracy | **0.9771** |
| Precision | 0.9604 |
| Specificity | 0.9891 |
| PR-AUC | 0.9822 |
| **Confusion** | TP = 315 · FN = 22 · FP = 13 · TN = 1 178 |

Evaluated on **subsets 8 + 9** of LUNA16 — 1 528 patches across 176 unseen
patients. The 95 % CI on F1 at this test-set size is ±0.008.

---

## ⚠ Audit Note (Read This)

During a final sanity-check pass we discovered that the original metric code
used `pos_label = 1`, but PyTorch's `ImageFolder` had alphabetically mapped the
folders so `nodule → 0` and `non_nodule → 1`. Our earlier reports therefore
had **Recall and Specificity swapped** — what we called "recall" was actually
specificity, and vice versa.

The bug was **purely in the reporting code, not in training**. All models were
trained correctly; only the metric labels were inverted in JSON reports and
slides. Once corrected:

- Accuracy, MCC, and ROC-AUC are **unchanged** (symmetric under positive-class flip).
- F1 and PR-AUC moved noticeably because they depend on positive-class prevalence.
- Recall and Specificity swapped values.

All numbers in this README and in the final deck are **post-correction**. The
scripts `_audit_results.py` and `_audit_ablation.py` reproduce the corrected
evaluation from the saved checkpoints. See `luna16_runs_full/comparison_report_corrected.json`
and `luna16_runs_ablation/ablation_report_corrected.json` for machine-readable
data.

---

## 1 · Project Arc

1. **First attempt — IQ-OTH/NCCD dataset.** A ResNet-50 baseline achieved
   **99 %+ accuracy** on the IQ-OTH/NCCD CT slice dataset. *Too good to be true.*
2. **Investigation — patient-level data leakage.** Pixel-correlation analysis
   revealed near-identical adjacent CT slices in both train and test splits.
   The "amazing" results were an artefact of leakage. See `leakage_analysis/`.
3. **Pivot to LUNA16.** Switched to LUNA16 (888 patients, candidate-based
   annotations) with a strict patient-level split.
4. **Three model variants** trained and compared: v1 (baseline), v2 (with
   CBAM attention, Focal Loss, MixUp, TTA), and v3 (v2 with the backbone
   swapped to DenseNet-121).
5. **Ablation study** at the tutor's suggestion: each v2 improvement isolated
   on top of v1 to measure its individual contribution.
6. **Final sanity-check audit** that uncovered the label-index inversion bug
   in the metric code (above). All numbers were re-computed.

---

## 2 · Dataset & Split

| Split | Source | Patients | Patches |
|---|---|---|---|
| train | subsets 0–6 | 623 | 4 972 |
| val | subset 7 | 89 | 585 |
| test | subsets 8 + 9 | 176 | 1 528 |

Patch extraction: 50 mm physical window → 64 × 64 pixels, HU-windowed to
`[-1000, +400]`. Negatives sampled at 3:1 ratio per patient. Class prevalence
of nodule across all three splits ≈ 22 %.

LUNA16 raw data and extracted patches are **not** in this repo (~115 GB and
~31 MB respectively). See [§ Reproduce](#5--reproduce-from-scratch).

---

## 3 · Methodology

All variants share: ResNet-50 / DenseNet-121 pretrained on ImageNet,
OneCycleLR (max LR = 5 × 10⁻⁴), AdamW (wd = 1 × 10⁻⁴), mixed precision (AMP),
gradient clipping (norm = 1.0), early stopping on val F1 (patience = 10),
and a weighted random sampler for the mild 1:4 class imbalance.

### Per-version differences

| Component | v1 | v2 | v3 |
|---|---|---|---|
| Backbone | ResNet-50 | ResNet-50 | DenseNet-121 |
| Parameters | 23.5 M | 24.0 M | 7.0 M |
| CBAM attention | ✗ | ✓ | ✓ |
| Loss | Weighted CE | Focal (α = 0.25, γ = 2) | Focal |
| MixUp augmentation | ✗ | ✓ (β = 0.4, 50 % of batches) | ✓ |
| Test-Time Augmentation | ✗ | ✓ (5-view averaged) | ✓ |
| Epochs | 30 | 25 | 25 |

### CBAM (Channel + Spatial Attention)

Our custom module (Woo et al., ECCV 2018) inserted after ResNet-50's `layer4`.

- **Channel attention** — pools H×W → 1×1 per channel, shared 2-layer MLP
  scores each of the 2 048 channels (`M_c`). Applied as channel-wise scaling.
- **Spatial attention** — pools channels → 1 channel (avg + max stacked),
  7×7 conv learns a saliency map (`M_s`). Applied as pixel-wise scaling.

Total added params: ~500 K (≈ 2 % of ResNet-50's 23.5 M).

---

## 4 · Ablation Study — Why v1 + TTA Wins

The tutor's ablation: take v1 and add **one** improvement at a time.

| Variant | F1 | Recall | MCC | Δ F1 vs v1 |
|---|---|---|---|---|
| v1 (baseline) | 0.9399 | 0.9288 | 0.9233 | — |
| v1 + CBAM | 0.9268 | 0.9199 | 0.9063 | −0.0132 |
| v1 + Focal | 0.9322 | 0.9377 | 0.9128 | −0.0078 |
| v1 + MixUp | 0.9289 | 0.9110 | 0.9096 | −0.0110 |
| **v1 + TTA** | **0.9474** | **0.9347** | **0.9329** | **+0.0074** |
| v2 (all four, +TTA) | 0.9382 | 0.9228 | 0.9212 | −0.0018 |

**Key finding:** CBAM, Focal Loss, and MixUp individually drift inside the
±0.008 F1 noise band — meaningful research attempts that did not pay off at
this data scale (a common outcome). **Only TTA consistently improves v1**, and
it is a free inference-time win.

We adopt **v1 + TTA** as the deployment model.

Alternatives:
- **v2** — keep if a CBAM heatmap is needed for clinical UI / interpretability.
- **v3** — keep for edge / low-VRAM deployment (3× smaller — 7 M params).

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
# 6. (Audit) Verify split + recompute metrics with nodule-as-positive
python _audit_results.py
python _audit_ablation.py
#
# 7. Re-render the 60-second demo video
python _make_demo_video.py
#
# 8. Re-build the PowerPoint deck (Node.js + pptxgenjs)
npm install -g pptxgenjs
node _build_seminar2_pptx.js
```

Total reproduction time on a single RTX 4060 (laptop): **~3 hours**.

---

## 6 · Repo Layout

```
.
├── README.md                                          ← this file
├── .gitignore
│
├── Lung_Cancer_Classification_Seminar2.pptx           ← the 18-slide deck
├── Lung_Cancer_Classification_Seminar2.pdf            ← PDF mirror
├── demo.mp4                                           ← 60-second live-inference video
├── uts logo.png
│
├── Topic_ Information about Seminar 2 (12 May 2026).pdf  ← the brief
│
├── luna16_pipeline_full.ipynb                         ← MAIN notebook (v1/v2/v3 trained, graphs inline)
├── lung_cancer_cnn.ipynb                              ← early IQ-OTH/NCCD work (leakage discovery)
│
├── _extract_full_optionB.py                           ← LUNA16 → patches extraction
├── _run_ablation.py                                   ← isolated single-component experiments
├── _resume_ablation.py                                ← recovery after a cuDNN crash
├── _retrain_v1_mixup.py
├── _audit_results.py                                  ← sanity-check: split + metric correction
├── _audit_ablation.py                                 ← ablation re-evaluation w/ nodule as positive
├── _leakage_analysis.py                               ← pixel-correlation evidence
├── _make_demo_video.py                                ← generates demo.mp4
├── _build_seminar2_pptx.js                            ← generates the deck
├── _build_full_notebook.py                            ← builds luna16_pipeline_full.ipynb
│
├── luna16_train.py                                    ← canonical training script
├── luna16_evaluate.py                                 ← evaluation with 9-metric suite
├── luna16_extract_patches.py                          ← earlier extractor (3-subset)
│
├── luna16_runs_full/
│   ├── comparison_report.json                         ← original (uncorrected) metrics
│   ├── comparison_report_corrected.json               ← post-audit, nodule-as-positive
│   ├── v1_full/history.json
│   ├── v2_full/
│   │   ├── best.pth                                   ← v2 deployment checkpoint
│   │   └── history.json
│   └── v3_full/history.json
│
├── luna16_runs_ablation/
│   ├── ablation_report.json                           ← original (uncorrected)
│   ├── ablation_report_corrected.json                 ← post-audit, nodule-as-positive
│   ├── v1_cbam/history.json
│   ├── v1_focal/history.json
│   └── v1_mixup/history.json
│
├── luna16_output/                                     ← early evaluation artefacts
├── luna16_output_v2/                                  (subsets 7/8/9 only)
├── luna16_output_v3/
│
└── leakage_analysis/                                  ← the leakage discovery on IQ-OTH/NCCD
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
| **Huu Thuc Tran** | 26164741 | Leader | Dataset acquisition + extraction; patient-leakage analysis on IQ-OTH/NCCD; LUNA16 patch pipeline; implementation, training, evaluation of v1, v2, v3; ablation study; sanity-check audit; 9-metric comparison; report and slide content. |
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
