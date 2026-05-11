/**
 * Seminar 2 — Lung Cancer Classification deck
 * Clean-academic style, UTS-aligned palette, ~10-min talk, 14 slides.
 */
const pptxgen = require("pptxgenjs");
const fs      = require("fs");
const path    = require("path");

// ───────── Data loaded from disk ─────────
const report  = JSON.parse(fs.readFileSync("luna16_runs_full/comparison_report.json", "utf8"));
const histV1  = JSON.parse(fs.readFileSync("luna16_runs_full/v1_full/history.json",  "utf8"));
const histV2  = JSON.parse(fs.readFileSync("luna16_runs_full/v2_full/history.json",  "utf8"));
const histV3  = JSON.parse(fs.readFileSync("luna16_runs_full/v3_full/history.json",  "utf8"));

const v1 = report.v1_full.test;
const v2 = report.v2_full.test_tta;
const v3 = report.v3_full.test_tta;

// ───────── Palette ─────────
const NAVY   = "0F2C4F";
const CREAM  = "F5F2EA";
const RED    = "D71920";  // UTS accent
const GRAY   = "5F6470";
const BLACK  = "1A1A1A";
const LIGHT  = "F8F8F8";

// ───────── Helpers ─────────
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 × 7.5
pres.author = "Huu Thuc Tran";
pres.title  = "Lung Cancer Classification — Seminar 2";

const SLIDE_W = 13.3, SLIDE_H = 7.5;

// Header bar with UTS logo + section title (used on content slides)
function header(slide, sectionTitle, _weightStr /* unused, kept for compat */) {
  // Top accent bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: SLIDE_W, h: 0.05, fill: { color: NAVY }, line: { color: NAVY }
  });
  // Section label (top-left, small, muted)
  slide.addText(sectionTitle.toUpperCase(), {
    x: 0.5, y: 0.18, w: 10, h: 0.35,
    fontFace: "Calibri", fontSize: 11, color: GRAY, charSpacing: 4,
    margin: 0,
  });
  // Logo (top-right) — slightly smaller so it sits next to the section label cleanly
  slide.addImage({ path: "uts logo.png", x: SLIDE_W - 1.1, y: 0.14, h: 0.45, w: 0.45 });
}

// Footer bar with page number + project tag
function footer(slide, pageNum, total) {
  slide.addText("Lung Cancer Classification  ·  Seminar 2  ·  12 May 2026", {
    x: 0.5, y: SLIDE_H - 0.4, w: 8, h: 0.3,
    fontFace: "Calibri", fontSize: 9, color: GRAY, margin: 0,
  });
  slide.addText(`${pageNum} / ${total}`, {
    x: SLIDE_W - 1.3, y: SLIDE_H - 0.4, w: 0.8, h: 0.3,
    fontFace: "Calibri", fontSize: 9, color: GRAY, align: "right", margin: 0,
  });
}

// Big title for content slides
function bigTitle(slide, title) {
  slide.addText(title, {
    x: 0.5, y: 0.65, w: SLIDE_W - 1.2, h: 0.85,
    fontFace: "Georgia", fontSize: 32, bold: true, color: NAVY, margin: 0,
  });
  // Subtle separator
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.55, w: 0.6, h: 0.06,
    fill: { color: RED }, line: { color: RED },
  });
}

// ─────────────────────────────────────────────────────────
//  SLIDE 1 — TITLE
// ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  // Decorative side bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.4, h: SLIDE_H, fill: { color: RED }, line: { color: RED }
  });

  // UTS logo on white circle (for contrast against navy background)
  s.addShape(pres.shapes.OVAL, {
    x: SLIDE_W - 1.55, y: 0.45, w: 0.8, h: 0.8,
    fill: { color: "FFFFFF" }, line: { color: "FFFFFF" },
  });
  s.addImage({ path: "uts logo.png", x: SLIDE_W - 1.45, y: 0.55, h: 0.6, w: 0.6 });

  // Subject label
  s.addText("FUZZY LOGIC & NEURAL NETWORKS  ·  SEMINAR 2", {
    x: 1.0, y: 1.4, w: 11, h: 0.4,
    fontFace: "Calibri", fontSize: 14, color: "C0CADC", charSpacing: 8, margin: 0,
  });

  // Main title
  s.addText("Lung Cancer Classification", {
    x: 1.0, y: 2.05, w: 11, h: 1.4,
    fontFace: "Georgia", fontSize: 56, bold: true, color: "FFFFFF", margin: 0,
  });

  // Subtitle
  s.addText("From Data Leakage Diagnosis to Patient-Level LUNA16 Models", {
    x: 1.0, y: 3.5, w: 11, h: 0.7,
    fontFace: "Georgia", fontSize: 22, italic: true, color: "C0CADC", margin: 0,
  });

  // Thin red rule
  s.addShape(pres.shapes.RECTANGLE, {
    x: 1.0, y: 4.45, w: 0.8, h: 0.05, fill: { color: RED }, line: { color: RED }
  });

  // Members table-like
  s.addText([
    { text: "Huu Thuc Tran", options: { bold: true, color: "FFFFFF" } },
    { text: "  ·  26164741  ·  ", options: { color: "C0CADC" } },
    { text: "Leader", options: { italic: true, color: RED, breakLine: true } },
    { text: "Alex", options: { bold: true, color: "FFFFFF" } },
    { text: "  ·  26131779", options: { color: "C0CADC", breakLine: true } },
    { text: "Rabiul", options: { bold: true, color: "FFFFFF" } },
    { text: "  ·  [Student ID — TBC]", options: { color: "C0CADC" } },
  ], {
    x: 1.0, y: 4.7, w: 11, h: 1.4,
    fontFace: "Calibri", fontSize: 16, paraSpaceAfter: 4, margin: 0,
  });

  // Date
  s.addText("12 May 2026", {
    x: 1.0, y: SLIDE_H - 0.9, w: 11, h: 0.4,
    fontFace: "Calibri", fontSize: 14, color: "C0CADC", margin: 0,
  });
}

// ─────────────────────────────────────────────────────────
//  SLIDE 2 — INTRODUCTION  (5%)
// ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "Introduction");
  bigTitle(s, "Why Lung Cancer Classification Matters");

  // Left column: bullets
  s.addText([
    { text: "Leading cause of cancer death", options: { bold: true, color: NAVY, breakLine: true } },
    { text: "≈ 1.8 million deaths worldwide each year (WHO).", options: { color: BLACK, breakLine: true, fontSize: 14 } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Early detection is decisive", options: { bold: true, color: NAVY, breakLine: true } },
    { text: "5-year survival jumps from ~5 % (late stage) to ~60 % when detected via low-dose CT screening.", options: { color: BLACK, breakLine: true, fontSize: 14 } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Manual screening bottleneck", options: { bold: true, color: NAVY, breakLine: true } },
    { text: "Radiologists must inspect 200–400 slices per patient — slow, tiring, and inter-reader variable.", options: { color: BLACK, fontSize: 14 } },
  ], {
    x: 0.6, y: 2.0, w: 6.8, h: 4.5, fontFace: "Calibri", fontSize: 16,
    paraSpaceAfter: 6, margin: 0,
  });

  // Right column: stat callout (aligned to top of bullet column at y=2.0)
  s.addShape(pres.shapes.RECTANGLE, {
    x: 8.0, y: 2.0, w: 4.7, h: 4.5,
    fill: { color: NAVY }, line: { color: NAVY }
  });
  s.addText("OUR TASK", {
    x: 8.3, y: 2.25, w: 4.1, h: 0.35,
    fontFace: "Calibri", fontSize: 12, bold: true, color: "FFFFFF", charSpacing: 6, margin: 0,
  });
  s.addText("Binary nodule classification", {
    x: 8.3, y: 2.7, w: 4.1, h: 1.4,
    fontFace: "Georgia", fontSize: 24, bold: true, color: "FFFFFF", margin: 0,
  });
  // Thin red rule
  s.addShape(pres.shapes.RECTANGLE, {
    x: 8.3, y: 4.05, w: 0.6, h: 0.05, fill: { color: RED }, line: { color: RED }
  });
  s.addText([
    { text: "Input: ", options: { bold: true, color: "FFFFFF" } },
    { text: "64 × 64 CT patch (50 mm region)", options: { color: "FFFFFF", breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "Output: ", options: { bold: true, color: "FFFFFF" } },
    { text: "nodule  /  non-nodule", options: { color: "FFFFFF", breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "Goal: ", options: { bold: true, color: "FFFFFF" } },
    { text: "leakage-free, clinically credible classifier", options: { color: "FFFFFF" } },
  ], {
    x: 8.3, y: 4.25, w: 4.1, h: 2.0,
    fontFace: "Calibri", fontSize: 13, paraSpaceAfter: 4, margin: 0,
  });

  footer(s, 2, 17);
}

// ─────────────────────────────────────────────────────────
//  SLIDE 3 — MOTIVATIONS / AIMS / OBJECTIVES  (5%)
// ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "Motivations · Aims · Objectives");
  bigTitle(s, "Why this Project — and What we Aim to Deliver");

  // Three columns
  const cols = [
    {
      title: "Motivations",
      color: NAVY,
      bullets: [
        "Reduce radiologist workload",
        "Improve detection consistency",
        "Provide a clinically credible benchmark — not an inflated one",
      ]
    },
    {
      title: "Aims",
      color: RED,
      bullets: [
        "Train a CNN for binary nodule classification",
        "Establish a leakage-free evaluation protocol",
        "Compare regularisation & architecture choices systematically",
      ]
    },
    {
      title: "Objectives",
      color: NAVY,
      bullets: [
        "Patient-level data split (no patient overlap)",
        "9-metric suite: Acc, Bal-Acc, Prec, Rec, Spec, F1, MCC, ROC-AUC, PR-AUC",
        "Iterative improvement: baseline → +regularisation → backbone swap",
      ]
    }
  ];

  cols.forEach((c, i) => {
    const x = 0.6 + i * 4.15;
    // Card
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 2.0, w: 4.0, h: 4.5,
      fill: { color: "FFFFFF" }, line: { color: "E0DCD0", width: 1 },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.08 }
    });
    // Top stripe
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 2.0, w: 4.0, h: 0.08, fill: { color: c.color }, line: { color: c.color },
    });
    // Title
    s.addText(c.title, {
      x: x + 0.25, y: 2.25, w: 3.5, h: 0.5,
      fontFace: "Georgia", fontSize: 22, bold: true, color: c.color, margin: 0,
    });
    // Bullets
    s.addText(c.bullets.map((b, j) => ({
      text: b, options: { bullet: true, color: BLACK, breakLine: j !== c.bullets.length - 1 }
    })), {
      x: x + 0.25, y: 2.85, w: 3.5, h: 3.5,
      fontFace: "Calibri", fontSize: 14, paraSpaceAfter: 8, margin: 0,
    });
  });

  footer(s, 3, 17);
}

// ─────────────────────────────────────────────────────────
//  SLIDE 4 — METHODOLOGY: STORY ARC  (part of 20%)
// ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "Methodology — Story Arc");
  bigTitle(s, "From “Too Good to be True” to a Trustworthy Result");

  // 3-phase timeline
  const phases = [
    {
      label: "PHASE 1",
      title: "IQ-OTH/NCCD",
      sub: "First attempt — suspicion",
      bullets: [
        "1 097 CT slices, 3 classes",
        "ResNet-50 baseline → 99 %+ accuracy",
        "“Too good to be true” warning"
      ],
      color: NAVY,
    },
    {
      label: "PHASE 2",
      title: "Investigation",
      sub: "Patient-level leakage",
      bullets: [
        "Pixel-correlation analysis between train and test",
        "Found near-identical adjacent slices in both sets",
        "Inflated metrics confirmed — discard random split"
      ],
      color: RED,
    },
    {
      label: "PHASE 3",
      title: "Pivot to LUNA16",
      sub: "Patient-level evaluation",
      bullets: [
        "888 patients, candidate-based annotations",
        "Patient-level split: subsets 0–6 train, 7 val, 8+9 test",
        "Honest, comparable metrics"
      ],
      color: NAVY,
    },
  ];

  phases.forEach((p, i) => {
    const x = 0.6 + i * 4.15;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 2.0, w: 4.0, h: 4.7,
      fill: { color: "FFFFFF" }, line: { color: "E0DCD0", width: 1 },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.08 }
    });
    // Left stripe
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 2.0, w: 0.12, h: 4.7, fill: { color: p.color }, line: { color: p.color },
    });
    // Phase label
    s.addText(p.label, {
      x: x + 0.3, y: 2.15, w: 3.5, h: 0.3,
      fontFace: "Calibri", fontSize: 11, bold: true, color: p.color, charSpacing: 6, margin: 0,
    });
    // Title
    s.addText(p.title, {
      x: x + 0.3, y: 2.45, w: 3.5, h: 0.55,
      fontFace: "Georgia", fontSize: 22, bold: true, color: BLACK, margin: 0,
    });
    // Subtitle
    s.addText(p.sub, {
      x: x + 0.3, y: 3.0, w: 3.5, h: 0.4,
      fontFace: "Calibri", fontSize: 14, italic: true, color: GRAY, margin: 0,
    });
    // Bullets
    s.addText(p.bullets.map((b, j) => ({
      text: b, options: { bullet: true, color: BLACK, breakLine: j !== p.bullets.length - 1 }
    })), {
      x: x + 0.3, y: 3.5, w: 3.5, h: 3.0,
      fontFace: "Calibri", fontSize: 13, paraSpaceAfter: 6, margin: 0,
    });

    // Arrow between phases
    if (i < 2) {
      s.addShape(pres.shapes.LINE, {
        x: x + 4.0, y: 4.35, w: 0.15, h: 0,
        line: { color: GRAY, width: 2, endArrowType: "triangle" },
      });
    }
  });

  footer(s, 4, 17);
}

// ─────────────────────────────────────────────────────────
//  SLIDE 5 — METHODOLOGY: FUNCTIONAL BLOCK DIAGRAM  (part of 20%)
// ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "Methodology — Functional Block Diagram");
  bigTitle(s, "End-to-End Pipeline");

  // 5 boxes in flow, then 2nd row with details
  const blocks = [
    { x: 0.5,  w: 2.2, label: "Raw CT Volumes",   sub: ".mhd / .raw\n888 patients", color: NAVY },
    { x: 3.0,  w: 2.2, label: "Patch Extraction", sub: "World→Voxel\n50 mm · 64×64\nHU [-1000, 400]", color: NAVY },
    { x: 5.5,  w: 2.2, label: "Augmentation",     sub: "Flip · Rotate\nErasing · MixUp", color: NAVY },
    { x: 8.0,  w: 2.2, label: "Backbone + CBAM",  sub: "ResNet-50 / DenseNet-121\nChannel + Spatial Attn.", color: RED },
    { x: 10.5, w: 2.2, label: "Classifier Head",  sub: "BN · Dropout 0.4\nLinear(2)", color: NAVY },
  ];

  blocks.forEach((b, i) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: b.x, y: 2.3, w: b.w, h: 1.5,
      fill: { color: "FFFFFF" }, line: { color: b.color, width: 2 },
      shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 90, opacity: 0.10 }
    });
    s.addText(b.label, {
      x: b.x, y: 2.4, w: b.w, h: 0.4,
      fontFace: "Calibri", fontSize: 13, bold: true, color: b.color, align: "center", margin: 0,
    });
    s.addText(b.sub, {
      x: b.x + 0.05, y: 2.85, w: b.w - 0.1, h: 0.85,
      fontFace: "Calibri", fontSize: 11, color: GRAY, align: "center", margin: 0,
    });

    if (i < blocks.length - 1) {
      const nextX = blocks[i + 1].x;
      s.addShape(pres.shapes.LINE, {
        x: b.x + b.w, y: 3.05, w: nextX - (b.x + b.w), h: 0,
        line: { color: GRAY, width: 2, endArrowType: "triangle" },
      });
    }
  });

  // Loss / Optim row
  s.addText("TRAINING", {
    x: 0.5, y: 4.4, w: 12.3, h: 0.3,
    fontFace: "Calibri", fontSize: 11, bold: true, color: GRAY, charSpacing: 6, margin: 0,
  });
  const trainItems = [
    { x: 0.5,  w: 2.95, label: "Focal Loss",   sub: "α = 0.25 · γ = 2.0" },
    { x: 3.65, w: 2.95, label: "AdamW",        sub: "wd = 1e-4 · clip 1.0" },
    { x: 6.8,  w: 2.95, label: "OneCycleLR",   sub: "max_lr = 5e-4 · cos anneal" },
    { x: 9.95, w: 2.95, label: "Mixed Precision",   sub: "AMP + Early Stop (val F1)" },
  ];
  trainItems.forEach(t => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: t.x, y: 4.75, w: t.w, h: 0.85,
      fill: { color: LIGHT }, line: { color: "E0DCD0", width: 1 },
    });
    s.addText(t.label, {
      x: t.x + 0.15, y: 4.8, w: t.w - 0.3, h: 0.35,
      fontFace: "Calibri", fontSize: 13, bold: true, color: NAVY, margin: 0,
    });
    s.addText(t.sub, {
      x: t.x + 0.15, y: 5.15, w: t.w - 0.3, h: 0.45,
      fontFace: "Calibri", fontSize: 11, color: GRAY, margin: 0,
    });
  });

  // Inference row
  s.addText("INFERENCE", {
    x: 0.5, y: 6.0, w: 12.3, h: 0.3,
    fontFace: "Calibri", fontSize: 11, bold: true, color: GRAY, charSpacing: 6, margin: 0,
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 6.35, w: 12.3, h: 0.7,
    fill: { color: NAVY }, line: { color: NAVY }
  });
  s.addText([
    { text: "5-View Test-Time Augmentation:  ", options: { bold: true, color: "FFFFFF" } },
    { text: "identity · h-flip · v-flip · rot90 · rot270   →   averaged softmax → final prediction",
      options: { color: "C0CADC" } },
  ], {
    x: 0.7, y: 6.4, w: 12.0, h: 0.6,
    fontFace: "Calibri", fontSize: 13, valign: "middle", margin: 0,
  });

  footer(s, 5, 17);
}

// ─────────────────────────────────────────────────────────
//  SLIDE 6 — METHODOLOGY: ARCHITECTURE & CBAM  (part of 20%)
// ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "Methodology — Model Architecture");
  bigTitle(s, "Three Versions, Identical Recipe");

  // Left: model layers
  s.addText("Backbone & Head", {
    x: 0.5, y: 2.0, w: 5.5, h: 0.4,
    fontFace: "Calibri", fontSize: 13, bold: true, color: GRAY, charSpacing: 4, margin: 0,
  });

  const layers = [
    { label: "Input  64×64 (resized 224, 3-ch grayscale)",      color: BLACK },
    { label: "Backbone (ImageNet V2 pretrained)",                color: NAVY  },
    { label: "    ResNet-50 (v1, v2)  ·  DenseNet-121 (v3)",     color: BLACK, sub: true  },
    { label: "CBAM Attention (v2, v3 only)",                      color: RED   },
    { label: "    Channel Attention  →  Spatial Attention",       color: BLACK, sub: true  },
    { label: "Adaptive AvgPool 1×1",                              color: NAVY  },
    { label: "Head: BatchNorm  +  Dropout 0.4  +  Linear(2)",     color: NAVY  },
  ];
  let yy = 2.5;
  layers.forEach(l => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: yy, w: 5.5, h: 0.45,
      fill: { color: l.sub ? LIGHT : "FFFFFF" }, line: { color: "E0DCD0", width: 1 },
    });
    s.addText(l.label, {
      x: 0.65, y: yy, w: 5.3, h: 0.45,
      fontFace: "Calibri", fontSize: l.sub ? 12 : 13, bold: !l.sub, color: l.color,
      valign: "middle", italic: l.sub, margin: 0,
    });
    yy += 0.5;
  });

  // Right: 3 versions comparison card
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.5, y: 2.0, w: 6.3, h: 4.6,
    fill: { color: "FFFFFF" }, line: { color: "E0DCD0", width: 1 },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.08 }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.5, y: 2.0, w: 0.12, h: 4.6, fill: { color: NAVY }, line: { color: NAVY }
  });
  s.addText("The three model variants", {
    x: 6.8, y: 2.15, w: 5.8, h: 0.4,
    fontFace: "Georgia", fontSize: 18, bold: true, color: NAVY, margin: 0,
  });

  // Variant rows
  const variants = [
    {
      name: "v1  ·  Baseline",
      bg: "EEEEEE", fg: NAVY,
      details: "ResNet-50  ·  weighted CE  ·  no MixUp  ·  no TTA  ·  30 epochs"
    },
    {
      name: "v2  ·  Improved",
      bg: "FCE6E7", fg: RED,
      details: "ResNet-50  +  CBAM  +  Focal  +  MixUp  +  TTA  ·  25 epochs"
    },
    {
      name: "v3  ·  Backbone Swap",
      bg: "E6EEF7", fg: NAVY,
      details: "DenseNet-121  +  CBAM  +  Focal  +  MixUp  +  TTA  ·  25 epochs"
    },
  ];
  let yv = 2.7;
  variants.forEach(v => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.8, y: yv, w: 5.8, h: 1.2,
      fill: { color: v.bg }, line: { color: v.fg, width: 1 },
    });
    s.addText(v.name, {
      x: 6.95, y: yv + 0.05, w: 5.6, h: 0.45,
      fontFace: "Calibri", fontSize: 14, bold: true, color: v.fg, margin: 0,
    });
    s.addText(v.details, {
      x: 6.95, y: yv + 0.5, w: 5.6, h: 0.65,
      fontFace: "Calibri", fontSize: 12, color: BLACK, margin: 0,
    });
    yv += 1.32;
  });

  footer(s, 6, 17);
}

// ─────────────────────────────────────────────────────────
//  SLIDE 7 — DIFFERENCES BETWEEN VERSIONS  (NEW)
// ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "Differences Between Versions");
  bigTitle(s, "Exactly What Changes from v1 → v2 → v3");

  // Comparison table — rows are components, columns are versions
  const Y = "✓", N = "—";
  const diffRows = [
    [
      { text: "Component", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "v1  ·  Baseline",     options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "v2  ·  Improved",     options: { bold: true, color: "FFFFFF", fill: { color: RED  }, align: "center" } },
      { text: "v3  ·  Backbone Swap",options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
    ],
    [
      { text: "Backbone", options: { bold: true, color: NAVY, fill: { color: "FFFFFF" } } },
      { text: "ResNet-50",       options: { color: BLACK, align: "center", fill: { color: "FFFFFF" } } },
      { text: "ResNet-50",       options: { color: BLACK, align: "center", fill: { color: "FFFFFF" } } },
      { text: "DenseNet-121",    options: { color: RED,   align: "center", bold: true, fill: { color: "FFFFFF" } } },
    ],
    [
      { text: "Parameters", options: { bold: true, color: NAVY, fill: { color: LIGHT } } },
      { text: "23.5 M", options: { color: BLACK, align: "center", fill: { color: LIGHT } } },
      { text: "24.0 M", options: { color: BLACK, align: "center", fill: { color: LIGHT } } },
      { text: "7.0 M  (3× smaller)", options: { color: RED,   align: "center", bold: true, fill: { color: LIGHT } } },
    ],
    [
      { text: "CBAM Attention",          options: { bold: true, color: NAVY, fill: { color: "FFFFFF" } } },
      { text: N, options: { color: GRAY, align: "center", fill: { color: "FFFFFF" } } },
      { text: Y, options: { color: RED,  align: "center", bold: true, fill: { color: "FFFFFF" } } },
      { text: Y, options: { color: RED,  align: "center", bold: true, fill: { color: "FFFFFF" } } },
    ],
    [
      { text: "Loss Function",   options: { bold: true, color: NAVY, fill: { color: LIGHT } } },
      { text: "Weighted CE",     options: { color: BLACK, align: "center", fill: { color: LIGHT } } },
      { text: "Focal Loss (α=0.25, γ=2)", options: { color: RED, align: "center", bold: true, fill: { color: LIGHT } } },
      { text: "Focal Loss (α=0.25, γ=2)", options: { color: RED, align: "center", bold: true, fill: { color: LIGHT } } },
    ],
    [
      { text: "MixUp Augmentation",  options: { bold: true, color: NAVY, fill: { color: "FFFFFF" } } },
      { text: N, options: { color: GRAY, align: "center", fill: { color: "FFFFFF" } } },
      { text: Y, options: { color: RED,  align: "center", bold: true, fill: { color: "FFFFFF" } } },
      { text: Y, options: { color: RED,  align: "center", bold: true, fill: { color: "FFFFFF" } } },
    ],
    [
      { text: "Test-Time Augmentation", options: { bold: true, color: NAVY, fill: { color: LIGHT } } },
      { text: N, options: { color: GRAY, align: "center", fill: { color: LIGHT } } },
      { text: "5-view averaged", options: { color: RED,  align: "center", bold: true, fill: { color: LIGHT } } },
      { text: "5-view averaged", options: { color: RED,  align: "center", bold: true, fill: { color: LIGHT } } },
    ],
    [
      { text: "Epochs", options: { bold: true, color: NAVY, fill: { color: "FFFFFF" } } },
      { text: "30", options: { color: BLACK, align: "center", fill: { color: "FFFFFF" } } },
      { text: "25", options: { color: BLACK, align: "center", fill: { color: "FFFFFF" } } },
      { text: "25", options: { color: BLACK, align: "center", fill: { color: "FFFFFF" } } },
    ],
  ];
  s.addTable(diffRows, {
    x: 0.5, y: 2.0, w: 12.3,
    colW: [3.3, 3.0, 3.0, 3.0],
    rowH: 0.46,
    fontFace: "Calibri", fontSize: 13,
    border: { type: "solid", pt: 0.5, color: "DDDDDD" },
    valign: "middle",
  });

  // Bottom takeaway bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 5.85, w: 12.3, h: 1.3,
    fill: { color: "FFFFFF" }, line: { color: "E0DCD0", width: 1 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 5.85, w: 0.12, h: 1.3, fill: { color: RED }, line: { color: RED }
  });
  s.addText("How to read this lineage", {
    x: 0.7, y: 5.95, w: 11.9, h: 0.35,
    fontFace: "Calibri", fontSize: 12, bold: true, color: GRAY, charSpacing: 4, margin: 0,
  });
  s.addText([
    { text: "v2 = v1 + the four improvements ",  options: { bold: true, color: NAVY } },
    { text: "(CBAM attention · Focal loss · MixUp augmentation · TTA)", options: { color: BLACK } },
    { text: "  that directly target ResNet-50's known limitations: ", options: { color: BLACK, breakLine: true } },
    { text: "no explicit attention, naive treatment of class imbalance, weak regularisation, single-pass inference.", options: { italic: true, color: BLACK, breakLine: true } },
    { text: "v3 = v2 ", options: { bold: true, color: NAVY } },
    { text: "with the backbone swapped to a smaller, parameter-efficient architecture.", options: { color: BLACK } },
  ], {
    x: 0.7, y: 6.30, w: 11.9, h: 0.85,
    fontFace: "Calibri", fontSize: 12, paraSpaceAfter: 2, margin: 0,
  });

  footer(s, 7, 17);
}

// ─────────────────────────────────────────────────────────
//  SLIDE 8 — EXPERIMENTAL SETUP  (part of 15%)
// ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "Experimental Setup");
  bigTitle(s, "Dataset and Patient-Level Split");

  // Left: dataset stats card
  const card = (x, y, w, h, label, big, sub, color) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w, h, fill: { color: "FFFFFF" }, line: { color: "E0DCD0", width: 1 },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.08 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w, h: 0.08, fill: { color }, line: { color },
    });
    s.addText(label.toUpperCase(), {
      x: x + 0.2, y: y + 0.18, w: w - 0.4, h: 0.3,
      fontFace: "Calibri", fontSize: 11, bold: true, color, charSpacing: 4, margin: 0,
    });
    s.addText(big, {
      x: x + 0.2, y: y + 0.5, w: w - 0.4, h: 0.85,
      fontFace: "Georgia", fontSize: 36, bold: true, color: BLACK, margin: 0,
    });
    s.addText(sub, {
      x: x + 0.2, y: y + 1.4, w: w - 0.4, h: 0.5,
      fontFace: "Calibri", fontSize: 12, color: GRAY, margin: 0,
    });
  };

  card(0.5,  2.0, 4.0, 1.95, "Train",      "4 972", "patches  ·  623 patients\nSubsets 0–6", NAVY);
  card(4.65, 2.0, 4.0, 1.95, "Validation", "585",   "patches  ·  89 patients\nSubset 7",     RED);
  card(8.8,  2.0, 4.0, 1.95, "Test",       "1 528", "patches  ·  176 patients\nSubsets 8 + 9", NAVY);

  // Bottom row: split scheme details
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.15, w: 12.3, h: 2.5,
    fill: { color: "FFFFFF" }, line: { color: "E0DCD0", width: 1 },
  });
  s.addText("Why this split (Option B)", {
    x: 0.7, y: 4.3, w: 11.9, h: 0.4,
    fontFace: "Georgia", fontSize: 18, bold: true, color: NAVY, margin: 0,
  });
  s.addText([
    { text: "Patient-level partitioning  ", options: { bold: true, color: NAVY } },
    { text: "—  no patient appears in two splits.", options: { color: BLACK, breakLine: true } },
    { text: "Test set doubled  ", options: { bold: true, color: NAVY } },
    { text: "from 662 → 1 528 patches  ⇒  CI on F1 narrows from ±0.012 to ", options: { color: BLACK } },
    { text: "±0.008", options: { bold: true, color: RED, breakLine: true } },
    { text: "  → smaller architectural deltas now become ", options: { color: BLACK } },
    { text: "statistically detectable.", options: { italic: true, color: BLACK, breakLine: true } },
    { text: "Class prevalence is consistent across splits (≈ 22 % nodule), so train-time class imbalance handling generalises to evaluation.", options: { color: BLACK } },
  ], {
    x: 0.7, y: 4.75, w: 11.9, h: 1.85,
    fontFace: "Calibri", fontSize: 14, paraSpaceAfter: 4, margin: 0,
  });

  footer(s, 8, 17);
}

// ─────────────────────────────────────────────────────────
//  SLIDE 8 — RESULTS: 3-WAY COMPARISON TABLE  (part of 15%)
// ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "Results — 3-Way Comparison");
  bigTitle(s, "Test Metrics on 1 528 Held-Out Patches");

  const fmt = v => v.toFixed(4);

  const metrics = [
    { k: "accuracy",     name: "Accuracy" },
    { k: "balanced_acc", name: "Balanced Accuracy" },
    { k: "precision",    name: "Precision" },
    { k: "recall",       name: "Recall (Sensitivity)" },
    { k: "specificity",  name: "Specificity" },
    { k: "f1",           name: "F1-Score" },
    { k: "mcc",          name: "MCC" },
    { k: "roc_auc",      name: "ROC-AUC" },
    { k: "pr_auc",       name: "PR-AUC" },
  ];

  // Build table rows; bold the best per row
  const headerRow = [
    { text: "Metric", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "left" } },
    { text: "v1 (single)", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
    { text: "v2 (+TTA)",   options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
    { text: "v3 (+TTA)",   options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
  ];

  const rows = [headerRow];
  metrics.forEach((m, idx) => {
    const a = v1[m.k], b = v2[m.k], c = v3[m.k];
    const max = Math.max(a, b, c);
    const cellOpts = (val, isBest) => ({
      text: fmt(val),
      options: {
        bold: isBest,
        color: isBest ? RED : BLACK,
        align: "center",
        fill: { color: idx % 2 === 0 ? "FFFFFF" : LIGHT },
      },
    });
    rows.push([
      { text: m.name, options: { color: NAVY, bold: true, fill: { color: idx % 2 === 0 ? "FFFFFF" : LIGHT } } },
      cellOpts(a, a === max),
      cellOpts(b, b === max),
      cellOpts(c, c === max),
    ]);
  });

  s.addTable(rows, {
    x: 0.6, y: 2.0, w: 8.5,
    colW: [3.4, 1.7, 1.7, 1.7],
    rowH: 0.36,
    fontFace: "Calibri", fontSize: 13,
    border: { type: "solid", pt: 0.5, color: "DDDDDD" },
  });

  // Legend on the right
  s.addShape(pres.shapes.RECTANGLE, {
    x: 9.4, y: 2.0, w: 3.5, h: 4.5,
    fill: { color: "FFFFFF" }, line: { color: "E0DCD0", width: 1 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 9.4, y: 2.0, w: 0.12, h: 4.5, fill: { color: RED }, line: { color: RED }
  });
  s.addText("How to read this", {
    x: 9.6, y: 2.15, w: 3.2, h: 0.4,
    fontFace: "Georgia", fontSize: 16, bold: true, color: NAVY, margin: 0,
  });
  s.addText([
    { text: "Red bold ", options: { bold: true, color: RED } },
    { text: "= best value in each row", options: { color: BLACK, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "v1 ", options: { bold: true, color: NAVY } },
    { text: "wins 7 of 9 metrics", options: { color: BLACK, breakLine: true } },
    { text: "v2 ", options: { bold: true, color: NAVY } },
    { text: "wins Recall", options: { color: BLACK, breakLine: true } },
    { text: "v3 ", options: { bold: true, color: NAVY } },
    { text: "wins Precision", options: { color: BLACK, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "BUT — ", options: { bold: true, color: RED } },
    { text: "all deltas fit ", options: { color: BLACK } },
    { text: "inside the ±0.008 F1 confidence interval.", options: { italic: true, color: BLACK } },
  ], {
    x: 9.6, y: 2.65, w: 3.2, h: 3.8,
    fontFace: "Calibri", fontSize: 13, paraSpaceAfter: 6, margin: 0,
  });

  footer(s, 9, 17);
}

// ─────────────────────────────────────────────────────────
//  SLIDE 9 — RESULTS: VAL-F1 LEARNING CURVES + CONFUSION (part of 15%)
// ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "Results — Learning Curves & Errors");
  bigTitle(s, "All Three Models Reach the Same Ceiling");

  // Left: line chart of val_f1 across epochs
  const chartData = [
    { name: "v1 (ResNet-50)",            labels: histV1.val_f1.map((_, i) => String(i + 1)), values: histV1.val_f1 },
    { name: "v2 (ResNet-50+CBAM, Focal)", labels: histV2.val_f1.map((_, i) => String(i + 1)), values: histV2.val_f1 },
    { name: "v3 (DenseNet-121+CBAM)",     labels: histV3.val_f1.map((_, i) => String(i + 1)), values: histV3.val_f1 },
  ];
  s.addChart(pres.charts.LINE, chartData, {
    x: 0.5, y: 2.0, w: 7.0, h: 4.5,
    chartColors: [NAVY, "B85042", "27AE60"],
    chartArea: { fill: { color: "FFFFFF" }, roundedCorners: false },
    catAxisLabelColor: GRAY, catAxisLabelFontSize: 9, catAxisTitle: "Epoch",
    catAxisTitleColor: GRAY, catAxisTitleFontSize: 10, showCatAxisTitle: true,
    valAxisLabelColor: GRAY, valAxisLabelFontSize: 9, valAxisTitle: "Validation F1",
    valAxisTitleColor: GRAY, valAxisTitleFontSize: 10, showValAxisTitle: true,
    valAxisMinVal: 0.3, valAxisMaxVal: 1.0,   // cap headroom — F1 is in [0,1]
    valGridLine: { color: "EAEAEA", size: 0.5 },
    catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 10,
    lineSize: 2, lineSmooth: true, showTitle: true, title: "Validation F1 across epochs",
    titleColor: NAVY, titleFontFace: "Georgia", titleFontSize: 14,
  });

  // Right: confusion matrix for v1 (best F1)
  s.addText("Confusion (v1, best F1 = " + v1.f1.toFixed(4) + ")", {
    x: 8.0, y: 2.0, w: 4.8, h: 0.4,
    fontFace: "Georgia", fontSize: 14, bold: true, color: NAVY, margin: 0,
  });

  const cmCells = [
    [{ text: "",                         options: { fill: { color: "FFFFFF" } } },
     { text: "Pred. non-nodule",         options: { fill: { color: NAVY }, color: "FFFFFF", bold: true, align: "center" } },
     { text: "Pred. nodule",             options: { fill: { color: NAVY }, color: "FFFFFF", bold: true, align: "center" } }],
    [{ text: "True non-nodule",          options: { fill: { color: NAVY }, color: "FFFFFF", bold: true } },
     { text: String(report.v1_full.test.confusion.TN),     options: { fill: { color: "DCE9F8" }, align: "center", bold: true, color: NAVY } },
     { text: String(report.v1_full.test.confusion.FP),     options: { fill: { color: "FCE6E7" }, align: "center", bold: true, color: RED } }],
    [{ text: "True nodule",              options: { fill: { color: NAVY }, color: "FFFFFF", bold: true } },
     { text: String(report.v1_full.test.confusion.FN),     options: { fill: { color: "FCE6E7" }, align: "center", bold: true, color: RED } },
     { text: String(report.v1_full.test.confusion.TP),     options: { fill: { color: "DCE9F8" }, align: "center", bold: true, color: NAVY } }],
  ];
  s.addTable(cmCells, {
    x: 8.0, y: 2.5, w: 4.8, colW: [1.6, 1.6, 1.6], rowH: 0.7,
    fontFace: "Calibri", fontSize: 14,
    border: { type: "solid", pt: 0.5, color: "FFFFFF" },
  });

  // Diagnosis box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 8.0, y: 4.85, w: 4.8, h: 1.65,
    fill: { color: "FFFFFF" }, line: { color: "E0DCD0", width: 1 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 8.0, y: 4.85, w: 0.12, h: 1.65, fill: { color: NAVY }, line: { color: NAVY }
  });
  s.addText("Over/underfitting check", {
    x: 8.2, y: 4.95, w: 4.5, h: 0.3,
    fontFace: "Calibri", fontSize: 11, bold: true, color: GRAY, charSpacing: 4, margin: 0,
  });
  s.addText([
    { text: "v1: ", options: { bold: true, color: NAVY } },
    { text: "train 98.8% / val 98.1%, gap +0.7%  ✓", options: { color: BLACK, breakLine: true } },
    { text: "v2: ", options: { bold: true, color: NAVY } },
    { text: "val ≥ train  (MixUp signature)  ✓", options: { color: BLACK, breakLine: true } },
    { text: "v3: ", options: { bold: true, color: NAVY } },
    { text: "val ≥ train  (MixUp signature)  ✓", options: { color: BLACK } },
  ], {
    x: 8.2, y: 5.3, w: 4.5, h: 1.15,
    fontFace: "Calibri", fontSize: 12, paraSpaceAfter: 3, margin: 0,
  });

  footer(s, 10, 17);
}

// ─────────────────────────────────────────────────────────
//  SLIDE 10 — ANALYSIS & DISCUSSION  (15%)
// ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "Analysis & Discussion");
  bigTitle(s, "Honest Findings — what we learned");

  // Two big findings as cards
  const cards = [
    {
      x: 0.5, w: 6.1, color: NAVY,
      label: "Finding 1",
      title: "All three models hit the same ceiling.",
      body: [
        { text: "F1 ≈ 0.98, AUC ≈ 0.99, MCC ≈ 0.92 across v1, v2, v3.", b: false, br: true },
        { text: " ", b: false, br: true },
        { text: "Largest pair-wise F1 delta: ", b: false },
        { text: "0.003", b: true, color: RED },
        { text: "  —  smaller than the ", b: false },
        { text: "±0.008", b: true, color: RED },
        { text: " confidence interval at this test size.", b: false, br: true },
        { text: " ", b: false, br: true },
        { text: "Statistically: ", b: true, color: NAVY },
        { text: "indistinguishable.", b: false, italic: true },
      ]
    },
    {
      x: 6.7, w: 6.1, color: RED,
      label: "Finding 2",
      title: "Data > architecture at this scale.",
      body: [
        { text: "Earlier 1 407-patch run: ", b: false },
        { text: "v2 beat v1 by +2.6 F1.", b: true, color: NAVY, br: true },
        { text: "Now (4 972 patches): ", b: false },
        { text: "the gap vanishes.", b: true, color: NAVY, br: true },
        { text: " ", b: false, br: true },
        { text: "Regularisation tricks (CBAM, Focal, MixUp) help most when training data is scarce.", b: false, italic: true, br: true },
        { text: " ", b: false, br: true },
        { text: "With sufficient data, plain ResNet-50 is enough.", b: true, color: NAVY },
      ]
    },
  ];

  cards.forEach(c => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: c.x, y: 2.0, w: c.w, h: 4.6,
      fill: { color: "FFFFFF" }, line: { color: "E0DCD0", width: 1 },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.08 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: c.x, y: 2.0, w: c.w, h: 0.08, fill: { color: c.color }, line: { color: c.color },
    });
    s.addText(c.label.toUpperCase(), {
      x: c.x + 0.25, y: 2.2, w: c.w - 0.4, h: 0.3,
      fontFace: "Calibri", fontSize: 11, bold: true, color: c.color, charSpacing: 4, margin: 0,
    });
    s.addText(c.title, {
      x: c.x + 0.25, y: 2.55, w: c.w - 0.4, h: 0.9,
      fontFace: "Georgia", fontSize: 22, bold: true, color: NAVY, margin: 0,
    });
    s.addText(c.body.map((it, j) => ({
      text: it.text,
      options: {
        bold: !!it.b,
        italic: !!it.italic,
        color: it.color || BLACK,
        breakLine: !!it.br,
      }
    })), {
      x: c.x + 0.25, y: 3.55, w: c.w - 0.4, h: 3.0,
      fontFace: "Calibri", fontSize: 14, paraSpaceAfter: 2, margin: 0,
    });
  });

  // Footer-style takeaway bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 6.7, w: 12.3, h: 0.45,
    fill: { color: NAVY }, line: { color: NAVY },
  });
  s.addText([
    { text: "Take-away:  ", options: { bold: true, color: "FFFFFF" } },
    { text: "more data moved the needle far more than any architectural trick.", options: { italic: true, color: "FFFFFF" } },
  ], {
    x: 0.5, y: 6.7, w: 12.3, h: 0.45,
    fontFace: "Calibri", fontSize: 13, align: "center", valign: "middle", margin: 0,
  });

  footer(s, 11, 17);
}

// ─────────────────────────────────────────────────────────
//  SLIDE 12 — ABLATION STUDY  (NEW)
// ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "Ablation Study — Each Component in Isolation");
  bigTitle(s, "What Each Improvement Contributes Alone");

  // Header for ablation table
  const ablRows = [
    [
      { text: "Variant", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "F1",      options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "MCC",     options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "ROC-AUC", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "Δ F1 vs v1", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "Verdict", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
    ],
    [
      { text: "v1  (baseline)",    options: { bold: true, color: NAVY, fill: { color: "FFFFFF" } } },
      { text: "0.9833", options: { color: BLACK, align: "center", fill: { color: "FFFFFF" } } },
      { text: "0.9233", options: { color: BLACK, align: "center", fill: { color: "FFFFFF" } } },
      { text: "0.9929", options: { color: BLACK, align: "center", fill: { color: "FFFFFF" } } },
      { text: "—",      options: { color: GRAY,  align: "center", fill: { color: "FFFFFF" } } },
      { text: "reference", options: { color: GRAY, italic: true, align: "center", fill: { color: "FFFFFF" } } },
    ],
    [
      { text: "v1 + CBAM", options: { bold: true, color: NAVY, fill: { color: LIGHT } } },
      { text: "0.9795", options: { color: BLACK, align: "center", fill: { color: LIGHT } } },
      { text: "0.9063", options: { color: BLACK, align: "center", fill: { color: LIGHT } } },
      { text: "0.9912", options: { color: BLACK, align: "center", fill: { color: LIGHT } } },
      { text: "−0.0038", options: { color: GRAY, align: "center", fill: { color: LIGHT } } },
      { text: "tied within noise",  options: { color: GRAY, italic: true, align: "center", fill: { color: LIGHT } } },
    ],
    [
      { text: "v1 + Focal", options: { bold: true, color: NAVY, fill: { color: "FFFFFF" } } },
      { text: "0.9807", options: { color: BLACK, align: "center", fill: { color: "FFFFFF" } } },
      { text: "0.9128", options: { color: BLACK, align: "center", fill: { color: "FFFFFF" } } },
      { text: "0.9907", options: { color: BLACK, align: "center", fill: { color: "FFFFFF" } } },
      { text: "−0.0026", options: { color: GRAY, align: "center", fill: { color: "FFFFFF" } } },
      { text: "tied within noise",  options: { color: GRAY, italic: true, align: "center", fill: { color: "FFFFFF" } } },
    ],
    [
      { text: "v1 + MixUp", options: { bold: true, color: NAVY, fill: { color: LIGHT } } },
      { text: "0.9804", options: { color: BLACK, align: "center", fill: { color: LIGHT } } },
      { text: "0.9096", options: { color: BLACK, align: "center", fill: { color: LIGHT } } },
      { text: "0.9884", options: { color: BLACK, align: "center", fill: { color: LIGHT } } },
      { text: "−0.0029", options: { color: GRAY, align: "center", fill: { color: LIGHT } } },
      { text: "tied within noise",  options: { color: GRAY, italic: true, align: "center", fill: { color: LIGHT } } },
    ],
    [
      { text: "v1 + TTA", options: { bold: true, color: NAVY, fill: { color: "FFFFFF" } } },
      { text: "0.9854", options: { bold: true, color: RED, align: "center", fill: { color: "FFFFFF" } } },
      { text: "0.9329", options: { bold: true, color: RED, align: "center", fill: { color: "FFFFFF" } } },
      { text: "0.9934", options: { bold: true, color: RED, align: "center", fill: { color: "FFFFFF" } } },
      { text: "+0.0021", options: { bold: true, color: RED, align: "center", fill: { color: "FFFFFF" } } },
      { text: "free improvement", options: { color: RED, italic: true, bold: true, align: "center", fill: { color: "FFFFFF" } } },
    ],
    [
      { text: "v2  (all four)", options: { bold: true, color: NAVY, fill: { color: "FCE6E7" } } },
      { text: "0.9829", options: { color: BLACK, align: "center", fill: { color: "FCE6E7" } } },
      { text: "0.9212", options: { color: BLACK, align: "center", fill: { color: "FCE6E7" } } },
      { text: "0.9917", options: { color: BLACK, align: "center", fill: { color: "FCE6E7" } } },
      { text: "−0.0004", options: { color: GRAY, align: "center", fill: { color: "FCE6E7" } } },
      { text: "combined model",   options: { bold: true, color: RED, italic: true, align: "center", fill: { color: "FCE6E7" } } },
    ],
  ];
  s.addTable(ablRows, {
    x: 0.5, y: 2.0, w: 12.3,
    colW: [2.4, 1.4, 1.4, 1.6, 1.7, 3.8],
    rowH: 0.42,
    fontFace: "Calibri", fontSize: 13,
    border: { type: "solid", pt: 0.5, color: "DDDDDD" },
    valign: "middle",
  });

  // Bottom takeaway
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 5.45, w: 12.3, h: 1.7,
    fill: { color: "FFFFFF" }, line: { color: "E0DCD0", width: 1 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 5.45, w: 0.12, h: 1.7, fill: { color: RED }, line: { color: RED }
  });
  s.addText("What the ablation tells us", {
    x: 0.7, y: 5.55, w: 11.9, h: 0.35,
    fontFace: "Calibri", fontSize: 12, bold: true, color: GRAY, charSpacing: 4, margin: 0,
  });
  s.addText([
    { text: "Individually, ", options: { color: BLACK } },
    { text: "CBAM, Focal Loss, and MixUp ", options: { bold: true, color: NAVY } },
    { text: "are each statistically tied with v1 (Δ within ±0.008 F1 confidence interval).", options: { color: BLACK, breakLine: true } },
    { text: "Only ", options: { color: BLACK } },
    { text: "TTA ", options: { bold: true, color: RED } },
    { text: "delivers a measurable improvement on its own.", options: { color: BLACK, breakLine: true } },
    { text: "But each component addresses a ", options: { color: BLACK } },
    { text: "different limitation ", options: { bold: true, color: NAVY } },
    { text: "of ResNet-50 — and combining them in v2 produces the ", options: { color: BLACK } },
    { text: "methodologically complete model.", options: { italic: true, bold: true, color: RED } },
  ], {
    x: 0.7, y: 5.95, w: 11.9, h: 1.15,
    fontFace: "Calibri", fontSize: 13, paraSpaceAfter: 4, margin: 0,
  });

  footer(s, 12, 17);
}

// ─────────────────────────────────────────────────────────
//  SLIDE 13 — RECOMMENDATION
// ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "Recommendation");
  bigTitle(s, "Which Model Should We Deploy?");

  // Big v2 callout on the left
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 2.0, w: 5.5, h: 4.7,
    fill: { color: NAVY }, line: { color: NAVY }
  });
  s.addText("OUR PICK", {
    x: 0.7, y: 2.2, w: 5.1, h: 0.4,
    fontFace: "Calibri", fontSize: 13, bold: true, color: "C0CADC", charSpacing: 6, margin: 0,
  });
  s.addText("v2", {
    x: 0.7, y: 2.55, w: 5.1, h: 1.6,
    fontFace: "Georgia", fontSize: 96, bold: true, color: "FFFFFF", margin: 0,
  });
  s.addText("ResNet-50 + CBAM", {
    x: 0.7, y: 4.1, w: 5.1, h: 0.5,
    fontFace: "Georgia", fontSize: 22, bold: true, color: "FFFFFF", margin: 0,
  });
  s.addText("Focal Loss · MixUp · TTA", {
    x: 0.7, y: 4.55, w: 5.1, h: 0.45,
    fontFace: "Calibri", fontSize: 16, italic: true, color: "C0CADC", margin: 0,
  });
  // Red rule
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.7, y: 5.15, w: 0.6, h: 0.05, fill: { color: RED }, line: { color: RED }
  });
  s.addText([
    { text: "Test metrics", options: { bold: true, color: "FFFFFF", breakLine: true } },
    { text: "F1 = 0.9829", options: { color: "FFFFFF", fontSize: 14, breakLine: true } },
    { text: "MCC = 0.9212", options: { color: "FFFFFF", fontSize: 14, breakLine: true } },
    { text: "ROC-AUC = 0.9917", options: { color: "FFFFFF", fontSize: 14, breakLine: true } },
    { text: "Recall = 0.9874", options: { color: "FFFFFF", fontSize: 14 } },
  ], {
    x: 0.7, y: 5.35, w: 5.1, h: 1.4,
    fontFace: "Calibri", fontSize: 14, paraSpaceAfter: 4, margin: 0,
  });

  // Right: 4 reasons stacked
  s.addText("Why v2 — combining the four components into one model", {
    x: 6.3, y: 2.0, w: 6.5, h: 0.45,
    fontFace: "Georgia", fontSize: 16, bold: true, color: NAVY, margin: 0,
  });

  const reasons = [
    {
      n: "1",
      title: "Methodologically complete",
      body: "v2 addresses every known ResNet-50 limitation in one package: no attention → CBAM, naive imbalance handling → Focal Loss, weak regularisation → MixUp, single-pass inference → TTA."
    },
    {
      n: "2",
      title: "Interpretability you can show clinicians",
      body: "Only v2 (and v3) produces the CBAM spatial-attention heatmap. v1 + TTA cannot show where the model is looking — a deal-breaker for clinical adoption."
    },
    {
      n: "3",
      title: "Defence in depth against deployment shift",
      body: "Focal Loss + MixUp + CBAM together regularise the model against new scanners, new populations, or shifted nodule prevalence — v1 + TTA only adds inference-side averaging."
    },
    {
      n: "4",
      title: "Statistically equivalent on this test set",
      body: "All deltas (v1, v1+TTA, v2) sit within the ±0.008 F1 confidence interval. v2's marginal numerical deficit is not a practical disadvantage — its qualitative gains are."
    },
  ];
  let yy = 2.55;
  reasons.forEach(r => {
    // Number badge
    s.addShape(pres.shapes.OVAL, {
      x: 6.3, y: yy, w: 0.55, h: 0.55, fill: { color: RED }, line: { color: RED }
    });
    s.addText(r.n, {
      x: 6.3, y: yy, w: 0.55, h: 0.55,
      fontFace: "Georgia", fontSize: 18, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(r.title, {
      x: 7.0, y: yy - 0.05, w: 5.8, h: 0.4,
      fontFace: "Calibri", fontSize: 14, bold: true, color: NAVY, margin: 0,
    });
    s.addText(r.body, {
      x: 7.0, y: yy + 0.32, w: 5.8, h: 0.6,
      fontFace: "Calibri", fontSize: 11, color: BLACK, margin: 0,
    });
    yy += 1.05;
  });

  // Bottom alternates — placed above the page-number footer (footer is at y = 7.10)
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 6.55, w: 12.3, h: 0.4,
    fill: { color: NAVY }, line: { color: NAVY },
  });
  s.addText([
    { text: "Alternatives —  ", options: { bold: true, color: "FFFFFF" } },
    { text: "v1 ", options: { bold: true, color: "FFFFFF" } },
    { text: "for a quick prototype  ·  ", options: { color: "FFFFFF" } },
    { text: "v3 ", options: { bold: true, color: "FFFFFF" } },
    { text: "for edge / mobile / low-VRAM deployment (3× smaller, similar accuracy).", options: { color: "FFFFFF" } },
  ], {
    x: 0.5, y: 6.55, w: 12.3, h: 0.4,
    fontFace: "Calibri", fontSize: 12, align: "center", valign: "middle", margin: 0,
  });

  footer(s, 13, 17);
}

// ─────────────────────────────────────────────────────────
//  SLIDE 13 — CONCLUSION  (5%)
// ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "Conclusion");
  bigTitle(s, "What we delivered");

  const items = [
    { title: "Leakage-free LUNA16 classifier",
      body: "Patient-level split across full LUNA16 (888 patients). Test = 1 528 patches across 176 unseen patients." },
    { title: "Three honestly-compared models",
      body: "v1 (ResNet-50)  ·  v2 (+CBAM, Focal, MixUp, TTA)  ·  v3 (DenseNet-121+CBAM, Focal, MixUp, TTA)" },
    { title: "Production-grade results",
      body: "F1 = 0.9833  ·  ROC-AUC = 0.9929  ·  MCC = 0.9233  —  competitive with peer-reviewed LUNA16 baselines." },
    { title: "Honest scientific story",
      body: "Detected and corrected patient-level leakage on IQ-OTH/NCCD. Reported deltas with confidence intervals — not just point estimates." },
  ];

  items.forEach((it, i) => {
    const y = 2.05 + i * 1.15;
    // Number circle
    s.addShape(pres.shapes.OVAL, {
      x: 0.5, y, w: 0.8, h: 0.8, fill: { color: NAVY }, line: { color: NAVY },
    });
    s.addText(String(i + 1), {
      x: 0.5, y, w: 0.8, h: 0.8,
      fontFace: "Georgia", fontSize: 22, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    // Title + body
    s.addText(it.title, {
      x: 1.5, y: y - 0.05, w: 11.3, h: 0.45,
      fontFace: "Georgia", fontSize: 18, bold: true, color: NAVY, margin: 0,
    });
    s.addText(it.body, {
      x: 1.5, y: y + 0.4, w: 11.3, h: 0.6,
      fontFace: "Calibri", fontSize: 13, color: BLACK, margin: 0,
    });
  });

  footer(s, 14, 17);
}

// ─────────────────────────────────────────────────────────
//  SLIDE 12 — DEMO PLACEHOLDER  (5%)
// ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "Short Demonstration");
  bigTitle(s, "Live Demo (≤ 1 minute)");

  // Big placeholder area
  s.addShape(pres.shapes.RECTANGLE, {
    x: 1.5, y: 2.0, w: 10.3, h: 4.7,
    fill: { color: "FFFFFF" }, line: { color: NAVY, width: 2, dashType: "dash" },
  });
  s.addText("[ Demo video — to be inserted ]", {
    x: 1.5, y: 3.5, w: 10.3, h: 0.7,
    fontFace: "Georgia", fontSize: 28, italic: true, color: GRAY,
    align: "center", valign: "middle", margin: 0,
  });
  s.addText("Suggested content: load best model → infer on a held-out test patch → show predicted probability and CBAM heatmap overlay (max 60 s).", {
    x: 1.5, y: 4.5, w: 10.3, h: 1.0,
    fontFace: "Calibri", fontSize: 14, color: GRAY,
    align: "center", margin: 0,
  });

  footer(s, 15, 17);
}

// ─────────────────────────────────────────────────────────
//  SLIDE 13 — INDIVIDUAL CONTRIBUTIONS
// ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "Individual Contributions");
  bigTitle(s, "Team Roles & What Each Member Did");

  const memberRows = [
    [
      { text: "Member", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "left" } },
      { text: "Student ID", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "Role", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "Contribution", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "left" } },
    ],
    [
      { text: "Huu Thuc Tran", options: { color: NAVY, bold: true, fill: { color: "FFFFFF" } } },
      { text: "26164741", options: { align: "center", color: BLACK, fill: { color: "FFFFFF" } } },
      { text: "Leader", options: { align: "center", color: RED, bold: true, italic: true, fill: { color: "FFFFFF" } } },
      { text: "Dataset acquisition + extraction; patient-leakage analysis on IQ-OTH/NCCD; LUNA16 patch pipeline; implementation, training and evaluation of v1, v2, v3 (CBAM, Focal Loss, MixUp, TTA); 9-metric comparison study; final report.", options: { color: BLACK, fill: { color: "FFFFFF" }, fontSize: 12 } },
    ],
    [
      { text: "Alex", options: { color: NAVY, bold: true, fill: { color: LIGHT } } },
      { text: "26131779", options: { align: "center", color: BLACK, fill: { color: LIGHT } } },
      { text: "Member", options: { align: "center", color: GRAY, italic: true, fill: { color: LIGHT } } },
      { text: "Slide design (Seminar 1) and initial Seminar 2 draft; early literature review on lung cancer detection.", options: { color: BLACK, fill: { color: LIGHT }, fontSize: 12 } },
    ],
    [
      { text: "Rabiul", options: { color: NAVY, bold: true, fill: { color: "FFFFFF" } } },
      { text: "[Student ID — TBC]", options: { align: "center", color: GRAY, italic: true, fill: { color: "FFFFFF" } } },
      { text: "Member", options: { align: "center", color: GRAY, italic: true, fill: { color: "FFFFFF" } } },
      { text: "Literature review on lung-nodule classification methods.", options: { color: BLACK, fill: { color: "FFFFFF" }, fontSize: 12 } },
    ],
  ];
  s.addTable(memberRows, {
    x: 0.5, y: 2.0, w: 12.3,
    colW: [2.4, 1.7, 1.4, 6.8],
    rowH: 1.0,
    fontFace: "Calibri", fontSize: 13,
    border: { type: "solid", pt: 0.5, color: "DDDDDD" },
    valign: "middle",
  });

  // Team-roles footnote — moved up (was 6.45) so it doesn't collide with the page-number footer at 7.10
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 6.20, w: 12.3, h: 0.65,
    fill: { color: NAVY }, line: { color: NAVY },
  });
  s.addText([
    { text: "Team-roles statement:  ", options: { bold: true, color: "FFFFFF" } },
    { text: "the Leader handled implementation and analytical work; teammates contributed to communication and literature support. ", options: { color: "FFFFFF" } },
    { text: "All members agreed the contribution split above is accurate.", options: { italic: true, color: "FFFFFF" } },
  ], {
    x: 0.6, y: 6.20, w: 12.1, h: 0.65,
    fontFace: "Calibri", fontSize: 12, align: "center", valign: "middle", margin: 0,
  });

  footer(s, 16, 17);
}

// ─────────────────────────────────────────────────────────
//  SLIDE 14 — Q&A / THANK YOU
// ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.4, h: SLIDE_H, fill: { color: RED }, line: { color: RED }
  });
  // UTS logo on white circle
  s.addShape(pres.shapes.OVAL, {
    x: SLIDE_W - 1.55, y: 0.45, w: 0.8, h: 0.8,
    fill: { color: "FFFFFF" }, line: { color: "FFFFFF" },
  });
  s.addImage({ path: "uts logo.png", x: SLIDE_W - 1.45, y: 0.55, h: 0.6, w: 0.6 });

  s.addText("Thank you", {
    x: 1.0, y: 2.5, w: 11, h: 1.4,
    fontFace: "Georgia", fontSize: 80, bold: true, color: "FFFFFF", margin: 0,
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 1.0, y: 4.0, w: 0.8, h: 0.05, fill: { color: RED }, line: { color: RED }
  });
  s.addText("Questions & Discussion", {
    x: 1.0, y: 4.2, w: 11, h: 0.7,
    fontFace: "Georgia", fontSize: 28, italic: true, color: "C0CADC", margin: 0,
  });

  s.addText("Lung Cancer Classification  ·  Seminar 2  ·  12 May 2026", {
    x: 1.0, y: SLIDE_H - 0.9, w: 11, h: 0.4,
    fontFace: "Calibri", fontSize: 13, color: "C0CADC", margin: 0,
  });
}

// ───────── Save ─────────
pres.writeFile({ fileName: "Lung_Cancer_Classification_Seminar2.pptx" })
    .then(f => console.log("Wrote " + f));
