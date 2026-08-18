# Drift-Sense — Wafer Inspection

**AI-assisted navigation-error recovery and precision image localization for periodic semiconductor wafer inspection.**

## 1. Project Overview

Drift-Sense addresses the **navigation-error recovery** problem in semiconductor wafer inspection.

A wafer inspection tool must repeatedly revisit the same site with very high positional consistency. Small stage errors caused by factors such as thermal expansion, vibration, and mechanical slack can shift the revisit location. Periodic semiconductor layouts make the problem harder because multiple locations can look visually similar.

The objective is to locate the intended pattern in a larger search image and output its center coordinate **(x, y)**.

## 2. Key Approach

The submitted implementation uses a **DRAM-style periodic layout** and a classical computer-vision localization pipeline.

Instead of relying only on a single template-matching peak, the pipeline combines:

- Global intensity-based candidate generation
- Global edge/Chamfer-based structural evidence
- SIFT feature matching with RANSAC
- Candidate merging and independent-source evidence fusion
- Local rotation and scale refinement
- Combined intensity + edge + Chamfer scoring
- Confidence/mode reporting

This design is intended to reduce errors caused by periodic visual ambiguity.

## 3. Processing Pipeline

```text
Reference Image
       |
       v
Image Preparation
       |
       v
Template Construction
       |
       +--------------------+
       |                    |
       v                    v
Global Intensity       Global Edge
Candidates             Candidates
       |                    |
       +---------+----------+
                 |
                 v
          SIFT / RANSAC
          Candidates
                 |
                 v
       Candidate Merging
       + Evidence Fusion
                 |
                 v
       Local Rotation /
       Scale Refinement
                 |
                 v
       NCC + Edge + Chamfer
              Score
                 |
                 v
        Best Candidate
                 |
                 v
             (x, y)
```

## 4. Repository Structure

```text
Drift-Sense-Wafer-Inspection/
├── README.md
├── requirements.txt
├── src/
│   └── step6_9_5_dram_rotation_scale_lock.py
├── results/
│   └── final_results.csv
└── docs/
    ├── Drift_Sense_i4C_Hackathon_2026_Complete_PPT.pptx
    └── Drift_Sense_i4C_Hackathon_2026_Complete_PPT.pdf
```

Additional files may be added as the project is packaged for submission.

## 5. Installation

Recommended environment: Python 3.x on Windows/Linux.

Install dependencies:

```bash
pip install -r requirements.txt
```

## 6. Running the Localization

Run the final localization script from the repository root:

```bash
python src/step6_9_5_dram_rotation_scale_lock.py
```

The script evaluates the configured reference/search image pairs and reports:

- Ground-truth center
- Predicted center
- Pixel localization error
- Structural score
- NCC score
- Edge score
- Chamfer score
- Rotation estimate
- Scale estimate
- Localization mode
- Inference time

## 7. Evaluation

The evaluation uses ground-truth coordinates only for **measuring prediction error**, not for selecting the prediction.

For each image pair:

```text
error = sqrt((predicted_x - gt_x)^2 + (predicted_y - gt_y)^2)
```

The final evaluation reports:

- Number of evaluated samples
- Mean error
- Median error
- Minimum/maximum error
- Accuracy within 1 pixel
- Accuracy within 3 pixels
- Accuracy within 5 pixels
- Accuracy within 10 pixels
- Average structural score
- Average inference time

## 8. Reported Results

The submitted project results should be taken from the exact final GitHub version used for hackathon submission.

The project presentation currently reports:

- Test cases: **30**
- Within 1 pixel: **90%**
- Within 3 pixels: **100%**
- Within 5 pixels: **100%**
- Mean error: **0.63 px**
- Inference time: **2092.48 ms/pair**

These values should be rechecked against `results/final_results.csv` before final submission.

## 9. Example Result Interpretation

### Success case

```text
True center:      (646, 408)
Predicted center: (646, 408)
Pixel error:      0.00 px
```

### Honest failure case

```text
True center:      (154, 813)
Predicted center: (154, 814)
Pixel error:      1.00 px
```

The failure case demonstrates a small residual localization error rather than hiding the limitation of the system.

## 10. Why This Approach

Simple template matching can fail in highly periodic layouts because several locations may have nearly identical local appearance.

Drift-Sense uses multiple independent structural signals:

1. Intensity similarity
2. Edge structure
3. SIFT feature correspondence
4. RANSAC geometric consistency
5. Local rotation/scale refinement
6. Combined structural scoring

The goal is to identify the intended structural location rather than simply selecting the strongest local pixel-similarity peak.

## 11. Reproducibility

For reproducibility, keep the following together in the repository:

- Source code
- `requirements.txt`
- Final evaluation CSV
- Documentation/PPT/PDF
- Example input data where permitted
- Reference/citation document

No manual source-code modification should be required to execute the final evaluation.

## 12. Team

**Team Name:** MCTE Team 2

**Institution:** Military College of Telecommunication Engineering (MCTE), Mhow, Indore (M.P.)

Team members:

- Manas Bajpai — Team Captain / AI-ML Engineer
- Sukhveer Singh — Dataset Engineer
- Borey Dutta — Testing & Deployment
- Sudheer Choudhary — Dataset Development

## 13. Hackathon

**SEMICON India Hackathon 2026 — Drift-Sense**

Problem area:

**Drift-Sense: Navigation-Error Recovery**

## 14. References

1. i4C / SEMICON India Hackathon 2026 — Drift-Sense problem statement and submission requirements.
2. Villarrubia, J. S. et al. (2001), *Edge Determination for Polycrystalline Silicon Lines on Gate Oxide*, NIST / SPIE.
3. ETH Zurich ScopeM, *SEM – Imaging with Secondary Electrons*.
4. Microelectronic Engineering (2019), *Deep learning denoising of SEM images towards noise-reduced LER measurements*.
5. Szeliski, R., *Computer Vision: Algorithms and Applications*, Springer.
6. Lowe, D. G. (2004), *Distinctive Image Features from Scale-Invariant Keypoints*, IJCV.
7. Oxford Academic, *Spatial resolution in secondary-electron microscopy*.

## 15. Repository

GitHub:

https://github.com/dattaborey2-ai/Drift-Sense-Wafer-Inspection
