Model Evaluation Results

Logistic Regression

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Safe   | 0.64      | 0.84   | 0.73     | 20140   |
| Unsafe | 0.58      | 0.32   | 0.41     | 14094   |

- **Overall Accuracy:** 63%
- **ROC-AUC:** 0.628
- **Confusion Matrix:** [[16941, 3199], [9585, 4509]]

---

Random Forest

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Safe   | 0.74      | 0.82   | 0.78     | 20140   |
| Unsafe | 0.70      | 0.58   | 0.64     | 14094   |

- **Overall Accuracy:** 72%
- **ROC-AUC:** 0.776
- **Confusion Matrix:** [[16557, 3583], [5861, 8233]]

---

Comparison Summary

| Model               | Accuracy | ROC-AUC | Safe F1 | Unsafe F1 |
|---------------------|----------|---------|---------|-----------|
| Logistic Regression | 63%      | 0.628   | 0.73    | 0.41      |
| **Random Forest**   | **72%**  | **0.776**| **0.78**| **0.64**  |

Why Random Forest Was Selected?
- 9% higher accuracy over Logistic Regression
- ROC-AUC improved from 0.628 → 0.776 (+23%)
- Unsafe class F1 improved from 0.41 → 0.64 — critical for 
  routing people away from dangerous roads
- Handles non-linear spatial crime patterns that LR misses
- No feature scaling required at inference time
