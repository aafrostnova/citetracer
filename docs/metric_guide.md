# Metric Calculation Guide

This document defines every metric we report for the citation hallucination
benchmark. It is written for someone scoring their own detector, with no
access to our code or data files. If, for each citation, you know

- the **ground-truth** class and subtype, and
- your detector's **predicted** class and subtype,

the formulas below produce the confusion matrix, the per-bucket table, and the
safe-error summary that appear in our paper.

## 1. The taxonomy

Each citation has one **class** (top level) and one **subtype** (fine-grained).

| Class          | Subtypes                           |
| -------------- | ---------------------------------- |
| `REAL`         | `R1`, `R2`, `R3`                   |
| `POTENTIAL`    | `P1`, `P3`                         |
| `HALLUCINATED` | `H1`, `H2`, `H3`, `H4`, `H5`, `H6` |

`REAL` citations point to a paper that exists and whose fields are correct.
`POTENTIAL` citations point to a real paper whose verification is ambiguous
(`P1` changes one author's given name; `P3` fabricates peripheral fields that
no source can corroborate or refute). `HALLUCINATED` citations have at least
one field that is wrong against the real record (`H1`..`H6` name which field).

For scoring we collapse `R1`, `R2`, `R3` into a single bucket `R` because the
distinction between them is cosmetic, not reviewer-actionable. The nine scoring
buckets are therefore:

```
R, P1, P3, H1, H2, H3, H4, H5, H6
```

## 2. Inputs per citation

Per citation, you supply four values. The first two are ground truth, the
last two come from the detector under evaluation:

| Field           | Values                                            |
| --------------- | ------------------------------------------------- |
| `true_class`    | `REAL`, `POTENTIAL`, `HALLUCINATED`               |
| `true_subtype`  | one of the 11 codes (`R1`..`H6`)                  |
| `pred_class`    | `REAL`, `POTENTIAL`, `HALLUCINATED`               |
| `pred_subtypes` | a **set** of codes (often one, sometimes several) |

`pred_subtypes` is a set because a detector may legitimately emit more than
one code for a citation (e.g. a reference with both a title error and a
pages error could be labeled `{H1, H6}`). If your detector emits a single
code, use a set of size one.

## 3. Bucketizing each citation

Map each citation to one of the nine buckets. Ground-truth bucket is
unambiguous; predicted bucket comes from a three-step rule that matches how
a reviewer would read the detector's output.

```
# Ground-truth bucket (unique)
true_bucket(citation)  = "R"                    if true_subtype ∈ {R1, R2, R3}
                       = true_subtype           otherwise

# Predicted bucket (pick the most specific code the detector commits to)
pred_bucket(citation)  = "R"                    if pred_class == "REAL"
                       = any code in pred_subtypes that equals true_bucket,
                                                if true_bucket ∈ pred_subtypes
                       = the single code in pred_subtypes,
                                                if |pred_subtypes| == 1
                       = the lexicographically smallest code in pred_subtypes,
                                                otherwise
```

The second branch gives the detector credit when its multi-code prediction
includes the correct code. The last tie-break matters only for multi-code
predictions whose true bucket is not among them; any deterministic rule
produces the same confusion matrix up to a relabeling of wrong codes.

## 4. Figure 1. Confusion matrix (9 × 9)

Rows are the ground-truth bucket, columns are the predicted bucket. Cell
`M[i, j]` counts citations with `true_bucket == i` and `pred_bucket == j`.
Diagonal cells are correct predictions.

```
           pred → R    P1   P3   H1   H2   H3   H4   H5   H6
true ↓
     R  [ ...                                              ]
    P1  [ ...                                              ]
    P3  [ ...                                              ]
    H1  [ ...                                              ]
    H2  [ ...                                              ]
    H3  [ ...                                              ]
    H4  [ ...                                              ]
    H5  [ ...                                              ]
    H6  [ ...                                              ]
```

Report raw counts; render the diagonal in bold or with a colored background in
figures. For readability, you may print row-normalized percentages alongside
the counts.

## 5. Table 2. Per-class performance

Nine rows (one per scoring bucket), eight columns. For bucket `T`:

```
N(T)   = number of citations with true_bucket == T
TP(T)  = |{ i : pred_bucket_i == T AND true_bucket_i == T }|
FP(T)  = |{ i : pred_bucket_i == T AND true_bucket_i != T }|
FN(T)  = |{ i : pred_bucket_i != T AND true_bucket_i == T }|
TN(T)  = N_total − TP(T) − FP(T) − FN(T)

Precision(T) = TP(T) / (TP(T) + FP(T))
Recall(T)    = TP(T) / (TP(T) + FN(T))
F1(T)        = 2 · P(T) · R(T) / (P(T) + R(T))
Accuracy(T)  = (TP(T) + TN(T)) / N_total
```

Reported columns:

| Bucket | N     | TP | FP | FN | Precision | Recall | F1  | Accuracy |
| ------ | ----- | -- | -- | -- | --------- | ------ | --- | -------- |
| R      | ...   | .. | .. | .. | ...       | ...    | ... | ...      |
| P1     | ...   | .. | .. | .. | ...       | ...    | ... | ...      |
| P3     | ...   | .. | .. | .. | ...       | ...    | ... | ...      |
| H1     | ...   | .. | .. | .. | ...       | ...    | ... | ...      |
| H2     | ...   | .. | .. | .. | ...       | ...    | ... | ...      |
| H3     | ...   | .. | .. | .. | ...       | ...    | ... | ...      |
| H4     | ...   | .. | .. | .. | ...       | ...    | ... | ...      |
| H5     | ...   | .. | .. | .. | ...       | ...    | ... | ...      |
| H6     | ...   | .. | .. | .. | ...       | ...    | ... | ...      |

Accuracy is a one-vs-rest metric here (how often the detector is right about
bucket `T`, whether positive or negative). It is low-variance across rows on
class-imbalanced datasets and should be read alongside precision/recall.

## 6. Table 3. Safe-error analysis

The three classes carry different risks for a reviewer: a citation that is
actually `REAL` but labeled `HALLUCINATED` is a reputational harm; a citation
that is actually `HALLUCINATED` but labeled `REAL` is a missed fabrication.
We use a severity ordering

```
REAL < POTENTIAL < HALLUCINATED
```

and call a misclassification **safe** when the predicted class is stricter
than the truth (the detector over-flags) and **dangerous** when the predicted
class is more permissive than the truth (the detector under-flags). The
class-level confusion is summarized in three rows:

| True class   | Predicted class | Direction  | Comment                          |
| ------------ | ---------------- | ---------- | -------------------------------- |
| `REAL`       | `POTENTIAL`     | Safe       | Over-cautious flag               |
| `REAL`       | `HALLUCINATED`  | Safe       | Strong over-flag                 |
| `POTENTIAL`  | `REAL`          | Dangerous  | Ambiguous case rubber-stamped    |
| `POTENTIAL`  | `HALLUCINATED`  | Safe       | Conservative escalation          |
| `HALLUCINATED` | `POTENTIAL`   | Dangerous  | Fabrication softened             |
| `HALLUCINATED` | `REAL`         | Dangerous  | Fabrication missed               |

Let

```
E_safe(R)  = |{ true=R AND pred ∈ {POTENTIAL, HALLUCINATED} }|
E_safe(P)  = |{ true=POTENTIAL AND pred=HALLUCINATED }|
E_safe(H)  = 0

E_dang(R)  = 0
E_dang(P)  = |{ true=POTENTIAL AND pred=REAL }|
E_dang(H)  = |{ true=HALLUCINATED AND pred ∈ {REAL, POTENTIAL} }|
```

and let `N(C)` denote the count of citations whose true class is `C`.

The Table 3 rows report, per true class:

| True class       | N   | Safe errors   | Safe rate       | Dangerous errors | Dangerous rate  |
| ---------------- | --- | ------------- | --------------- | ---------------- | --------------- |
| `REAL`           | ... | E_safe(R)     | E_safe(R)/N(R)  | E_dang(R)=0      | 0               |
| `POTENTIAL`      | ... | E_safe(P)     | E_safe(P)/N(P)  | E_dang(P)        | E_dang(P)/N(P)  |
| `HALLUCINATED`   | ... | E_safe(H)=0   | 0               | E_dang(H)        | E_dang(H)/N(H)  |

At the bottom, report two dataset-wide rates:

```
SER = ( E_safe(R) + E_safe(P) + E_safe(H) ) / N_total     # safe-error rate
DER = ( E_dang(R) + E_dang(P) + E_dang(H) ) / N_total     # dangerous-error rate
class_accuracy = 1 − SER − DER                             # identity
```

`DER` is the primary deployment metric because dangerous errors are the ones
that damage the citation graph. `SER` is the reviewer-annoyance metric; a
precision-first system trades higher `SER` for lower `DER`.

## 7. Edge cases and conventions

- **Multi-code predictions.** Section 3's `pred_bucket` rule gives credit for
  including the correct code without inflating recall: a detector that dumps
  every code is penalized as a false positive on every wrong bucket.
- **Detector emits no subtype.** Populate Figure 1 and Table 2 by deriving
  `pred_subtypes` from `pred_class` (`REAL` → `{R}`, otherwise skip
  subtype-level rows). `pred_bucket` for `POTENTIAL`/`HALLUCINATED` is the
  class itself (collapse all `P*` or `H*` bucket rows into one). Table 3
  works unchanged because it only needs class-level predictions.
- **Detector with fewer classes** (binary real / fake). Merge `POTENTIAL`
  into `HALLUCINATED` and rebuild the confusion matrix on two classes; in
  Table 3, `E_safe(P)` moves to `E_safe(R→H)` and `E_dang(P)` moves to
  `E_dang(H→R)`. Document the collapse explicitly.
- **Empty predictions / timeouts.** Treat as a prediction of `HALLUCINATED`
  with no subtype (the precision-first default). Alternatively, exclude
  timeouts from all tables and report coverage separately.

## 8. Reporting template

For a given detector on a given dataset, report all three artifacts:

1. **Figure 1** — 9×9 confusion matrix.
2. **Table 2** — per-bucket `N, TP, FP, FN, Precision, Recall, F1, Accuracy`.
3. **Table 3** — per-class safe and dangerous errors, with overall `SER`,
   `DER`, and implied `class_accuracy`.

Also state the dataset size `N_total`, the per-subtype counts, and any
deviations from Section 7 (output mapping, timeout handling). With those
five items your numbers are directly comparable to ours.
