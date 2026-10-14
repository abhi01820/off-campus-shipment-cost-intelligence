# Labeled Evaluation & Anti-Hallucination Audit Report

## Summary Scorecard
| Metric | Value | Target | Status |
| :--- | :--- | :--- | :--- |
| **Precision** | **100.0%** | 100.0% |  PASS |
| **Recall** | **100.0%** | >= 90.0% |  PASS |
| **F1 Score** | **1.000** | >= 0.90 |  PASS |
| **Fabricated Citation Rate** | **0.0%** | **0.0% (Zero Tolerance)** |  PASS |
| **Total Test Cases** | **20** | 20 | Complete |
| **True Positives (TP)** | 5 | 5 | All genuine cost surges justified |
| **True Negatives (TN)** | 15 | 15 | All 7 trap notes & out-of-bounds rejected |
| **False Positives (FP)** | 0 | 0 | Zero false justifications |
| **False Negatives (FN)** | 0 | 0 | Zero missed justifications |

---

## Detailed Test Case Audit Trail
| Case ID | Route | Week | Exp Justified | Act Justified | Exp Note | Act Note | Result | Audit Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `POS_N001_W1` | Chennai-Bangalore | 2025-02-24 | True | True | `N001` | `N001` | **TP (Pass)** | Justified |
| `POS_N001_W2` | Chennai-Bangalore | 2025-03-03 | True | True | `N001` | `N001` | **TP (Pass)** | Justified |
| `POS_N002` | Ahmedabad-Mumbai | 2025-01-20 | True | True | `N002` | `N002` | **TP (Pass)** | Justified |
| `POS_N003_W1` | Mumbai-Delhi | 2025-05-05 | True | True | `N003` | `N003` | **TP (Pass)** | Justified |
| `POS_N003_W2` | Mumbai-Delhi | 2025-05-12 | True | True | `N003` | `N003` | **TP (Pass)** | Justified |
| `NEG_N001_POST` | Chennai-Bangalore | 2025-03-10 | False | False | `` | `` | **TN (Pass)** | confirms route returned to normal conditions, not an active disruption |
| `NEG_N001_PRE` | Chennai-Bangalore | 2025-02-17 | False | False | `` | `` | **TN (Pass)** | effective window (2025-02-24 to 2025-03-08) does not cover week of 2025-02-17 |
| `NEG_N001_WRONG_ROUTE` | Mumbai-Pune | 2025-02-24 | False | False | `` | `` | **TN (Pass)** | Justified |
| `NEG_N002_WRONG_WEEK` | Ahmedabad-Mumbai | 2025-01-27 | False | False | `` | `` | **TN (Pass)** | effective window (2025-01-20 to 2025-01-26) does not cover week of 2025-01-27 |
| `NEG_N002_WRONG_ROUTE` | Delhi-Jaipur | 2025-01-20 | False | False | `` | `` | **TN (Pass)** | Justified |
| `NEG_N003_POST` | Delhi-Chennai | 2025-06-02 | False | False | `` | `` | **TN (Pass)** | effective window (2025-05-05 to 2025-05-18) does not cover week of 2025-06-02 |
| `TRAP_N004` | Delhi-Jaipur | 2024-03-11 | False | False | `` | `` | **TN (Pass)** | affected routes are not part of this dataset |
| `TRAP_N005` | Mumbai-Delhi | 2024-07-29 | False | False | `` | `` | **TN (Pass)** | costs were not significantly affected |
| `TRAP_N006` | Mumbai-Pune | 2025-09-22 | False | False | `` | `` | **TN (Pass)** | mentions stable demand with no major disruptions -- it does not describe a reason for a cost rise on this route |
| `TRAP_N007` | Delhi-Jaipur | 2024-05-20 | False | False | `` | `` | **TN (Pass)** | describes road improvements after resurfacing, not a cost increase |
| `TRAP_N008` | Kolkata-Bhubaneswar | 2025-06-09 | False | False | `` | `` | **TN (Pass)** | reports freight movement remained normal with no disruptions |
| `TRAP_N009` | Chennai-Bangalore | 2025-03-17 | False | False | `` | `` | **TN (Pass)** | confirms route returned to normal conditions, not an active disruption |
| `TRAP_N010` | Mumbai-Pune | 2025-10-27 | False | False | `` | `` | **TN (Pass)** | states compliance costs were absorbed by transporters without a rate change |
| `UNEXP_DELHI_JAIPUR` | Delhi-Jaipur | 2024-11-11 | False | False | `` | `` | **TN (Pass)** | Justified |
| `UNEXP_MUMBAI_PUNE` | Mumbai-Pune | 2025-09-15 | False | False | `` | `` | **TN (Pass)** | mentions stable demand with no major disruptions -- it does not describe a reason for a cost rise on this route |

---
## Trap Note Analysis & Anti-Hallucination Guarantees
1. **N004 (Toll plaza outside dataset)**: Successfully rejected on route match and dataset scope check.
2. **N005 (Mumbai-Delhi maintenance without cost effect)**: Successfully rejected because costs were not significantly affected.
3. **N006 (Stable demand report)**: Successfully rejected because stable demand does not justify rate inflation.
4. **N007 (Delhi-Jaipur road resurfacing)**: Successfully rejected because infrastructure improvements do not justify cost rises.
5. **N008 (Kolkata-Bhubaneswar normal movement)**: Successfully rejected because normal conditions do not justify cost rises.
6. **N009 (Chennai-Bangalore return to normal)**: Successfully rejected as resolution rather than disruption.
7. **N010 (Fleet compliance absorbed)**: Successfully rejected because compliance costs were absorbed without rate changes.
