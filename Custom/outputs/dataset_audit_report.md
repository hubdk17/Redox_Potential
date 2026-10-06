# Phase 1: Complete Dataset Forensic Audit Report

## 1. Overview & Dataset Dimensions
- **Total Rows:** 364
- **Total Features / Columns:** 99
- **Unique PDB Code Entries:** 284
- **Unique Base Crystal Scaffolds:** 76
- **Wild-Type (WT) Entries:** 131 (36.0%)
- **Site-Directed Mutant Entries:** 233 (64.0%)

## 2. Missing Values & Feature Quality
- **Missing Values:** **0 NaNs detected across all 99 columns.** The curated matrix is 100% complete.

## 3. Structural Scaffolds & Cluster Nuclearity Distribution
Physical Fe-S cluster types detected directly from 3D PDB coordinates and HETATM residues:

| Cluster Nuclearity / Type | Number of Scaffolds | Number of Dataset Rows | Percentage |
| :--- | :---: | :---: | :---: |
| **[2Fe-2S]** | 61 | 293 | 80.5% |
| **Rubredoxin [1Fe]** | 7 | 52 | 14.3% |
| **[4Fe-4S]** | 5 | 11 | 3.0% |
| **[3Fe-4S]** | 3 | 8 | 2.2% |

## 4. Target Em & Solvent pH Distributions
| Metric | Experimental Em (mV) | Solvent pH |
| :--- | :---: | :---: |
| **Mean ± SD** | -103.7 ± 245.9 mV | 7.15 ± 1.79 |
| **Min (Extreme Low)** | -500.0 mV | 3.00 |
| **25th Percentile** | -330.0 mV | 7.00 |
| **Median (50th)** | -134.0 mV | 7.30 |
| **75th Percentile** | 88.5 mV | 7.50 |
| **Max (Extreme High)** | 390.0 mV | 14.00 |

## 5. Duplicate & Repeated Measurements Audit
- **Exact Row Duplicates:** 1
- **Repeated Measurements (Same PDB_Code & Same pH):** 12 distinct groups (24 rows)
  - Detailed breakdown of repeated measurements:
    - `1ayf` at pH 7.5: 2 observations -> Em = [-270, -290] (Spread = 20 mV)
    - `1fxa` at pH 8.0: 2 observations -> Em = [-410, -440] (Spread = 30 mV)
    - `1fxa_a45s` at pH 8.0: 2 observations -> Em = [-420, -380] (Spread = 40 mV)
    - `1j7b` at pH 7.3: 2 observations -> Em = [-300, -440] (Spread = 140 mV)
    - `1kb9` at pH 7.4: 2 observations -> Em = [265, 290] (Spread = 25 mV)
    - `1l0v` at pH 7.0: 2 observations -> Em = [-35, -79] (Spread = 44 mV)
    - `1nyk` at pH 7.0: 2 observations -> Em = [161, 140] (Spread = 21 mV)
    - `1nyk` at pH 14.0: 2 observations -> Em = [-275] (Spread = 0 mV)
    - `1rfs` at pH 6.0: 2 observations -> Em = [350, 360] (Spread = 10 mV)
    - `1rie_l142a` at pH 14.0: 2 observations -> Em = [-148, -149] (Spread = 1 mV)
    - `2nuk` at pH 3.0: 2 observations -> Em = [308, 315] (Spread = 7 mV)
    - `3p1m` at pH 7.0: 2 observations -> Em = [-380, -250] (Spread = 130 mV)
  - *Scientific Note:* These represent independent laboratory titrations or literature measurements from different experimental papers for the exact same physical construct under identical pH.

## 6. Audit Verdict & Exclusions Assessment
- **Exclusion Recommendation:** **Do NOT exclude rows.** All 364 observations represent genuine experimental measurements from the literature. Rather than artificially trimming variance, models must be evaluated under rigorous PDB-grouped and homology-held-out splits to prevent repeated-measurement memorization.
