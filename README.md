# Team-Datasol
REPOSITORY FOR MAINTAINING OUR GROUP'S CODE FOR CSE558 COURSE 
# Debunking T20 Myths: Statistical Inference on Chasing Under Dew, Boundary Proportions, and Venue Profiles in the IPL

**Course**: CSE558 Data Science — Monsoon 2026, IIIT-Delhi  
**Project**: Midterm Project Report  

---

## 📌 Project Overview
This project investigates four core tactical and environmental hypotheses in T20 cricket using ball-by-ball records from 17 seasons of the Indian Premier League (2008–2024). We use statistical hypothesis testing techniques covered in the CSE558 syllabus:

1. **Problem 1 (Primary - Dew / Chasing Advantage)**: Does chasing under dew elevate average scoring rates per delivery ($\mu_2$ vs. $\mu_1$) at dew-prone evening grounds (Wankhede, Eden Gardens, Chinnaswamy)?  
   $\rightarrow$ **Welch's Two-Sample t-test**
2. **Problem 2 (Boundary Proportion: Death vs. Powerplay)**: Does the proportion of boundary deliveries ($p_{\text{boundary}}$) in the Death overs (overs 16–20) exceed the Powerplay (overs 1–6) by more than 2 percentage points?  
   $\rightarrow$ **Two-Proportion Z-test**
3. **Problem 3 (Wicket Mechanism vs. Match Phase)**: Is the mode of batsman dismissal (caught, bowled, lbw, run out) independent of the tactical match phase (Powerplay, Middle, Death)?  
   $\rightarrow$ **Chi-Square Test of Independence**
4. **Problem 4 (Ball Outcome vs. Stadium Profile)**: Is the distribution of delivery outcomes (Boundary, Dot, Strike Rotation, Wicket) independent of the iconic Indian stadium (Chinnaswamy, Chepauk, Wankhede, Eden Gardens)?  
   $\rightarrow$ **Chi-Square Test of Independence**

---

## 📂 Repository Structure

```
ipl_data_science_project/
├── data/
│   └── ipl_match_ball_by_ball_data.csv       # 225,954 deliveries × 22 raw columns
├── plots/ & report/
│   ├── fig_p1_dew.pdf                        # EDA Plot 1: Innings 1 vs 2 scoring at dew grounds
│   ├── fig_p2_boundaries.pdf                 # EDA Plot 2: Boundary, Dot & Rotation by phase
│   ├── fig_p3_wickets.pdf                    # EDA Plot 3: Dismissal mechanism across phases
│   └── fig_p4_venues.pdf                     # EDA Plot 4: Outcome percentage across stadiums
├── src/
│   └── run_ipl_analysis.py                   # Complete Python analysis pipeline
├── report/
│   ├── CSE558_Project_Report_Template.tex     # Populated course LaTeX report
│   ├── CSE558_Project_Report_Template_ORIGINAL.tex # Original course template backup
│   └── latex_table_snippets.txt              # Exact LaTeX tables generated from the data
└── README.md                                 # Reproducibility instructions
```

---

## 📊 Dataset Size & Arithmetic (Requirement: ≥ 3M Elements)

- **Raw Dimensions**: $225{,}954\text{ rows} \times 22\text{ columns} = 4{,}970{,}988\text{ cells}$.
- **After Preprocessing & One-Hot Encoding**: $225{,}775\text{ rows} \times 47\text{ columns} = \mathbf{10{,}611{,}425\text{ cells}} \ge 3{,}000{,}000$.
- One-hot encoded features include:
  - Match tactical phase (`Powerplay`, `Middle`, `Death`)
  - Top 10 iconic Indian venues (with minor venues merged into `Other`)
  - Dismissal categories (`caught`, `bowled`, `lbw`, `run out`, `other`, `none`)

---

## ⚙️ How to Reproduce the Analysis

### 1. Prerequisites
Ensure Python 3.9+ is installed along with the standard scientific stack:
```bash
pip install numpy pandas scipy matplotlib
```

### 2. Run the Analysis Script
Execute the main pipeline:
```bash
python src/run_ipl_analysis.py
```
This script will automatically:
1. Clean the raw data ($1.11\text{M}$ blank extras filled with 0, $161$ super over deliveries dropped, $18$ duplicates removed).
2. Generate all 4 publication-quality vector PDF plots into `plots/` and `report/`.
3. Compute the exact statistics, degrees of freedom, $p$-values, and 95% confidence intervals for all 4 hypothesis tests.
4. Export LaTeX snippets.

### 3. Compile the Report
Open `report/CSE558_Project_Report_Template.tex` in **Overleaf** or any local LaTeX environment and compile with **pdfLaTeX**.

---

## 📈 Summary of Statistical Inferences

| Test | Hypotheses | Test Statistic & d.f. | $p$-value | 95% Confidence Interval | Decision | Practical Significance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Test 1 (P1: Dew Chase)** | $H_0: \mu_2 - \mu_1 \le 0.05$ | $t = -5.23$ ($\text{df} = 61{,}689$) | $1.000$ | $[-0.046, +0.006]$ | **Fail to reject $H_0$** | **Refutes popular myth**: Chasing teams score $-0.02$ runs/ball lower due to target pacing. |
| **Test 2 (P2: Boundary Proportion)** | $H_0: p_{\text{Death}} - p_{\text{PP}} \le 0.02$ | $Z = -5.64$ | $1.000$ | $[+0.28\%, +1.17\%]$ | **Fail to reject $H_0$** | **Tiny increase ($+0.72\%$)**: Fails $2\%$ threshold; death acceleration is driven by running hard doubles. |
| **Test 3 (P3: Dismissals vs Phase)** | $H_0$: Phase \& dismissal independent | $\chi^2 = 161.16$ ($\text{df} = 6$) | $3.37 \times 10^{-32}$ | $+3.5\%$ run out surge ($+149$ dismissals) | **Reject $H_0$** | **Decisive shift**: Run outs surge in Death overs ($527$ observed vs. $378$ expected); LBWs peak in Powerplay. |
| **Test 4 (P4: Stadium Profiles)** | $H_0$: Stadium \& outcomes independent | $\chi^2 = 88.08$ ($\text{df} = 9$) | $3.96 \times 10^{-15}$ | $3.1\%$ boundary rate span between venues | **Reject $H_0$** | **Decisive shift**: Chinnaswamy over-indexes on boundaries ($17.8\%$), Chepauk on dots ($30.3\%$). |
