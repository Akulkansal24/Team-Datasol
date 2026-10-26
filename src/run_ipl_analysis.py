import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_PATH = os.path.join(DATA_DIR, 'ipl_match_ball_by_ball_data.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'ipl_cleaned_encoded.csv')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
REPORT_DIR = os.path.join(BASE_DIR, 'report')

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def standardize_venue(venue_str):
    if not isinstance(venue_str, str):
        return 'Other'
    v = venue_str.strip()
    if 'Chinnaswamy' in v:
        return 'M Chinnaswamy Stadium'
    elif 'Wankhede' in v:
        return 'Wankhede Stadium'
    elif 'Eden Gardens' in v:
        return 'Eden Gardens'
    elif 'Feroz Shah Kotla' in v or 'Arun Jaitley' in v:
        return 'Arun Jaitley Stadium'
    elif 'Chidambaram' in v or 'Chepauk' in v:
        return 'MA Chidambaram Stadium'
    elif 'Rajiv Gandhi' in v or 'Uppal' in v:
        return 'Rajiv Gandhi Intl Stadium'
    elif 'Punjab Cricket Association' in v or 'Mohali' in v or 'IS Bindra' in v:
        return 'PCA Stadium Mohali'
    elif 'Sawai Mansingh' in v:
        return 'Sawai Mansingh Stadium'
    elif 'Narendra Modi' in v or 'Motera' in v or 'Sardar Patel' in v:
        return 'Narendra Modi Stadium'
    elif 'Maharashtra Cricket Association' in v or 'Subrata Roy' in v:
        return 'MCA Stadium Pune'
    else:
        return 'Other'

def main():
    print("=" * 80)
    print("CSE558 DATA SCIENCE: IPL 4-TEST ANALYSIS PIPELINE")
    print("=" * 80)

    if not os.path.exists(DATA_PATH):
        fallback = r'C:\Users\hp\Downloads\ipl_match_ball_by_ball_data.csv\ipl_match_ball_by_ball_data.csv'
        path = fallback if os.path.exists(fallback) else DATA_PATH
    else:
        path = DATA_PATH

    print(f"\n[1] Loading dataset from: {path}")
    raw_df = pd.read_csv(path, low_memory=False)
    raw_rows, raw_cols = raw_df.shape
    raw_elements = raw_rows * raw_cols

    extras_cols = ['wides', 'noballs', 'byes', 'legbyes', 'penalty']
    df_clean = raw_df.copy()
    for col in extras_cols:
        df_clean[col] = df_clean[col].fillna(0)

    df_clean = df_clean[df_clean['innings'].isin([1, 2])].copy()
    df_clean = df_clean.drop_duplicates(subset=['match_id', 'innings', 'ball']).copy()

    dropped_cols = ['other_wicket_type', 'other_player_dismissed']
    df_clean = df_clean.drop(columns=dropped_cols, errors='ignore')

    df_clean['standard_venue'] = df_clean['venue'].apply(standardize_venue)
    df_clean['runs_off_bat'] = pd.to_numeric(df_clean['runs_off_bat'], errors='coerce').fillna(0)
    df_clean['extras'] = pd.to_numeric(df_clean['extras'], errors='coerce').fillna(0)
    df_clean['total_runs'] = df_clean['runs_off_bat'] + df_clean['extras']
    df_clean['over_num'] = df_clean['ball'].astype(float).astype(int)
    df_clean['phase'] = np.where(
        df_clean['over_num'] < 6, 'Powerplay',
        np.where(df_clean['over_num'] < 15, 'Middle', 'Death')
    )
    df_clean['is_boundary'] = df_clean['runs_off_bat'].isin([4, 6]).astype(int)
    df_clean['is_dot'] = (df_clean['total_runs'] == 0).astype(int)
    df_clean['is_wicket'] = df_clean['wicket_type'].notna().astype(int)

    phase_ohe = pd.get_dummies(df_clean['phase'], prefix='phase', dtype=int)
    venue_ohe = pd.get_dummies(df_clean['standard_venue'], prefix='venue', dtype=int)
    clean_wicket_cat = df_clean['wicket_type'].fillna('none').apply(
        lambda w: w if w in ['caught', 'bowled', 'run out', 'lbw', 'none'] else 'other_dismissal'
    )
    wicket_ohe = pd.get_dummies(clean_wicket_cat, prefix='dismissal', dtype=int)

    encoded_df = pd.concat([df_clean, phase_ohe, venue_ohe, wicket_ohe], axis=1)
    enc_rows, enc_cols = encoded_df.shape
    enc_elements = enc_rows * enc_cols

    print(f"    Raw matrix:     {raw_rows:,} rows × {raw_cols} cols = {raw_elements:,} cells")
    print(f"    Encoded matrix: {enc_rows:,} rows × {enc_cols} cols = {enc_elements:,} cells")
    print(f"    Size Check Met: {enc_elements >= 3_000_000} (>{3_000_000:,})")

    encoded_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"    Saved processed dataset to: {PROCESSED_DATA_PATH}")

    print("\n[2] Generating 4 Publication-Ready Figures (One per Problem Statement)...")

    fig1, ax1 = plt.subplots(figsize=(6, 3.8), dpi=300)
    dew_venues = ['Wankhede Stadium', 'Eden Gardens', 'M Chinnaswamy Stadium']
    dew_sub = df_clean[df_clean['standard_venue'].isin(dew_venues)]
    dew_grp = dew_sub.groupby(['standard_venue', 'innings'])['total_runs'].mean().unstack()
    
    x_pos = np.arange(len(dew_venues))
    width = 0.35
    b1 = ax1.bar(x_pos - width/2, dew_grp[1], width, label='Innings 1 (Setting Target)', color='#34495e', edgecolor='black', alpha=0.9)
    b2 = ax1.bar(x_pos + width/2, dew_grp[2], width, label='Innings 2 (Chasing under Dew)', color='#3498db', edgecolor='black', alpha=0.9)
    
    ax1.set_ylabel('Mean Runs Scored per Delivery', fontsize=10, fontweight='bold')
    ax1.set_title('P1 EDA: Innings 1 vs Innings 2 Scoring at Dew-Prone Stadiums', fontsize=11, fontweight='bold', pad=8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(['Wankhede\n(Mumbai)', 'Eden Gardens\n(Kolkata)', 'Chinnaswamy\n(Bengaluru)'], fontsize=9)
    ax1.set_ylim(1.15, 1.48)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', frameon=True, fontsize=9)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    fig1.tight_layout()
    f1_p = os.path.join(PLOTS_DIR, 'fig_p1_dew.pdf')
    f1_r = os.path.join(REPORT_DIR, 'fig_p1_dew.pdf')
    fig1.savefig(f1_p); fig1.savefig(f1_r); plt.close(fig1)
    print(f"    Saved Fig 1 to: {f1_p}")

    fig2, ax2 = plt.subplots(figsize=(6, 3.8), dpi=300)
    phases = ['Powerplay', 'Middle', 'Death']
    conds = [df_clean['is_boundary'] == 1, df_clean['is_dot'] == 1]
    df_clean['ball_type'] = np.select(conds, ['Boundary (4/6)', 'Dot Ball (0)'], default='Strike Rotation (1-3)')
    phase_breakdown = pd.crosstab(df_clean['phase'], df_clean['ball_type'], normalize='index').loc[phases] * 100

    x2 = np.arange(len(phases))
    w2 = 0.25
    ax2.bar(x2 - w2, phase_breakdown['Boundary (4/6)'], w2, label='Boundary (4s & 6s)', color='#e74c3c', edgecolor='black', alpha=0.85)
    ax2.bar(x2, phase_breakdown['Dot Ball (0)'], w2, label='Dot Ball (0 runs)', color='#7f8c8d', edgecolor='black', alpha=0.85)
    ax2.bar(x2 + w2, phase_breakdown['Strike Rotation (1-3)'], w2, label='Strike Rotation (1-3)', color='#2ecc71', edgecolor='black', alpha=0.85)

    ax2.set_ylabel('Percentage of Deliveries (%)', fontsize=10, fontweight='bold')
    ax2.set_title('P2 EDA: Delivery Outcome Proportions Across Innings Phases', fontsize=11, fontweight='bold', pad=8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(['Powerplay (Overs 1–6)', 'Middle (Overs 7–15)', 'Death (Overs 16–20)'], fontsize=9)
    ax2.set_ylim(0, 60)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True, fontsize=8.5)

    for i in range(len(phases)):
        b_val = phase_breakdown.loc[phases[i], 'Boundary (4/6)']
        ax2.annotate(f'{b_val:.1f}%', xy=(x2[i] - w2, b_val), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    fig2.tight_layout()
    f2_p = os.path.join(PLOTS_DIR, 'fig_p2_boundaries.pdf')
    f2_r = os.path.join(REPORT_DIR, 'fig_p2_boundaries.pdf')
    fig2.savefig(f2_p); fig2.savefig(f2_r); plt.close(fig2)
    print(f"    Saved Fig 2 to: {f2_p}")

    fig3, ax3 = plt.subplots(figsize=(6, 3.8), dpi=300)
    wickets_sub = df_clean[df_clean['wicket_type'].isin(['caught', 'bowled', 'run out', 'lbw'])].copy()
    w_ct = pd.crosstab(wickets_sub['phase'], wickets_sub['wicket_type']).loc[phases]

    x3 = np.arange(len(phases))
    w3 = 0.18
    ax3.bar(x3 - 1.5*w3, w_ct['caught'], w3, label='Caught', color='#2980b9', edgecolor='black', alpha=0.85)
    ax3.bar(x3 - 0.5*w3, w_ct['bowled'], w3, label='Bowled', color='#e67e22', edgecolor='black', alpha=0.85)
    ax3.bar(x3 + 0.5*w3, w_ct['run out'], w3, label='Run Out', color='#c0392b', edgecolor='black', alpha=0.85)
    ax3.bar(x3 + 1.5*w3, w_ct['lbw'], w3, label='LBW', color='#8e44ad', edgecolor='black', alpha=0.85)

    ax3.set_ylabel('Number of Dismissals', fontsize=10, fontweight='bold')
    ax3.set_title('P3 EDA: Dismissal Mechanisms Across Match Phases', fontsize=11, fontweight='bold', pad=8)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(['Powerplay', 'Middle', 'Death'], fontsize=9)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right', frameon=True, fontsize=8.5)

    fig3.tight_layout()
    f3_p = os.path.join(PLOTS_DIR, 'fig_p3_wickets.pdf')
    f3_r = os.path.join(REPORT_DIR, 'fig_p3_wickets.pdf')
    fig3.savefig(f3_p); fig3.savefig(f3_r); plt.close(fig3)
    print(f"    Saved Fig 3 to: {f3_p}")

    fig4, ax4 = plt.subplots(figsize=(6, 3.8), dpi=300)
    v4_names = ['M Chinnaswamy Stadium', 'MA Chidambaram Stadium', 'Wankhede Stadium', 'Eden Gardens']
    v4_sub = df_clean[df_clean['standard_venue'].isin(v4_names)].copy()
    v4_sub['outcome_cat'] = np.select(
        [v4_sub['is_boundary'] == 1, v4_sub['is_dot'] == 1, v4_sub['wicket_type'].notna()],
        ['Boundary (4/6)', 'Dot Ball (0)', 'Wicket'],
        default='Strike Rotation (1-3)'
    )
    v4_pct = pd.crosstab(v4_sub['standard_venue'], v4_sub['outcome_cat'], normalize='index').loc[v4_names] * 100

    x4 = np.arange(len(v4_names))
    w4 = 0.20
    ax4.bar(x4 - 1.5*w4, v4_pct['Boundary (4/6)'], w4, label='Boundary', color='#e74c3c', edgecolor='black', alpha=0.85)
    ax4.bar(x4 - 0.5*w4, v4_pct['Dot Ball (0)'], w4, label='Dot Ball', color='#7f8c8d', edgecolor='black', alpha=0.85)
    ax4.bar(x4 + 0.5*w4, v4_pct['Strike Rotation (1-3)'], w4, label='Strike Rotation', color='#2ecc71', edgecolor='black', alpha=0.85)
    ax4.bar(x4 + 1.5*w4, v4_pct['Wicket'], w4, label='Wicket', color='#9b59b6', edgecolor='black', alpha=0.85)

    ax4.set_ylabel('Percentage of Deliveries (%)', fontsize=10, fontweight='bold')
    ax4.set_title('P4 EDA: Ball Outcome Distribution Across Iconic Stadiums', fontsize=11, fontweight='bold', pad=8)
    ax4.set_xticks(x4)
    ax4.set_xticklabels(['Chinnaswamy\n(Bengaluru)', 'Chepauk\n(Chennai)', 'Wankhede\n(Mumbai)', 'Eden Gardens\n(Kolkata)'], fontsize=9)
    ax4.set_ylim(0, 55)
    ax4.grid(axis='y', linestyle='--', alpha=0.5)
    ax4.legend(loc='upper right', frameon=True, fontsize=8.5)

    fig4.tight_layout()
    f4_p = os.path.join(PLOTS_DIR, 'fig_p4_venues.pdf')
    f4_r = os.path.join(REPORT_DIR, 'fig_p4_venues.pdf')
    fig4.savefig(f4_p); fig4.savefig(f4_r); plt.close(fig4)
    print(f"    Saved Fig 4 to: {f4_p}")

    print("\n[3] Computing Exact Hypothesis Testing Results...")

    inn2_v = dew_sub[dew_sub['innings'] == 2]['total_runs'].values
    inn1_v = dew_sub[dew_sub['innings'] == 1]['total_runs'].values
    m2, m1 = np.mean(inn2_v), np.mean(inn1_v)
    s2, s1 = np.var(inn2_v, ddof=1), np.var(inn1_v, ddof=1)
    n2, n1 = len(inn2_v), len(inn1_v)
    diff1 = m2 - m1
    se1 = np.sqrt(s2/n2 + s1/n1)
    df1 = (s2/n2 + s1/n1)**2 / (((s2/n2)**2)/(n2-1) + ((s1/n1)**2)/(n1-1))
    t_h0_1 = (diff1 - 0.05) / se1
    p1_val = 1 - stats.t.cdf(t_h0_1, df1)
    t_crit1 = stats.t.ppf(0.975, df1)
    ci1_l, ci1_h = diff1 - t_crit1 * se1, diff1 + t_crit1 * se1

    print(f"\n>>> TEST 1: Welch's Two-Sample t-test (P1: Dew Chase)")
    print(f"    Estimate: {diff1:.4f} runs/ball (Inn2: {m2:.3f}, Inn1: {m1:.3f})")
    print(f"    95% CI: [{ci1_l:.4f}, {ci1_h:.4f}]")
    print(f"    t-stat: {t_h0_1:.2f}, df: {int(df1):,}, p-value: {p1_val:.4f}")

    death_b = df_clean[df_clean['phase'] == 'Death']['is_boundary']
    pp_b = df_clean[df_clean['phase'] == 'Powerplay']['is_boundary']
    x_d, n_d = death_b.sum(), len(death_b)
    x_p, n_p = pp_b.sum(), len(pp_b)
    p_d, p_p = x_d / n_d, x_p / n_p
    diff2 = p_d - p_p
    se_ci2 = np.sqrt(p_d*(1-p_d)/n_d + p_p*(1-p_p)/n_p)
    ci2_l, ci2_h = diff2 - 1.96 * se_ci2, diff2 + 1.96 * se_ci2
    z_stat2 = (diff2 - 0.02) / se_ci2
    p2_val = 1 - stats.norm.cdf(z_stat2)

    print(f"\n>>> TEST 2: Two-Proportion Z-test (P2: Boundary Rate Death vs Powerplay)")
    print(f"    Estimate: {diff2*100:+.2f}% (Death: {p_d*100:.2f}%, PP: {p_p*100:.2f}%)")
    print(f"    95% CI: [{ci2_l*100:.2f}%, {ci2_h*100:.2f}%]")
    print(f"    Z-stat: {z_stat2:.2f}, p-value: {p2_val:.4f}")

    ct3 = pd.crosstab(wickets_sub['wicket_type'], wickets_sub['phase']).loc[['caught', 'bowled', 'run out', 'lbw'], phases]
    chi3, p3_val, dof3, exp3 = stats.chi2_contingency(ct3)
    exp3_df = pd.DataFrame(exp3, index=ct3.index, columns=ct3.columns)
    ro_death_obs = ct3.loc['run out', 'Death']
    ro_death_exp = exp3_df.loc['run out', 'Death']
    death_total = ct3['Death'].sum()
    ro_death_obs_pct = ro_death_obs / death_total * 100
    ro_death_exp_pct = ro_death_exp / death_total * 100

    print(f"\n>>> TEST 3: Chi-Square Test of Independence (P3: Dismissal vs Phase)")
    print(f"    Chi2: {chi3:.2f}, df: {dof3}, p-value: {p3_val:.2e}")
    print(f"    Death Run Outs: Obs={ro_death_obs} ({ro_death_obs_pct:.1f}%) vs Exp={ro_death_exp:.1f} ({ro_death_exp_pct:.1f}%) -> Surge: +{ro_death_obs - ro_death_exp:.1f} (+{ro_death_obs_pct - ro_death_exp_pct:.1f}%)")

    v4_sub['outcome_simple'] = np.select(
        [v4_sub['is_boundary'] == 1, v4_sub['is_dot'] == 1, v4_sub['wicket_type'].notna()],
        ['Boundary', 'Dot', 'Wicket'],
        default='Strike Rotation'
    )
    ct4 = pd.crosstab(v4_sub['standard_venue'], v4_sub['outcome_simple']).loc[v4_names, ['Boundary', 'Dot', 'Strike Rotation', 'Wicket']]
    chi4, p4_val, dof4, exp4 = stats.chi2_contingency(ct4)
    exp4_df = pd.DataFrame(exp4, index=ct4.index, columns=ct4.columns)
    chin_b_obs = ct4.loc['M Chinnaswamy Stadium', 'Boundary']
    chin_b_exp = exp4_df.loc['M Chinnaswamy Stadium', 'Boundary']
    chep_b_obs = ct4.loc['MA Chidambaram Stadium', 'Boundary']
    chep_b_exp = exp4_df.loc['MA Chidambaram Stadium', 'Boundary']

    print(f"\n>>> TEST 4: Chi-Square Test of Independence (P4: Venue vs Outcome)")
    print(f"    Chi2: {chi4:.2f}, df: {dof4}, p-value: {p4_val:.2e}")
    print(f"    Chinnaswamy Boundaries: Obs={chin_b_obs} vs Exp={chin_b_exp:.1f} (Diff: +{chin_b_obs - chin_b_exp:.1f})")
    print(f"    Chepauk Boundaries: Obs={chep_b_obs} vs Exp={chep_b_exp:.1f} (Diff: {chep_b_obs - chep_b_exp:.1f})")

    snip_path = os.path.join(REPORT_DIR, 'latex_table_snippets.txt')
    with open(snip_path, 'w', encoding='utf-8') as f:
        f.write(f"Test 1: Estimate={diff1:.2f}, CI=[{ci1_l:.3f}, {ci1_h:.3f}], t={t_h0_1:.2f}, df={int(df1)}, p={p1_val:.3f}\n")
        f.write(f"Test 2: Estimate={diff2*100:+.2f}%, CI=[{ci2_l*100:.2f}%, {ci2_h*100:.2f}%], Z={z_stat2:.2f}, p={p2_val:.3f}\n")
        f.write(f"Test 3: Chi2={chi3:.2f}, df={dof3}, p={p3_val:.2e}, Death Run Outs Obs={ro_death_obs} vs Exp={ro_death_exp:.1f}\n")
        f.write(f"Test 4: Chi2={chi4:.2f}, df={dof4}, p={p4_val:.2e}, Chinnaswamy Boundary Diff=+{chin_b_obs - chin_b_exp:.1f}\n")
    print(f"\n[4] Saved summary snippets to: {snip_path}")
    print("=" * 80)
    print("ALL 4 TESTS AND 4 PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    main()
