import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaVigilance · Signal Intelligence",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #080f1a;
    color: #c9d8e8;
}
.stApp { background-color: #080f1a; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0a1628 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] .stMarkdown { color: #7ba3c8; }

.dash-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #0a2040 50%, #061428 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.dash-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,168,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.dash-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem; font-weight: 600;
    color: #00a8ff; letter-spacing: -0.5px;
    margin: 0 0 6px 0;
}
.dash-subtitle {
    font-size: 0.85rem; color: #4a7fa5;
    letter-spacing: 2px; text-transform: uppercase; font-weight: 600;
}
.dash-badge {
    display: inline-block;
    background: rgba(0,168,255,0.1);
    border: 1px solid rgba(0,168,255,0.3);
    color: #00a8ff; padding: 3px 10px;
    border-radius: 20px; font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-right: 8px; margin-top: 10px;
}
.kpi-card {
    background: linear-gradient(135deg, #0d1b2a, #0a1f35);
    border: 1px solid #1e3a5f; border-radius: 10px;
    padding: 20px 24px; position: relative; overflow: hidden;
}
.kpi-card::after {
    content: ''; position: absolute;
    bottom: 0; left: 0; right: 0; height: 3px;
    border-radius: 0 0 10px 10px;
}
.kpi-card.blue::after  { background: linear-gradient(90deg, #00a8ff, #0066cc); }
.kpi-card.green::after { background: linear-gradient(90deg, #00d4aa, #00a878); }
.kpi-card.red::after   { background: linear-gradient(90deg, #ff4757, #cc2233); }
.kpi-card.gold::after  { background: linear-gradient(90deg, #ffd700, #ffaa00); }
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem; font-weight: 600;
    line-height: 1; margin: 8px 0 4px 0;
}
.kpi-card.blue  .kpi-value { color: #00a8ff; }
.kpi-card.green .kpi-value { color: #00d4aa; }
.kpi-card.red   .kpi-value { color: #ff4757; }
.kpi-card.gold  .kpi-value { color: #ffd700; }
.kpi-label {
    font-size: 0.72rem; color: #4a7fa5;
    text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;
}
.kpi-sub {
    font-size: 0.78rem; color: #2d5a7a;
    margin-top: 4px; font-family: 'IBM Plex Mono', monospace;
}
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem; color: #00a8ff;
    text-transform: uppercase; letter-spacing: 2px;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px; margin: 24px 0 16px 0;
}
.stTabs [data-baseweb="tab-list"] {
    background-color: #0d1b2a;
    border-bottom: 1px solid #1e3a5f; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent; color: #4a7fa5;
    border-radius: 6px 6px 0 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem; letter-spacing: 1px;
}
.stTabs [aria-selected="true"] {
    background-color: rgba(0,168,255,0.1) !important;
    color: #00a8ff !important;
    border-bottom: 2px solid #00a8ff !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #1e3a5f; border-radius: 8px;
}
[data-testid="metric-container"] {
    background: #0d1b2a; border: 1px solid #1e3a5f;
    border-radius: 10px; padding: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Plot helpers ──────────────────────────────────────────────────────────────
PLOT_BG = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#0a1628',
    font=dict(family='IBM Plex Sans', color='#7ba3c8', size=12),
    margin=dict(l=20, r=20, t=40, b=20),
)

def styled(fig, height=400, **kwargs):
    """Apply dark theme to any figure. Pass extra layout kwargs directly."""
    fig.update_layout(**PLOT_BG, height=height, **kwargs)
    fig.update_xaxes(gridcolor='#1e3a5f', linecolor='#1e3a5f',
                     zerolinecolor='#1e3a5f')
    fig.update_yaxes(gridcolor='#1e3a5f', linecolor='#1e3a5f',
                     zerolinecolor='#1e3a5f')
    return fig

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_signals():
    for path in ["data/cleaned/signals_scored.csv",
                 "data/cleaned/signals_output.csv"]:
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_pairs_summary():
    path = "data/cleaned/drug_event_pairs.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, usecols=['drug', 'quarter'])
    return df.groupby(['drug','quarter']).size().reset_index(name='reports')

@st.cache_data
def load_demo():
    path = "data/cleaned/demo_clean.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    sample = pd.read_csv(path, nrows=1)
    want   = ['caseid','age','sex','reporter_country','quarter']
    cols   = [c for c in want if c in sample.columns]
    return pd.read_csv(path, usecols=cols, low_memory=False)

@st.cache_data
def load_model_comparison():
    for path in ["data/cleaned/model_comparison_external.csv",
                 "data/cleaned/model_comparison_rulebased.csv"]:
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()

# ── Load ──────────────────────────────────────────────────────────────────────
signals_df  = load_signals()
pairs_sum   = load_pairs_summary()
demo_df     = load_demo()
model_df    = load_model_comparison()

if len(signals_df) == 0:
    st.error("No signal data found. Run src/signal_detection.py first.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <div class="dash-subtitle">FDA FAERS · Real-World Safety Intelligence</div>
  <div class="dash-title">⚕ PharmaVigilance Signal Detection</div>
  <span class="dash-badge">ROR</span>
  <span class="dash-badge">PRR</span>
  <span class="dash-badge">IC / WHO-UMC</span>
  <span class="dash-badge">ML RANKED</span>
  <span class="dash-badge">5 QUARTERS</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
available_drugs = sorted(signals_df['drug'].unique().tolist())

with st.sidebar:
    st.markdown('<div class="section-header">Drug Selection</div>',
                unsafe_allow_html=True)
    selected_drug = st.selectbox("Primary Drug", available_drugs,
                                 label_visibility="collapsed")

    st.markdown('<div class="section-header">Comparison Drug</div>',
                unsafe_allow_html=True)

    other_drugs    = [d for d in available_drugs if d != selected_drug]
    compare_options = ["None"] + other_drugs
    compare_drug   = st.selectbox("Compare with", compare_options,
                                  label_visibility="collapsed")

    st.markdown('<div class="section-header">Signal Filters</div>',
                unsafe_allow_html=True)
    min_cases     = st.slider("Min. Cases (N)", 1, 100, 3)
    methods_agree = st.slider("Min. Methods Agreeing", 0, 3, 2)
    top_n         = st.slider("Top N Signals", 10, 100, 25)

    st.markdown('<div class="section-header">Thresholds</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem;color:#4a7fa5;line-height:1.8;">
    🔵 <b style="color:#7ba3c8">ROR</b> — Lower 95% CI > 1<br>
    🟡 <b style="color:#7ba3c8">PRR</b> — ≥ 2, Chi² ≥ 4, N ≥ 3<br>
    🟢 <b style="color:#7ba3c8">IC025</b> — > 0 (WHO-UMC)<br>
    🔴 <b style="color:#7ba3c8">Strong</b> — All 3 methods
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Dataset</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;
                color:#4a7fa5;line-height:2;">
    QUARTERS · 25Q1–26Q1<br>
    CASES · 1.8M deduplicated<br>
    PAIRS · 14.6M drug-event<br>
    DRUGS · {len(available_drugs):,} in signals<br>
    EVENTS · 17,356 MedDRA PT
    </div>""", unsafe_allow_html=True)

# ── Filter helpers ────────────────────────────────────────────────────────────
def get_drug_signals(drug):
    return signals_df[signals_df['drug'] == drug].copy()

def apply_filters(df):
    return df[
        (df['n_cases'] >= min_cases) &
        (df['signal_strength'] >= methods_agree)
    ].copy()

drug_all  = get_drug_signals(selected_drug)
filtered  = apply_filters(drug_all)

do_compare   = compare_drug != "None"
drug_all_cmp = get_drug_signals(compare_drug) if do_compare else pd.DataFrame()
filtered_cmp = apply_filters(drug_all_cmp)    if do_compare else pd.DataFrame()

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
strong = int((drug_all['signal_strength'] == 3).sum())
pct    = strong / len(drug_all) * 100 if len(drug_all) > 0 else 0
max_ror = drug_all['ror'].max() if len(drug_all) > 0 else 0

for col, card_class, label, value, sub in [
    (k1, "blue",  "Events Analysed",    f"{len(drug_all):,}",  "MedDRA preferred terms"),
    (k2, "green", "Strong Signals",      f"{strong:,}",         "All 3 methods agree"),
    (k3, "red",   "Max ROR",             f"{max_ror:.0f}",      "Highest disproportionality"),
    (k4, "gold",  "Signal Rate",         f"{pct:.1f}%",         "of events are signals"),
    (k5, "blue",  "Filtered Results",    f"{len(filtered):,}",  "after current filters"),
]:
    with col:
        st.markdown(f"""
        <div class="kpi-card {card_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋  Signal Table",
    "🌲  Forest Plot",
    "🫧  Bubble Chart",
    "⚖️  Drug Comparison",
    "👥  Demographics",
    "🤖  ML Models",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Signal Table
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([2, 1])

    with left:
        st.markdown(f'<div class="section-header">Signal Table — {selected_drug}</div>',
                    unsafe_allow_html=True)
        if len(filtered) == 0:
            st.info("No signals match current filters.")
        else:
            display_cols = ['event','n_cases','ror','ror_lower_ci',
                            'ror_upper_ci','prr','ic025','signal_strength']
            if 'ml_score' in filtered.columns:
                display_cols.append('ml_score')
            show = filtered.sort_values('signal_strength',
                                        ascending=False).head(top_n)
            st.dataframe(show[display_cols], use_container_width=True,
                         hide_index=True, height=420)
            st.download_button("⬇️ Download CSV",
                               filtered.to_csv(index=False),
                               f"signals_{selected_drug}.csv", "text/csv")

    with right:
        st.markdown('<div class="section-header">Strength Breakdown</div>',
                    unsafe_allow_html=True)
        sc = drug_all['signal_strength'].value_counts().sort_index()
        colors_donut = {0:'#1e3a5f', 1:'#ff8c42', 2:'#ffd700', 3:'#00d4aa'}
        fig_donut = go.Figure(go.Pie(
            labels=[f"Strength {i}" for i in sc.index],
            values=sc.values,
            hole=0.65,
            marker=dict(colors=[colors_donut.get(i,'#1e3a5f') for i in sc.index]),
            textinfo='percent',
            textfont=dict(size=11, family='IBM Plex Mono'),
            hovertemplate="<b>%{label}</b><br>%{value} events<extra></extra>"
        ))
        fig_donut.add_annotation(
            text=f"<b>{strong}</b><br>STRONG",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#00d4aa', family='IBM Plex Mono')
        )
        styled(fig_donut, height=260, showlegend=True,
               legend=dict(font=dict(size=10)))
        st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown('<div class="section-header">Top 5 by ROR</div>',
                    unsafe_allow_html=True)
        top5 = drug_all.nlargest(5,'ror')[['event','ror','n_cases']].copy()
        top5['ror'] = top5['ror'].round(1)
        st.dataframe(top5, use_container_width=True, hide_index=True, height=210)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Forest Plot
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="section-header">ROR Forest Plot — Top {top_n}</div>',
                unsafe_allow_html=True)

    fp_data = filtered.nlargest(top_n, 'ror').dropna(
        subset=['ror','ror_lower_ci','ror_upper_ci'])

    if len(fp_data) == 0:
        st.info("No signals to plot with current filters.")
    else:
        c_map = {3:'#00d4aa', 2:'#ffd700', 1:'#ff8c42', 0:'#2d5a7a'}
        pt_colors = [c_map.get(int(s),'#2d5a7a') for s in fp_data['signal_strength']]

        fig_forest = go.Figure()
        fig_forest.add_trace(go.Scatter(
            x=fp_data['ror'],
            y=fp_data['event'],
            mode='markers',
            marker=dict(size=11, color=pt_colors,
                        line=dict(width=1, color='rgba(255,255,255,0.2)')),
            error_x=dict(
                type='data', symmetric=False,
                array=(fp_data['ror_upper_ci'] - fp_data['ror']).tolist(),
                arrayminus=(fp_data['ror'] - fp_data['ror_lower_ci']).tolist(),
                color='rgba(100,160,220,0.4)', thickness=1.5
            ),
            text=fp_data.apply(
                lambda r: f"n={int(r['n_cases'])} | PRR={r['prr']:.1f}", axis=1),
            hovertemplate="<b>%{y}</b><br>ROR: %{x:.2f}<br>%{text}<extra></extra>",
        ))
        fig_forest.add_vline(x=1, line_dash="dot", line_color="#ff4757",
                              line_width=1.5,
                              annotation_text="threshold ROR=1",
                              annotation_font_color="#ff4757",
                              annotation_font_size=11)
        styled(fig_forest,
               height=max(500, len(fp_data) * 22),
               xaxis_title="Reporting Odds Ratio (ROR) with 95% CI")
        fig_forest.update_xaxes(type='log')
        fig_forest.update_yaxes(autorange="reversed",
                                tickfont=dict(size=11))
        st.plotly_chart(fig_forest, use_container_width=True)
        st.caption("🟢 Strong · 🟡 Moderate · 🟠 Weak · Log scale X-axis")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Bubble Chart
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">ROR vs IC025 Bubble Chart</div>',
                unsafe_allow_html=True)

    bubble_data = drug_all.nlargest(50, 'n_cases').dropna(
        subset=['ror','ic025','prr'])

    if len(bubble_data) == 0:
        st.info("No data available.")
    else:
        fig_bubble = go.Figure()
        for strength, color in [(3,'#00d4aa'),(2,'#ffd700'),
                                 (1,'#ff8c42'),(0,'#2d5a7a')]:
            sub = bubble_data[bubble_data['signal_strength'] == strength]
            if len(sub) == 0:
                continue
            fig_bubble.add_trace(go.Scatter(
                x=sub['ror'],
                y=sub['ic025'],
                mode='markers',
                name=f"Strength {strength}",
                marker=dict(
                    size=np.sqrt(sub['n_cases'].clip(lower=1)) * 1.5,
                    color=color, opacity=0.75,
                    line=dict(width=1, color='rgba(255,255,255,0.15)')
                ),
                text=sub['event'],
                customdata=sub[['n_cases','prr']].values,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "ROR: %{x:.2f} | IC025: %{y:.2f}<br>"
                    "N: %{customdata[0]} | PRR: %{customdata[1]:.2f}"
                    "<extra></extra>"
                )
            ))
        fig_bubble.add_hline(y=0, line_dash="dot", line_color="#ff4757",
                              annotation_text="IC025=0",
                              annotation_font_color="#ff4757")
        fig_bubble.add_vline(x=1, line_dash="dot", line_color="#ff4757",
                              annotation_text="ROR=1",
                              annotation_font_color="#ff4757")
        styled(fig_bubble, height=520,
               xaxis_title="Reporting Odds Ratio (ROR)",
               yaxis_title="Information Component (IC025)",
               legend=dict(bgcolor='rgba(0,0,0,0)',
                           bordercolor='#1e3a5f', borderwidth=1))
        fig_bubble.update_xaxes(type='log')
        st.plotly_chart(fig_bubble, use_container_width=True)
        st.caption("Bubble size = number of cases. Top 50 events. "
                   "Upper-right quadrant = strongest signals.")

    # Quarterly trend bar
    st.markdown('<div class="section-header">Quarterly Reporting Trend</div>',
                unsafe_allow_html=True)
    if len(pairs_sum) > 0:
        trend = pairs_sum[pairs_sum['drug'] == selected_drug]
        if len(trend) > 0:
            fig_trend = go.Figure(go.Bar(
                x=trend['quarter'], y=trend['reports'],
                marker=dict(
                    color=trend['reports'].tolist(),
                    colorscale=[[0,'#0a2040'],[0.5,'#0066cc'],[1,'#00a8ff']],
                    line=dict(width=0)
                ),
                text=trend['reports'].apply(lambda x: f"{x:,}"),
                textposition='outside',
                textfont=dict(family='IBM Plex Mono', size=11, color='#7ba3c8'),
                hovertemplate="<b>%{x}</b><br>Reports: %{y:,}<extra></extra>"
            ))
            styled(fig_trend, height=300,
                   xaxis_title="Quarter",
                   yaxis_title="Reports",
                   showlegend=False)
            st.plotly_chart(fig_trend, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Drug Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Drug-vs-Drug Signal Comparison</div>',
                unsafe_allow_html=True)

    if not do_compare:
        st.info("👈 Select a comparison drug in the sidebar to enable this view.")
    else:
        # KPI comparison
        c1, c2 = st.columns(2)
        d1_strong = int((drug_all['signal_strength'] == 3).sum())
        d2_strong = int((drug_all_cmp['signal_strength'] == 3).sum())

        with c1:
            st.markdown(f"""
            <div class="kpi-card blue" style="text-align:center">
                <div class="kpi-label">{selected_drug}</div>
                <div class="kpi-value" style="font-size:1.8rem">{d1_strong}</div>
                <div class="kpi-sub">strong signals · {len(drug_all):,} total</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-card green" style="text-align:center">
                <div class="kpi-label">{compare_drug}</div>
                <div class="kpi-value" style="font-size:1.8rem">{d2_strong}</div>
                <div class="kpi-sub">strong signals · {len(drug_all_cmp):,} total</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ROR distribution overlay
        st.markdown('<div class="section-header">ROR Distribution Overlay</div>',
                    unsafe_allow_html=True)
        fig_cmp = go.Figure()
        for drug_name, df_c, color in [
            (selected_drug, drug_all,     '#00a8ff'),
            (compare_drug,  drug_all_cmp, '#00d4aa'),
        ]:
            ror_vals = df_c['ror'].dropna()
            q95 = ror_vals.quantile(0.95)
            ror_vals = ror_vals[ror_vals <= q95]
            fig_cmp.add_trace(go.Histogram(
                x=ror_vals, name=drug_name,
                opacity=0.65, nbinsx=40,
                marker_color=color,
                hovertemplate=f"<b>{drug_name}</b><br>ROR: %{{x:.1f}}<br>Count: %{{y}}<extra></extra>"
            ))
        styled(fig_cmp, height=360,
               barmode='overlay',
               xaxis_title="Reporting Odds Ratio (ROR)",
               yaxis_title="Number of Events",
               legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.caption("Top 95th percentile shown to exclude extreme outliers.")

        # Shared signals
        st.markdown('<div class="section-header">Signal Overlap Analysis</div>',
                    unsafe_allow_html=True)
        events1 = set(drug_all[drug_all['signal_strength'] == 3]['event'])
        events2 = set(drug_all_cmp[drug_all_cmp['signal_strength'] == 3]['event'])
        shared  = events1 & events2

        ov1, ov2, ov3 = st.columns(3)
        ov1.metric(f"{selected_drug} only",  len(events1 - events2))
        ov2.metric("Shared signals",          len(shared))
        ov3.metric(f"{compare_drug} only",   len(events2 - events1))

        if shared:
            st.markdown('<div class="section-header">Shared Signal Details</div>',
                        unsafe_allow_html=True)
            d1_shared = (drug_all[drug_all['event'].isin(shared)]
                         [['event','ror','n_cases']]
                         .rename(columns={'ror': f'ROR_{selected_drug[:6]}',
                                          'n_cases': f'N_{selected_drug[:6]}'}))
            d2_shared = (drug_all_cmp[drug_all_cmp['event'].isin(shared)]
                         [['event','ror','n_cases']]
                         .rename(columns={'ror': f'ROR_{compare_drug[:6]}',
                                          'n_cases': f'N_{compare_drug[:6]}'}))
            shared_table = d1_shared.merge(d2_shared, on='event')
            st.dataframe(shared_table, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Demographics
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Patient Demographics</div>',
                unsafe_allow_html=True)

    if len(demo_df) == 0:
        st.info("Demographics file not found.")
    else:
        d1, d2 = st.columns(2)

        with d1:
            st.markdown('<div class="section-header">Sex Distribution</div>',
                        unsafe_allow_html=True)
            if 'sex' in demo_df.columns:
                sc = demo_df['sex'].value_counts().head(4)
                sc.index = [{'M':'Male','F':'Female',
                              'UNK':'Unknown','NS':'Not Specified'}.get(i,i)
                             for i in sc.index]
                fig_sex = go.Figure(go.Bar(
                    x=sc.index, y=sc.values,
                    marker_color=['#00a8ff','#ff6b9d','#4a7fa5','#2d5a7a'],
                    text=sc.values.tolist(), textposition='outside',
                    textfont=dict(family='IBM Plex Mono', size=11)
                ))
                styled(fig_sex, height=300,
                       yaxis_title="Cases", showlegend=False)
                st.plotly_chart(fig_sex, use_container_width=True)

        with d2:
            st.markdown('<div class="section-header">Age Distribution</div>',
                        unsafe_allow_html=True)
            if 'age' in demo_df.columns:
                ages = pd.to_numeric(demo_df['age'], errors='coerce').dropna()
                ages = ages[(ages > 0) & (ages < 120)]
                fig_age = go.Figure(go.Histogram(
                    x=ages, nbinsx=20,
                    marker=dict(color='#00a8ff', opacity=0.8,
                                line=dict(width=0))
                ))
                styled(fig_age, height=300,
                       xaxis_title="Age (years)",
                       yaxis_title="Cases", showlegend=False)
                st.plotly_chart(fig_age, use_container_width=True)

        # Country
        st.markdown('<div class="section-header">Top 15 Reporting Countries</div>',
                    unsafe_allow_html=True)
        if 'reporter_country' in demo_df.columns:
            cc = demo_df['reporter_country'].value_counts().head(15)
            fig_country = go.Figure(go.Bar(
                x=cc.values, y=cc.index,
                orientation='h',
                marker=dict(
                    color=cc.values.tolist(),
                    colorscale=[[0,'#0a2040'],[0.5,'#0066cc'],[1,'#00a8ff']],
                    line=dict(width=0)
                ),
                text=cc.values.tolist(), textposition='outside',
                textfont=dict(family='IBM Plex Mono', size=10)
            ))
            styled(fig_country, height=420,
                   xaxis_title="Reports", showlegend=False)
            fig_country.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_country, use_container_width=True)

        # Quarterly cases
        if 'quarter' in demo_df.columns:
            st.markdown('<div class="section-header">Cases Over Time</div>',
                        unsafe_allow_html=True)
            qc = demo_df['quarter'].value_counts().sort_index()
            fig_q = go.Figure(go.Scatter(
                x=qc.index, y=qc.values,
                mode='lines+markers',
                line=dict(color='#00a8ff', width=2),
                marker=dict(size=8, color='#00a8ff',
                            line=dict(width=2, color='#080f1a')),
                fill='tozeroy',
                fillcolor='rgba(0,168,255,0.08)',
            ))
            styled(fig_q, height=280,
                   xaxis_title="Quarter",
                   yaxis_title="Cases", showlegend=False)
            st.plotly_chart(fig_q, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ML Models
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">ML Model Performance</div>',
                unsafe_allow_html=True)

    if len(model_df) == 0:
        st.info("Run src/ml_classifier.py to generate model comparison data.")
    else:
        m1, m2 = st.columns([1, 1])

        with m1:
            st.dataframe(model_df, use_container_width=True,
                         hide_index=True, height=200)

            # AUC bar
            if 'AUC-ROC' in model_df.columns:
                fig_auc = go.Figure(go.Bar(
                    x=model_df['Model'],
                    y=model_df['AUC-ROC'],
                    marker=dict(
                        color=model_df['AUC-ROC'].tolist(),
                        colorscale=[[0,'#0a2040'],[0.7,'#0066cc'],[1,'#00d4aa']],
                        line=dict(width=0)
                    ),
                    text=model_df['AUC-ROC'].round(3).tolist(),
                    textposition='outside',
                    textfont=dict(family='IBM Plex Mono', size=12)
                ))
                fig_auc.add_hline(y=0.8, line_dash="dot",
                                   line_color="#ffd700",
                                   annotation_text="Good (0.8)",
                                   annotation_font_color="#ffd700")
                fig_auc.add_hline(y=0.9, line_dash="dot",
                                   line_color="#00d4aa",
                                   annotation_text="Excellent (0.9)",
                                   annotation_font_color="#00d4aa")
                styled(fig_auc, height=320,
                       yaxis_title="AUC-ROC",
                       showlegend=False)
                fig_auc.update_yaxes(range=[0, 1.15])
                st.plotly_chart(fig_auc, use_container_width=True)

        with m2:
            # Radar chart
            metrics_avail = [m for m in ['AUC-ROC','F1','Precision','Recall']
                              if m in model_df.columns]
            if len(metrics_avail) >= 3:
                colors_ml = ['#00d4aa','#00a8ff','#ffd700','#ff4757']
                fig_radar = go.Figure()
                for i, row in model_df.iterrows():
                    vals = [row[m] for m in metrics_avail]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=metrics_avail + [metrics_avail[0]],
                        fill='toself',
                        name=row['Model'],
                        line=dict(color=colors_ml[i % len(colors_ml)], width=2),
                        fillcolor=f"rgba({int(colors_ml[i%len(colors_ml)][1:3],16)},"
                                  f"{int(colors_ml[i%len(colors_ml)][3:5],16)},"
                                  f"{int(colors_ml[i%len(colors_ml)][5:7],16)},0.1)"
                    ))
                fig_radar.update_layout(
                    **PLOT_BG, height=380,
                    polar=dict(
                        bgcolor='#0a1628',
                        radialaxis=dict(
                            visible=True, range=[0,1],
                            gridcolor='#1e3a5f', linecolor='#1e3a5f',
                            tickfont=dict(size=9)
                        ),
                        angularaxis=dict(
                            gridcolor='#1e3a5f', linecolor='#1e3a5f'
                        )
                    ),
                    legend=dict(bgcolor='rgba(0,0,0,0)')
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("""
        <div style="background:rgba(0,168,255,0.05);
                    border:1px solid rgba(0,168,255,0.15);
                    border-radius:8px; padding:16px;
                    font-size:0.82rem; color:#7ba3c8;
                    line-height:1.8; margin-top:12px;">
        <b style="color:#00a8ff">⚕ Clinical Interpretation</b><br>
        Gradient Boosting (AUC=0.917) outperforms linear models, confirming the
        signal-noise boundary is <b style="color:#c9d8e8">non-linear</b> in the
        ROR/PRR/IC feature space. External validation (n=22) shows more realistic
        performance than rule-based labels — in production, 200+ validated signals
        from EMA/FDA safety communications would yield stable estimates.
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid #1e3a5f; padding-top:16px;
            font-family:'IBM Plex Mono',monospace; font-size:0.72rem;
            color:#2d5a7a; text-align:center;">
⚕ PharmaVigilance Signal Detection · FDA FAERS Real-World Data ·
ROR / PRR / IC (WHO-UMC) · Random Forest · Gradient Boosting ·
Built for AstraZeneca Real World Science
</div>""", unsafe_allow_html=True)