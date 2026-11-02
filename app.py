import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="CloudIQ",
    page_icon="☁️",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR / "data" / "processed"

BLUE   = "#1B4F8A"
RED    = "#C0392B"
GREEN  = "#27AE60"
ORANGE = "#E67E22"
GREY   = "#95A5A6"

# ── loaders ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_all():
    kpis        = pd.read_csv(DATA / "kpis_summary.csv")
    monthly     = pd.read_csv(DATA / "monthly_revenue.csv",  parse_dates=["month"])
    growth      = pd.read_csv(DATA / "monthly_growth.csv",   parse_dates=["month"])
    categories  = pd.read_csv(DATA / "revenue_by_category.csv")
    countries   = pd.read_csv(DATA / "revenue_by_country.csv")
    products    = pd.read_csv(DATA / "top_products.csv")
    status      = pd.read_csv(DATA / "order_status.csv")
    returns     = pd.read_csv(DATA / "return_rate_by_category.csv")
    cohort      = pd.read_csv(DATA / "cohort_retention.csv",  parse_dates=["cohort_month"])
    new_ret     = pd.read_csv(DATA / "new_vs_returning.csv",  parse_dates=["month"])
    funnel      = pd.read_csv(DATA / "conversion_funnel.csv")
    funnel_src  = pd.read_csv(DATA / "funnel_by_source.csv")
    ttp         = pd.read_csv(DATA / "time_to_purchase.csv")
    rfm_seg     = pd.read_csv(DATA / "rfm_segment_summary.csv")
    rfm_scores  = pd.read_csv(DATA / "rfm_scores.csv")
    bqml_eval   = pd.read_csv(DATA / "bqml_evaluation.csv")
    bqml_feat   = pd.read_csv(DATA / "bqml_feature_weights.csv")
    preds       = pd.read_csv(DATA / "repeat_purchase_predictions.csv")
    return (kpis, monthly, growth, categories, countries, products,
            status, returns, cohort, new_ret, funnel, funnel_src,
            ttp, rfm_seg, rfm_scores, bqml_eval, bqml_feat, preds)

try:
    (kpis, monthly, growth, categories, countries, products,
     status, returns, cohort, new_ret, funnel, funnel_src,
     ttp, rfm_seg, rfm_scores, bqml_eval, bqml_feat, preds) = load_all()
    data_ok = True
except Exception as e:
    st.error(f"Could not load data: {e}. Place all CSV files in data/processed/.")
    st.stop()

k = kpis.iloc[0]

# ── sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("## ☁️ CloudIQ")
st.sidebar.markdown("**E-commerce Analytics on Google BigQuery**")
st.sidebar.markdown("Dataset: `thelook_ecommerce` (public)")
st.sidebar.markdown("---")
st.sidebar.metric("Total Revenue",    f"${k['total_revenue']:,.0f}")
st.sidebar.metric("Total Customers",  f"{int(k['total_customers']):,}")
st.sidebar.metric("Total Orders",     f"{int(k['total_orders']):,}")
st.sidebar.metric("Repeat Rate",      f"{k['repeat_rate_pct']:.1f}%")
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Data: {k['data_start']} → {k['data_end']}\n\n"
    "Built by Emmanuel Aghedo"
)

# ── tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Overview",
    "👥 Customer Analytics",
    "🔁 Conversion Funnel",
    "🎯 RFM Segmentation",
    "🤖 BigQuery ML",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    st.title("Executive Overview")
    st.markdown("Top-level business performance — queried directly from Google BigQuery.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Revenue",     f"${k['total_revenue']:,.0f}")
    c2.metric("Total Orders",      f"{int(k['total_orders']):,}")
    c3.metric("Unique Customers",  f"{int(k['total_customers']):,}")
    c4.metric("Avg Order Value",   f"${k['avg_order_value']:.2f}")
    c5.metric("Return Rate",       f"{k['return_rate_pct']:.1f}%")

    st.markdown("---")

    # Monthly revenue + orders
    col_l, col_r = st.columns([2, 1])
    with col_l:
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Scatter(
            x=monthly["month"], y=monthly["revenue"],
            mode="lines", fill="tozeroy",
            line=dict(color=BLUE, width=2.5),
            fillcolor="rgba(27,79,138,0.15)",
            name="Revenue",
        ))
        fig_rev.add_trace(go.Scatter(
            x=growth["month"], y=growth["revenue_3mo_avg"],
            mode="lines", line=dict(color=ORANGE, width=2, dash="dot"),
            name="3-month rolling avg",
        ))
        fig_rev.update_layout(
            title="Monthly Revenue Trend",
            xaxis_title="Month", yaxis_title="Revenue ($)",
            height=380, plot_bgcolor="#F9F9F9", paper_bgcolor="white",
            yaxis=dict(tickprefix="$"),
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    with col_r:
        fig_mom = go.Figure()
        colors_mom = [GREEN if v > 0 else RED
                      for v in growth["mom_growth_pct"].fillna(0)]
        fig_mom.add_trace(go.Bar(
            x=growth["month"], y=growth["mom_growth_pct"],
            marker_color=colors_mom, opacity=0.85,
        ))
        fig_mom.add_hline(y=0, line_color="black", line_width=1)
        fig_mom.update_layout(
            title="Month-over-Month Growth (%)",
            xaxis_title="", yaxis_title="MoM growth (%)",
            height=380, plot_bgcolor="#F9F9F9", paper_bgcolor="white",
            showlegend=False,
        )
        st.plotly_chart(fig_mom, use_container_width=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Revenue by Product Category")
        fig_cat = go.Figure(go.Bar(
            y=categories["category"],
            x=categories["revenue"],
            orientation="h",
            marker_color=BLUE, opacity=0.85,
            text=[f"${v:,.0f}  ({s:.1f}%)"
                  for v, s in zip(categories["revenue"],
                                  categories["revenue_share_pct"])],
            textposition="outside",
        ))
        fig_cat.update_layout(
            xaxis_title="Revenue ($)", height=400,
            plot_bgcolor="#F9F9F9", paper_bgcolor="white",
            xaxis=dict(tickprefix="$"), showlegend=False,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_b:
        st.subheader("Revenue by Country (Top 15)")
        fig_geo = go.Figure(go.Bar(
            y=countries["country"],
            x=countries["revenue"],
            orientation="h",
            marker_color=GREY, opacity=0.85,
        ))
        fig_geo.update_layout(
            xaxis_title="Revenue ($)", height=400,
            plot_bgcolor="#F9F9F9", paper_bgcolor="white",
            xaxis=dict(tickprefix="$"), showlegend=False,
        )
        st.plotly_chart(fig_geo, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 15 Products")
    st.dataframe(
        products[["product","category","brand","orders","revenue","avg_price"]],
        use_container_width=True, hide_index=True,
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("Order Status Breakdown")
        fig_status = go.Figure(go.Pie(
            labels=status["status"],
            values=status["orders"],
            hole=0.45,
            marker_colors=[GREEN, BLUE, ORANGE, RED, GREY][:len(status)],
        ))
        fig_status.update_layout(height=320, paper_bgcolor="white")
        st.plotly_chart(fig_status, use_container_width=True)

    with col_s2:
        st.subheader("Return Rate by Category")
        ret_sorted = returns.sort_values("return_rate_pct", ascending=True)
        fig_ret = go.Figure(go.Bar(
            y=ret_sorted["category"],
            x=ret_sorted["return_rate_pct"],
            orientation="h",
            marker_color=[RED if v > returns["return_rate_pct"].mean() else GREY
                          for v in ret_sorted["return_rate_pct"]],
            opacity=0.85,
        ))
        fig_ret.add_vline(x=returns["return_rate_pct"].mean(),
                          line_dash="dash", line_color=ORANGE,
                          annotation_text="Avg")
        fig_ret.update_layout(
            xaxis_title="Return rate (%)", height=320,
            plot_bgcolor="#F9F9F9", paper_bgcolor="white",
        )
        st.plotly_chart(fig_ret, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — CUSTOMER ANALYTICS
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.title("Customer Analytics")

    # ── Cohort heatmap ─────────────────────────────────────────────────────
    st.subheader("Cohort Retention Matrix")
    st.markdown(
        "Each row is a cohort (month of first purchase). "
        "Each column is months since first purchase. "
        "Values show what % of the cohort placed another order."
    )

    matrix = cohort.pivot_table(
        index="cohort_month", columns="period_number", values="retention_rate"
    )
    matrix.index = pd.to_datetime(matrix.index).strftime("%Y-%m")
    matrix = matrix.iloc[-18:, :13]

    fig_cohort = go.Figure(go.Heatmap(
        z=matrix.values,
        x=[f"M+{c}" for c in matrix.columns],
        y=matrix.index.tolist(),
        colorscale="Blues",
        text=[[f"{v:.0f}%" if not np.isnan(v) else ""
               for v in row] for row in matrix.values],
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="Retention %"),
    ))
    fig_cohort.update_layout(
        xaxis_title="Months since first purchase",
        yaxis_title="Cohort (first purchase month)",
        height=520,
        plot_bgcolor="#F9F9F9", paper_bgcolor="white",
    )
    st.plotly_chart(fig_cohort, use_container_width=True)

    # ── Average retention curve ────────────────────────────────────────────
    avg_ret = cohort.groupby("period_number")["retention_rate"].mean().reset_index()

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("Average Retention Curve")
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=avg_ret["period_number"], y=avg_ret["retention_rate"],
            mode="lines+markers",
            line=dict(color=BLUE, width=2.5),
            marker=dict(size=7),
            fill="tozeroy", fillcolor="rgba(27,79,138,0.1)",
        ))
        fig_curve.update_layout(
            xaxis_title="Months since first purchase",
            yaxis_title="Avg retention rate (%)",
            height=360, plot_bgcolor="#F9F9F9", paper_bgcolor="white",
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    with col_c2:
        st.subheader("New vs Returning Customers")
        fig_nr = go.Figure()
        fig_nr.add_trace(go.Bar(
            x=new_ret["month"], y=new_ret["new_customers"],
            name="New", marker_color=BLUE, opacity=0.85,
        ))
        fig_nr.add_trace(go.Bar(
            x=new_ret["month"], y=new_ret["returning_customers"],
            name="Returning", marker_color=GREEN, opacity=0.85,
        ))
        fig_nr.update_layout(
            barmode="stack",
            xaxis_title="Month", yaxis_title="Customers",
            height=360, plot_bgcolor="#F9F9F9", paper_bgcolor="white",
        )
        st.plotly_chart(fig_nr, use_container_width=True)

    # ── Time to purchase ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Time to First Purchase")
    cumulative = ttp["customers"].cumsum() / ttp["customers"].sum() * 100

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        fig_ttp = go.Figure(go.Bar(
            x=ttp["days_to_purchase"], y=ttp["customers"],
            marker_color=BLUE, opacity=0.75,
        ))
        fig_ttp.update_layout(
            title="Days from First Session to First Purchase",
            xaxis_title="Days", yaxis_title="Customers",
            height=340, plot_bgcolor="#F9F9F9", paper_bgcolor="white",
        )
        st.plotly_chart(fig_ttp, use_container_width=True)

    with col_t2:
        fig_cum = go.Figure(go.Scatter(
            x=ttp["days_to_purchase"], y=cumulative,
            mode="lines", line=dict(color=GREEN, width=2.5),
            fill="tozeroy", fillcolor="rgba(39,174,96,0.1)",
        ))
        for pct in [25, 50, 75, 90]:
            idx = (cumulative >= pct).idxmax()
            d   = ttp.loc[idx, "days_to_purchase"]
            fig_cum.add_vline(x=d, line_dash="dot", line_color=GREY)
            fig_cum.add_annotation(x=d, y=pct + 3,
                                   text=f"{pct}% by day {d}",
                                   showarrow=False, font=dict(size=9))
        fig_cum.update_layout(
            title="Cumulative % of Customers Who Purchased",
            xaxis_title="Days", yaxis_title="Cumulative %",
            height=340, plot_bgcolor="#F9F9F9", paper_bgcolor="white",
        )
        st.plotly_chart(fig_cum, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — CONVERSION FUNNEL
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    st.title("Conversion Funnel Analysis")

    stages = funnel["stage"].tolist()
    values = funnel["sessions"].tolist()
    top    = values[0]

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.subheader("Funnel Volume")
        fig_funnel = go.Figure(go.Funnel(
            y=stages,
            x=values,
            textinfo="value+percent initial",
            marker=dict(color=[BLUE, "#2471A3", "#2980B9",
                                "#5DADE2", "#85C1E9", "#AED6F1"][:len(stages)]),
        ))
        fig_funnel.update_layout(
            height=420, paper_bgcolor="white",
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

    with col_f2:
        st.subheader("Step-by-Step Drop-off")
        step_labels = [f"{stages[i][:12]}→{stages[i+1][:12]}"
                       for i in range(len(stages)-1)]
        step_rates  = [values[i+1] / values[i] * 100
                       if values[i] > 0 else 0
                       for i in range(len(values)-1)]
        drop_rates  = [100 - r for r in step_rates]

        fig_drop = go.Figure()
        fig_drop.add_trace(go.Bar(
            x=step_labels, y=step_rates,
            name="Converted", marker_color=GREEN, opacity=0.85,
        ))
        fig_drop.add_trace(go.Bar(
            x=step_labels, y=drop_rates,
            name="Dropped off", marker_color=RED, opacity=0.85,
        ))
        fig_drop.update_layout(
            barmode="stack",
            yaxis_title="% of step entrants",
            height=420, plot_bgcolor="#F9F9F9", paper_bgcolor="white",
        )
        st.plotly_chart(fig_drop, use_container_width=True)

    st.markdown("---")
    st.subheader("Funnel by Traffic Source")
    st.markdown("Which channels convert best from session to purchase?")

    fig_src = go.Figure()
    metrics_to_plot = ["product_view_rate", "add_to_cart_rate",
                       "purchase_rate", "cart_to_purchase_rate"]
    colors_src = [BLUE, ORANGE, GREEN, RED]
    labels_src = ["Product View Rate", "Add to Cart Rate",
                  "Purchase Rate", "Cart → Purchase Rate"]

    for metric, color, label in zip(metrics_to_plot, colors_src, labels_src):
        fig_src.add_trace(go.Bar(
            name=label,
            x=funnel_src["traffic_source"],
            y=funnel_src[metric],
            marker_color=color, opacity=0.85,
        ))

    fig_src.update_layout(
        barmode="group",
        yaxis_title="Rate (%)",
        height=420, plot_bgcolor="#F9F9F9", paper_bgcolor="white",
    )
    st.plotly_chart(fig_src, use_container_width=True)
    st.dataframe(funnel_src, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — RFM SEGMENTATION
# ════════════════════════════════════════════════════════════════════════════

with tab4:
    st.title("RFM Customer Segmentation")
    st.markdown(
        "Customers scored on **Recency**, **Frequency**, and **Monetary** value "
        "using `NTILE(5)` window functions in BigQuery SQL."
    )

    # KPI row
    total_customers = rfm_seg["customers"].sum()
    total_customers = total_customers if total_customers > 0 else 1
    champions       = rfm_seg[rfm_seg["segment"] == "Champions"]
    at_risk         = rfm_seg[rfm_seg["segment"] == "At Risk"]
    lost_seg        = rfm_seg[rfm_seg["segment"] == "Lost"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers",   f"{total_customers:,}")
    if not champions.empty:
        c2.metric("Champions",
                  f"{int(champions['customers'].values[0]):,}",
                  delta=f"{champions['customers'].values[0]/total_customers*100:.1f}% of base")
    if not at_risk.empty:
        c3.metric("At Risk",
                  f"{int(at_risk['customers'].values[0]):,}",
                  delta=f"{at_risk['customers'].values[0]/total_customers*100:.1f}% of base",
                  delta_color="inverse")
    if not lost_seg.empty:
        c4.metric("Lost",
                  f"{int(lost_seg['customers'].values[0]):,}",
                  delta=f"{lost_seg['customers'].values[0]/total_customers*100:.1f}% of base",
                  delta_color="inverse")

    st.markdown("---")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.subheader("Customers per Segment")
        seg_sorted = rfm_seg.sort_values("customers", ascending=True)
        seg_colors = []
        for s in seg_sorted["segment"]:
            if s == "Champions":           seg_colors.append(GREEN)
            elif s in ("Loyal Customers", "Potential Loyalists"): seg_colors.append(BLUE)
            elif s in ("At Risk", "Can't Lose Them"): seg_colors.append(ORANGE)
            elif s == "Lost":              seg_colors.append(RED)
            else:                          seg_colors.append(GREY)

        fig_seg = go.Figure(go.Bar(
            y=seg_sorted["segment"],
            x=seg_sorted["customers"],
            orientation="h",
            marker_color=seg_colors, opacity=0.85,
            text=[f"{v:,}  ({v/total_customers*100:.1f}%)" if total_customers > 0 else f"{v:,}"
                  for v in seg_sorted["customers"]],
            textposition="outside",
        ))
        fig_seg.update_layout(
            xaxis_title="Customers", height=420,
            plot_bgcolor="#F9F9F9", paper_bgcolor="white",
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_r2:
        st.subheader("Revenue Share by Segment")
        total_rev = rfm_seg["total_revenue"].sum()
        total_rev = total_rev if total_rev > 0 else 1
        fig_rev_seg = go.Figure(go.Pie(
            labels=rfm_seg["segment"],
            values=rfm_seg["total_revenue"],
            hole=0.4,
            textinfo="label+percent",
        ))
        fig_rev_seg.update_layout(height=420, paper_bgcolor="white")
        st.plotly_chart(fig_rev_seg, use_container_width=True)

    # ── Segment map ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Segment Map — Frequency vs Recency")
    st.markdown("Bubble size = revenue share. Ideal customers are bottom-right (frequent, recent).")

    fig_map = go.Figure()
    for _, row in rfm_seg.iterrows():
        fig_map.add_trace(go.Scatter(
            x=[row["avg_frequency"]],
            y=[row["avg_recency"]],
            mode="markers+text",
            text=[row["segment"]],
            textposition="top center",
            marker=dict(
                size=max(row["total_revenue"] / total_rev * 400, 10),
                color=row["avg_monetary"],
                colorscale="Blues",
                showscale=False,
                opacity=0.8,
                line=dict(color="white", width=1.5),
            ),
            name=row["segment"],
            showlegend=False,
        ))
    fig_map.update_layout(
        xaxis_title="Avg purchase frequency",
        yaxis_title="Avg recency (days — lower is more recent)",
        yaxis=dict(autorange="reversed"),
        height=450, plot_bgcolor="#F9F9F9", paper_bgcolor="white",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    st.subheader("Segment Profiles")
    display_seg = rfm_seg[["segment","customers","avg_recency","avg_frequency",
                            "avg_monetary","total_revenue",
                            "customer_share_pct","revenue_share_pct"]].copy()
    display_seg.columns = ["Segment","Customers","Avg Recency (days)",
                            "Avg Frequency","Avg Spend ($)","Total Revenue ($)",
                            "Customer %","Revenue %"]
    st.dataframe(display_seg.sort_values("Total Revenue ($)", ascending=False),
                 use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — BIGQUERY ML
# ════════════════════════════════════════════════════════════════════════════

with tab5:
    st.title("BigQuery ML — Repeat Purchase Prediction")
    st.markdown(
        "A logistic regression model trained entirely in BigQuery using SQL. "
        "Predicts whether a customer will make a second purchase within 90 days "
        "of their first order."
    )

    # ── Model metrics ──────────────────────────────────────────────────────
    ev = bqml_eval.iloc[0]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("ROC AUC",   f"{ev['roc_auc']:.3f}")
    c2.metric("Accuracy",  f"{ev['accuracy']:.3f}")
    c3.metric("F1 Score",  f"{ev['f1_score']:.3f}")
    c4.metric("Precision", f"{ev['precision']:.3f}")
    c5.metric("Recall",    f"{ev['recall']:.3f}")
    c6.metric("Log Loss",  f"{ev['log_loss']:.3f}")

    st.info(
        f"**AUC = {ev['roc_auc']:.3f}** — the model correctly ranks a repeat buyer "
        f"above a non-repeat buyer {ev['roc_auc']*100:.1f}% of the time. "
        f"High recall ({ev['recall']:.3f}) with moderate precision ({ev['precision']:.3f}) "
        f"reflects `auto_class_weights=TRUE`, which prioritises catching genuine repeat "
        f"buyers over minimising false positives — the right trade-off for a marketing use case."
    )

    st.markdown("---")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.subheader("Predicted Repeat-Purchase Probability")
        fig_prob = go.Figure(go.Histogram(
            x=preds["repeat_prob"], nbinsx=60,
            marker_color=BLUE, opacity=0.8,
        ))
        fig_prob.add_vline(x=0.5, line_dash="dash", line_color=RED,
                           annotation_text="Threshold 0.5")
        fig_prob.update_layout(
            xaxis_title="P(repeat purchase within 90 days)",
            yaxis_title="Number of customers",
            height=360, plot_bgcolor="#F9F9F9", paper_bgcolor="white",
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    with col_m2:
        st.subheader("Feature Weights")
        fw = bqml_feat.sort_values("weight", key=abs, ascending=True).tail(15)
        fig_fw = go.Figure(go.Bar(
            y=fw["feature"],
            x=fw["weight"],
            orientation="h",
            marker_color=[GREEN if w > 0 else RED for w in fw["weight"]],
            opacity=0.85,
        ))
        fig_fw.add_vline(x=0, line_color="black", line_width=1.5)
        fig_fw.update_layout(
            xaxis_title="Coefficient weight",
            height=360, plot_bgcolor="#F9F9F9", paper_bgcolor="white",
        )
        st.plotly_chart(fig_fw, use_container_width=True)

    # ── Predictions by segment ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Repeat Probability by Category & Geography")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        cat_avg = (preds.groupby("first_category")["repeat_prob"]
                   .mean().sort_values(ascending=True).reset_index())
        fig_cat_prob = go.Figure(go.Bar(
            y=cat_avg["first_category"], x=cat_avg["repeat_prob"],
            orientation="h",
            marker_color=[GREEN if v > preds["repeat_prob"].mean() else GREY
                          for v in cat_avg["repeat_prob"]],
            opacity=0.85,
        ))
        fig_cat_prob.add_vline(x=preds["repeat_prob"].mean(),
                                line_dash="dash", line_color=ORANGE,
                                annotation_text="Avg")
        fig_cat_prob.update_layout(
            title="Avg Repeat Prob by First Purchase Category",
            xaxis_title="Avg repeat probability",
            height=360, plot_bgcolor="#F9F9F9", paper_bgcolor="white",
        )
        st.plotly_chart(fig_cat_prob, use_container_width=True)

    with col_p2:
        country_avg = (preds.groupby("country")["repeat_prob"]
                       .mean().sort_values(ascending=False)
                       .head(12).reset_index())
        fig_geo_prob = go.Figure(go.Bar(
            x=country_avg["country"], y=country_avg["repeat_prob"],
            marker_color=BLUE, opacity=0.85,
        ))
        fig_geo_prob.add_hline(y=preds["repeat_prob"].mean(),
                                line_dash="dash", line_color=ORANGE,
                                annotation_text="Avg")
        fig_geo_prob.update_layout(
            title="Avg Repeat Prob by Country (Top 12)",
            yaxis_title="Avg repeat probability",
            height=360, plot_bgcolor="#F9F9F9", paper_bgcolor="white",
        )
        st.plotly_chart(fig_geo_prob, use_container_width=True)

    # ── Top prospects table ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Top 50 Highest-Probability Customers")
    st.markdown("These are the customers most worth targeting for a re-engagement campaign.")

    top_preds = preds.nlargest(50, "repeat_prob")[
        ["user_id","repeat_prob","predicted_label","actual_label",
         "first_category","country","first_order_value","traffic_source"]
    ].copy()
    top_preds.columns = ["User ID","Repeat Prob","Predicted","Actual",
                          "First Category","Country","First Order Value ($)","Traffic Source"]
    st.dataframe(top_preds, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("BigQuery ML — How It Works")
    with st.expander("View SQL used to train the model"):
        st.code("""
-- Train a logistic regression model entirely in BigQuery SQL
-- No Python ML library. No data export. Model lives in the warehouse.

CREATE OR REPLACE MODEL `cloudiq-analytics.cloudiq.repeat_buyer_model`
OPTIONS(
    model_type         = 'logistic_reg',
    input_label_cols   = ['repeat_buyer'],
    auto_class_weights = TRUE,
    max_iterations     = 20,
    l2_reg             = 0.1
) AS
SELECT
    age,
    gender,
    country,
    first_order_value,
    first_category,
    num_items,
    days_account_to_first_order,
    traffic_source,
    repeat_buyer
FROM `cloudiq-analytics.cloudiq.repeat_purchase_training`
        """, language="sql")
    st.caption("CloudIQ · TheLook E-commerce · Google BigQuery Public Dataset")
