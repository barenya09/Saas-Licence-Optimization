"""
SaaS License Optimization Dashboard — Phase 8
Brings together all 6 build phases: SQL flagging, K-means clustering,
classification (at-risk early warning), regression (spend forecasting),
savings/ROI consolidation, and a Gemini-generated executive summary.
"""

import os
import json
import sqlite3

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="SaaS License Optimization", layout="wide", page_icon="💡")
INR = lambda x: f"Rs {x:,.0f}"

DATASET_FILENAME = "SaaS_License_Optimization_Dataset_Large.xlsx"


# ---------------------------------------------------------------------------
# Sidebar: dataset + Gemini key
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload dataset (.xlsx)", type=["xlsx"])
    try:
        default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATASET_FILENAME)
    except NameError:
        default_path = DATASET_FILENAME

    if uploaded is not None:
        xlsx_source = uploaded
        st.success(f"Using uploaded file: {uploaded.name}")
    elif os.path.exists(default_path):
        xlsx_source = default_path
        st.info("Using bundled sample dataset.")
    else:
        st.warning("No dataset found. Upload a .xlsx file to continue.")
        st.stop()

    st.divider()
    st.header("Executive Summary")
    gemini_key = st.text_input("Gemini API key (optional)", type="password",
                                help="Used only for this session to generate the executive "
                                     "summary with Gemini. Leave blank for the offline template.")
    st.caption("Key is held in memory for this session only — never written to disk or logged.")


# ---------------------------------------------------------------------------
# Full pipeline — cached so it only recomputes when the dataset changes
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Running the full pipeline (SQL flagging, clustering, "
                             "classification, regression, savings)...")
def run_pipeline(xlsx_source):
    xls = pd.ExcelFile(xlsx_source)
    employees = pd.read_excel(xls, "Employees")
    tools_catalog = pd.read_excel(xls, "Tools_Catalog")
    license_assignments = pd.read_excel(xls, "License_Assignments")
    usage_logs = pd.read_excel(xls, "Usage_Logs")

    employees["date_joined"] = pd.to_datetime(employees["date_joined"])
    employees["date_left"] = pd.to_datetime(employees["date_left"])
    license_assignments["assigned_date"] = pd.to_datetime(license_assignments["assigned_date"])
    license_assignments["renewal_date"] = pd.to_datetime(license_assignments["renewal_date"])
    usage_logs["login_date"] = pd.to_datetime(usage_logs["login_date"])

    # ---------------- Phase 2: License flagging (SQL) ----------------
    conn = sqlite3.connect(":memory:")
    employees.to_sql("Employees", conn, index=False, if_exists="replace")
    license_assignments.to_sql("License_Assignments", conn, index=False, if_exists="replace")
    usage_logs.to_sql("Usage_Logs", conn, index=False, if_exists="replace")

    ex_employee_df = pd.read_sql_query("""
        SELECT la.license_id, la.employee_id, e.employee_name, e.department,
               e.date_left, la.tool_id, la.monthly_cost
        FROM License_Assignments la JOIN Employees e ON la.employee_id = e.employee_id
        WHERE e.employment_status = 'Terminated' AND la.license_status = 'Active';
    """, conn)

    reference_date = usage_logs["login_date"].max()
    cutoff_date = reference_date - pd.Timedelta(days=90)
    dormant_df = pd.read_sql_query("""
        SELECT la.license_id, la.employee_id, la.tool_id, la.monthly_cost, ul.last_login_date
        FROM License_Assignments la
        LEFT JOIN (SELECT employee_id, tool_id, MAX(login_date) as last_login_date
                   FROM Usage_Logs GROUP BY employee_id, tool_id) ul
        ON la.employee_id = ul.employee_id AND la.tool_id = ul.tool_id
        WHERE la.license_status = 'Active'
          AND (ul.last_login_date IS NULL OR ul.last_login_date < ?);
    """, conn, params=(cutoff_date.strftime("%Y-%m-%d"),))

    all_flagged_ids = set(ex_employee_df["license_id"]) | set(dormant_df["license_id"])
    category_a_savings = int(license_assignments[
        license_assignments["license_id"].isin(all_flagged_ids)
    ]["monthly_cost"].sum())

    # ---------------- Phase 3: K-means clustering ----------------
    active_lic = license_assignments[license_assignments["license_status"] == "Active"].merge(
        employees[["employee_id", "department"]], on="employee_id", how="left"
    )
    dept_counts = active_lic.pivot_table(index="tool_id", columns="department",
                                          values="license_id", aggfunc="count", fill_value=0)
    dept_share = dept_counts.div(dept_counts.sum(axis=1), axis=0)
    usage_stats = usage_logs.groupby("tool_id").agg(
        avg_session_min=("session_duration_minutes", "mean"),
        avg_features_used=("features_used_count", "mean"))
    tool_features = dept_share.join(usage_stats).join(
        tools_catalog.set_index("tool_id")[["cost_per_seat_per_month"]]).fillna(0)

    X = StandardScaler().fit_transform(tool_features.values)
    sil_scores = {}
    for k in range(2, 13):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        sil_scores[k] = silhouette_score(X, km.labels_)
    best_k = max(sil_scores, key=sil_scores.get)
    final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X)
    tool_features["cluster"] = final_km.labels_
    tool_features = tool_features.join(tools_catalog.set_index("tool_id")[["tool_name", "category"]])

    sim = cosine_similarity(dept_share.values)
    sim_df = pd.DataFrame(sim, index=dept_share.index, columns=dept_share.index)
    cat_map = tools_catalog.set_index("tool_id")["category"]
    name_map = tools_catalog.set_index("tool_id")["tool_name"]

    CATEGORY_VERDICTS = {
        "Customer Support": ("Redundant", "Both are ticketing/helpdesk platforms — same job, two vendors."),
        "Communication": ("Complementary", "Chat, video, and meetings are different functions, not duplicates."),
        "Design": ("Review", "Overlap needs manual confirmation."),
    }

    candidates = []
    ids = tool_features.index.tolist()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if (tool_features.loc[a, "cluster"] == tool_features.loc[b, "cluster"]
                    and cat_map[a] == cat_map[b] and sim_df.loc[a, b] >= 0.90):
                seats_a = set(active_lic[active_lic["tool_id"] == a]["employee_id"])
                seats_b = set(active_lic[active_lic["tool_id"] == b]["employee_id"])
                both = seats_a & seats_b
                verdict, rationale = CATEGORY_VERDICTS.get(cat_map[a], ("Review", "Needs manual confirmation."))
                candidates.append({
                    "tool_a": name_map[a], "tool_b": name_map[b], "category": cat_map[a],
                    "tool_a_id": a, "tool_b_id": b,
                    "similarity": round(sim_df.loc[a, b], 3), "employees_both": len(both),
                    "duplicate_seats": 2 * len(both), "verdict": verdict, "rationale": rationale,
                })
    candidates_df = pd.DataFrame(candidates).sort_values("similarity", ascending=False).reset_index(drop=True)

    # Category B: incremental savings from the Redundant-verdict pair only
    category_b_savings = 0
    category_b_detail = None
    redundant = candidates_df[candidates_df["verdict"] == "Redundant"]
    if not redundant.empty:
        row = redundant.iloc[0]
        a_id, b_id = row["tool_a_id"], row["tool_b_id"]
        eng_a = tool_features.loc[a_id, "avg_session_min"]
        eng_b = tool_features.loc[b_id, "avg_session_min"]
        keep_id, drop_id = (a_id, b_id) if eng_a >= eng_b else (b_id, a_id)
        keep_name, drop_name = (row["tool_a"], row["tool_b"]) if eng_a >= eng_b else (row["tool_b"], row["tool_a"])

        keep_seats = active_lic[active_lic["tool_id"] == keep_id][["employee_id", "license_id"]].rename(
            columns={"license_id": "keep_license"})
        drop_seats = active_lic[active_lic["tool_id"] == drop_id][["employee_id", "license_id", "monthly_cost"]].rename(
            columns={"license_id": "drop_license"})
        both_df = keep_seats.merge(drop_seats, on="employee_id")
        both_df["keep_flagged"] = both_df["keep_license"].isin(all_flagged_ids)
        both_df["drop_flagged"] = both_df["drop_license"].isin(all_flagged_ids)
        genuinely_new = both_df[(~both_df["keep_flagged"]) & (~both_df["drop_flagged"])]
        category_b_savings = int(genuinely_new["monthly_cost"].sum())
        category_b_detail = {
            "pair": f"{row['tool_a']} / {row['tool_b']}", "keep": keep_name, "drop": drop_name,
            "incremental_seats": len(genuinely_new), "incremental_savings": category_b_savings,
        }

    # ---------------- Phase 4: Classification (at-risk early warning) ----------------
    start = usage_logs["login_date"].min()
    end = usage_logs["login_date"].max()
    midpoint = start + (end - start) / 2
    half_length = midpoint - start

    active = license_assignments[license_assignments["license_status"] == "Active"].copy()
    training_pop = active[active["assigned_date"] <= start].copy()
    first_half = usage_logs[usage_logs["login_date"] < midpoint]
    second_half = usage_logs[usage_logs["login_date"] >= midpoint]

    fh_agg = first_half.groupby(["employee_id", "tool_id"]).agg(
        login_count=("log_id", "count"), avg_session_duration=("session_duration_minutes", "mean"),
        avg_features_used=("features_used_count", "mean"), last_login=("login_date", "max")).reset_index()
    training_pop = training_pop.merge(fh_agg, on=["employee_id", "tool_id"], how="left")
    for col in ["login_count", "avg_session_duration", "avg_features_used"]:
        training_pop[col] = training_pop[col].fillna(0)
    training_pop["days_since_last_login"] = (midpoint - training_pop["last_login"]).dt.days
    training_pop["days_since_last_login"] = training_pop["days_since_last_login"].fillna(half_length.days)

    sh_pairs = set(zip(second_half["employee_id"], second_half["tool_id"]))
    training_pop["went_quiet"] = (~training_pop.apply(
        lambda r: (r["employee_id"], r["tool_id"]) in sh_pairs, axis=1)).astype(int)

    features = ["login_count", "avg_session_duration", "avg_features_used", "days_since_last_login"]
    X_clf = training_pop[features]
    y_clf = training_pop["went_quiet"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.25, random_state=42, stratify=y_clf)
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(random_state=42, class_weight="balanced")
    clf.fit(scaler.transform(X_train), y_train)
    test_accuracy = accuracy_score(y_test, clf.predict(scaler.transform(X_test)))

    # Retrain on full training population for the deployed model
    scaler_final = StandardScaler().fit(X_clf)
    clf_final = LogisticRegression(random_state=42, class_weight="balanced")
    clf_final.fit(scaler_final.transform(X_clf), y_clf)

    recent_start = end - half_length
    recent_usage = usage_logs[usage_logs["login_date"] >= recent_start]
    recent_agg = recent_usage.groupby(["employee_id", "tool_id"]).agg(
        login_count=("log_id", "count"), avg_session_duration=("session_duration_minutes", "mean"),
        avg_features_used=("features_used_count", "mean"), last_login=("login_date", "max")).reset_index()
    score_pop = active.merge(recent_agg, on=["employee_id", "tool_id"], how="left")
    for col in ["login_count", "avg_session_duration", "avg_features_used"]:
        score_pop[col] = score_pop[col].fillna(0)
    score_pop["days_since_last_login"] = (end - score_pop["last_login"]).dt.days
    score_pop["days_since_last_login"] = score_pop["days_since_last_login"].fillna(half_length.days)

    full_last_login = usage_logs.groupby(["employee_id", "tool_id"])["login_date"].max().reset_index()
    full_last_login.columns = ["employee_id", "tool_id", "true_last_login"]
    score_pop = score_pop.merge(full_last_login, on=["employee_id", "tool_id"], how="left")
    still_used = score_pop["true_last_login"].notna() & (score_pop["true_last_login"] >= (end - pd.Timedelta(days=90)))
    score_pop = score_pop[still_used].copy()

    score_pop["risk_score"] = clf_final.predict_proba(scaler_final.transform(score_pop[features]))[:, 1]

    ex_employee_ids = set(employees[employees["employment_status"] == "Terminated"]["employee_id"])
    high_risk = score_pop[score_pop["risk_score"] > 0.5].copy()
    watch_list = high_risk[~high_risk["employee_id"].isin(ex_employee_ids)].sort_values(
        "risk_score", ascending=False)
    watch_list = watch_list.merge(employees[["employee_id", "employee_name", "department"]],
                                   on="employee_id", how="left")
    watch_list = watch_list.merge(tools_catalog[["tool_id", "tool_name"]], on="tool_id", how="left")

    # ---------------- Phase 5: Regression (spend forecasting) ----------------
    active_all = license_assignments[license_assignments["license_status"] == "Active"].copy()
    active_all["assign_month"] = active_all["assigned_date"].dt.to_period("M")
    monthly_new_spend = active_all.groupby("assign_month")["monthly_cost"].sum().sort_index()
    cumulative_spend = monthly_new_spend.cumsum()
    recent_trend = cumulative_spend.tail(18)

    X_reg = np.arange(len(recent_trend)).reshape(-1, 1)
    y_reg = recent_trend.values
    poly = PolynomialFeatures(degree=2)
    X_reg_poly = poly.fit_transform(X_reg)
    reg_model = LinearRegression().fit(X_reg_poly, y_reg)

    future_months = np.arange(len(recent_trend), len(recent_trend) + 6).reshape(-1, 1)
    future_poly = poly.transform(future_months)
    forecast = reg_model.predict(future_poly)
    forecast_df = pd.DataFrame({"month_ahead": range(1, 7),
                                 "forecasted_total_spend": [round(v) for v in forecast]})

    # ---------------- Phase 6: Savings & ROI consolidation ----------------
    total_spend = int(active_lic["monthly_cost"].sum())
    total_monthly_savings = category_a_savings + category_b_savings
    waste_pct = 100 * total_monthly_savings / total_spend

    BUILD_HOURS, BUILD_RATE = 100, 1500
    ROLLOUT_MIN_PER_LIC, ROLLOUT_RATE = 8, 900
    ONGOING_HOURS, ONGOING_RATE = 4, 900

    total_actioned = len(all_flagged_ids) + (category_b_detail["incremental_seats"] if category_b_detail else 0)
    one_time_cost = BUILD_HOURS * BUILD_RATE + (total_actioned * ROLLOUT_MIN_PER_LIC / 60) * ROLLOUT_RATE
    ongoing_monthly_cost = ONGOING_HOURS * ONGOING_RATE
    monthly_net = total_monthly_savings - ongoing_monthly_cost
    payback_months = one_time_cost / monthly_net if monthly_net > 0 else float("inf")
    year1_gross = total_monthly_savings * 12
    year1_cost = one_time_cost + ongoing_monthly_cost * 12
    roi_multiple = year1_gross / year1_cost if year1_cost > 0 else float("inf")

    return {
        "employees": employees, "tools_catalog": tools_catalog,
        "license_assignments": license_assignments, "usage_logs": usage_logs,
        "dormant_df": dormant_df, "ex_employee_df": ex_employee_df, "all_flagged_ids": all_flagged_ids,
        "reference_date": reference_date, "cutoff_date": cutoff_date,
        "category_a_savings": category_a_savings,
        "tool_features": tool_features, "sil_scores": sil_scores, "best_k": best_k,
        "candidates_df": candidates_df, "category_b_savings": category_b_savings,
        "category_b_detail": category_b_detail,
        "test_accuracy": test_accuracy, "watch_list": watch_list,
        "cumulative_spend": cumulative_spend, "recent_trend": recent_trend, "forecast_df": forecast_df,
        "total_spend": total_spend, "total_monthly_savings": total_monthly_savings, "waste_pct": waste_pct,
        "one_time_cost": one_time_cost, "ongoing_monthly_cost": ongoing_monthly_cost,
        "payback_months": payback_months, "roi_multiple": roi_multiple,
    }


R = run_pipeline(xlsx_source)

st.title("💡 SaaS License Optimization Dashboard")
st.caption("Automated usage, billing, and HR-record audit — NMIMS Digital Transformation project")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Monthly SaaS Spend", INR(R["total_spend"]))
k2.metric("Confirmed Monthly Waste", INR(R["total_monthly_savings"]),
          delta=f"-{R['waste_pct']:.1f}% of spend", delta_color="inverse")
k3.metric("Confirmed Annual Savings", INR(R["total_monthly_savings"] * 12))
k4.metric("Year-1 ROI", f"{R['roi_multiple']:.2f}x")

st.divider()

tabs = st.tabs([
    "📋 Executive Summary", "😴 Dormant Licenses", "🚪 Ex-Employee Licenses",
    "🔗 Duplicate Tools", "⚠️ At-Risk Watch List", "📈 Savings, ROI & Forecast",
    "✅ Prioritized Actions",
])

# ---------------------------------------------------------------------------
# TAB 1 — Executive Summary
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Executive Summary")

    findings_payload = {
        "total_monthly_spend_inr": R["total_spend"],
        "dormant_licenses": {"license_count": len(R["dormant_df"]),
                              "monthly_savings_inr": int(R["dormant_df"]["monthly_cost"].sum())},
        "ex_employee_licenses": {"license_count": len(R["ex_employee_df"]),
                                  "monthly_savings_inr": int(R["ex_employee_df"]["monthly_cost"].sum())},
        "combined_note": f"{len(R['all_flagged_ids'])} total distinct licenses across both categories "
                          f"above (some overlap, so the two counts don't simply add up)",
        "duplicate_tool_consolidation": R["category_b_detail"] or {},
        "total_confirmed_monthly_savings_inr": R["total_monthly_savings"],
        "total_confirmed_annual_savings_inr": R["total_monthly_savings"] * 12,
        "waste_pct_of_spend": round(R["waste_pct"], 1),
        "at_risk_watch_list": {"license_count": len(R["watch_list"]),
                                "monthly_cost_at_risk_inr": int(R["watch_list"]["monthly_cost"].sum()),
                                "note": "Not yet wasted — early warning only, not counted in savings"},
        "spend_forecast_6_months_inr": int(R["forecast_df"]["forecasted_total_spend"].iloc[-1]),
        "roi": {"payback_months": round(R["payback_months"], 1),
                "year1_roi_multiple": round(R["roi_multiple"], 2)},
    }

    def build_offline_summary(payload):
        dorm, ex = payload["dormant_licenses"], payload["ex_employee_licenses"]
        dup = payload["duplicate_tool_consolidation"]
        watch = payload["at_risk_watch_list"]
        lines = [
            "## Headline",
            f"This audit identifies **Rs {payload['total_confirmed_monthly_savings_inr']:,}/month** "
            f"({payload['waste_pct_of_spend']}% of total spend) in recoverable SaaS costs.\n",
            "## Key Findings",
            f"- {dorm['license_count']} dormant licenses waste Rs {dorm['monthly_savings_inr']:,}/month.",
            f"- {ex['license_count']} licenses remain active for departed employees, "
            f"costing Rs {ex['monthly_savings_inr']:,}/month.",
        ]
        if dup:
            lines.append(f"- {dup['pair']} duplication: Rs {dup['incremental_savings']:,}/month "
                          f"in additional savings by consolidating onto {dup['keep']}.")
        lines += [
            f"- {watch['license_count']} more licenses show early signs of going unused "
            f"(Rs {watch['monthly_cost_at_risk_inr']:,}/month at risk — not counted as savings).",
            f"- If unchanged, spend is projected to reach "
            f"Rs {payload['spend_forecast_6_months_inr']:,}/month within 6 months.\n",
            "## Recommended Actions",
            f"1. IT/Finance to revoke the {dorm['license_count']} dormant and "
            f"{ex['license_count']} ex-employee licenses.",
        ]
        if dup:
            lines.append(f"2. Migrate remaining {dup['drop']} users onto {dup['keep']}.")
        lines += [
            "3. Check in with employees on the at-risk watch list before licenses go fully dormant.\n",
            "## Methodology Note",
            "These figures come from an automated audit of usage, billing, and HR records. "
            "The ROI figure depends on stated implementation-effort assumptions.",
        ]
        return "\n".join(lines)

    generate_live = st.button("Generate with Gemini", disabled=not gemini_key)
    if generate_live and gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            system_instruction = (
                "You are a SaaS cost-optimization analyst writing an executive summary for "
                "company leadership. You will be given a JSON object of already-computed, "
                "already-validated findings. Do not invent or round figures beyond what's given. "
                "Structure your response with markdown headers: Headline, Key Findings, "
                "Recommended Actions, Methodology Note. Under 350 words. No hype language."
            )
            user_prompt = f"Findings JSON:\n\n{json.dumps(findings_payload, indent=2)}"
            with st.spinner("Calling Gemini..."):
                response = client.models.generate_content(
                    model="gemini-3.6-flash", contents=user_prompt,
                    config={"system_instruction": system_instruction},
                )
            summary = response.text
            st.caption("Generated live via the Gemini API for this session.")
        except Exception as exc:
            st.error(f"Live generation failed ({exc}). Showing the offline summary instead.")
            summary = build_offline_summary(findings_payload)
    else:
        summary = build_offline_summary(findings_payload)
        st.caption("Offline template summary. Add a Gemini API key in the sidebar to generate "
                   "this live instead.")
    st.markdown(summary)

# ---------------------------------------------------------------------------
# TAB 2 — Dormant Licenses
# ---------------------------------------------------------------------------
with tabs[1]:
    dorm = R["dormant_df"].merge(R["employees"][["employee_id", "department", "employee_name"]],
                                  on="employee_id", how="left") \
                           .merge(R["tools_catalog"][["tool_id", "tool_name"]], on="tool_id", how="left")
    st.subheader(f"{len(dorm)} Dormant Licenses ({INR(dorm['monthly_cost'].sum())}/month)")
    st.caption(f"No login in the 90 days before {R['reference_date'].date()} "
               f"(cutoff: {R['cutoff_date'].date()}).")

    c1, c2 = st.columns(2)
    with c1:
        by_dept = dorm.groupby("department")["monthly_cost"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(by_dept, x="monthly_cost", y="department", orientation="h",
                     title="Dormant waste by department", labels={"monthly_cost": "Rs/month", "department": ""})
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")
    with c2:
        by_tool = dorm.groupby("tool_name")["monthly_cost"].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(by_tool, x="monthly_cost", y="tool_name", orientation="h",
                     title="Top 10 tools by dormant waste", labels={"monthly_cost": "Rs/month", "tool_name": ""})
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

    st.dataframe(dorm[["license_id", "employee_name", "department", "tool_name", "monthly_cost"]]
                 .sort_values("monthly_cost", ascending=False), width="stretch", hide_index=True)
    st.download_button("Download dormant license list (CSV)",
                        dorm.to_csv(index=False).encode(), "dormant_licenses.csv", "text/csv")

# ---------------------------------------------------------------------------
# TAB 3 — Ex-Employee Active Licenses
# ---------------------------------------------------------------------------
with tabs[2]:
    ex = R["ex_employee_df"]
    st.subheader(f"{len(ex)} Active Licenses on Terminated Employees ({INR(ex['monthly_cost'].sum())}/month)")
    st.caption("Compliance/security risk: these employees have left but still hold active access.")

    c1, c2 = st.columns(2)
    with c1:
        by_dept = ex.groupby("department")["monthly_cost"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(by_dept, x="monthly_cost", y="department", orientation="h",
                     title="Ex-employee waste by department", labels={"monthly_cost": "Rs/month", "department": ""})
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")
    with c2:
        ex_tool = ex.merge(R["tools_catalog"][["tool_id", "tool_name"]], on="tool_id", how="left")
        by_tool = ex_tool.groupby("tool_name")["monthly_cost"].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(by_tool, x="monthly_cost", y="tool_name", orientation="h",
                     title="Top 10 tools held by ex-employees", labels={"monthly_cost": "Rs/month", "tool_name": ""})
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

    st.dataframe(ex[["license_id", "employee_name", "department", "date_left", "tool_id", "monthly_cost"]]
                 .sort_values("monthly_cost", ascending=False), width="stretch", hide_index=True)
    st.download_button("Download ex-employee license list (CSV)",
                        ex.to_csv(index=False).encode(), "ex_employee_active_licenses.csv", "text/csv")

# ---------------------------------------------------------------------------
# TAB 4 — Duplicate Tools (K-means)
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Duplicate/Overlapping Tool Detection (K-Means)")

    c1, c2 = st.columns([3, 2])
    with c1:
        from sklearn.decomposition import PCA
        feat_cols = [c for c in R["tool_features"].columns if c not in ("cluster", "tool_name", "category")]
        X_viz = StandardScaler().fit_transform(R["tool_features"][feat_cols].values)
        coords = PCA(n_components=2, random_state=42).fit_transform(X_viz)
        proj = pd.DataFrame(coords, columns=["pc1", "pc2"], index=R["tool_features"].index)
        proj["cluster"] = R["tool_features"]["cluster"].astype(str)
        proj["tool_name"] = R["tool_features"]["tool_name"]
        proj["category"] = R["tool_features"]["category"]
        fig = px.scatter(proj, x="pc1", y="pc2", color="cluster", text="tool_name", hover_data=["category"],
                          title="Tools clustered by department-usage fingerprint (PCA projection)")
        fig.update_traces(textposition="top center", marker=dict(size=12))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Silhouette score by k**")
        fig2, ax = plt.subplots(figsize=(4.2, 3.2))
        ks = list(R["sil_scores"].keys())
        vals = list(R["sil_scores"].values())
        ax.plot(ks, vals, marker="o", color="#4C6EF5")
        ax.axvline(R["best_k"], color="#E8590C", linestyle="--", linewidth=1, label=f"selected k={R['best_k']}")
        ax.set_xlabel("k"); ax.set_ylabel("Silhouette score"); ax.legend(fontsize=8)
        fig2.tight_layout()
        st.pyplot(fig2)

    st.markdown("#### Candidate duplicate pairs")
    cand = R["candidates_df"].copy()
    verdict_color = {"Redundant": "🔴", "Review": "🟡", "Complementary": "🟢"}
    cand["verdict"] = cand["verdict"].map(lambda v: f"{verdict_color.get(v, '')} {v}")
    st.dataframe(cand[["tool_a", "tool_b", "category", "similarity", "duplicate_seats", "verdict", "rationale"]],
                 width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# TAB 5 — At-Risk Watch List (Classification)
# ---------------------------------------------------------------------------
with tabs[4]:
    watch = R["watch_list"]
    st.subheader(f"{len(watch)} Licenses Showing Early Warning Signs "
                 f"({INR(watch['monthly_cost'].sum())}/month at risk)")
    st.caption(f"Model accuracy on held-out test data: {R['test_accuracy']*100:.1f}%. "
               "These are NOT confirmed waste — usage is declining but hasn't stopped. "
               "A watch list for managers, not a savings claim.")
    st.dataframe(watch[["employee_name", "department", "tool_name", "risk_score", "monthly_cost"]]
                 .sort_values("risk_score", ascending=False),
                 width="stretch", hide_index=True)
    st.download_button("Download at-risk watch list (CSV)",
                        watch.to_csv(index=False).encode(), "at_risk_watch_list.csv", "text/csv")

# ---------------------------------------------------------------------------
# TAB 6 — Savings, ROI & Forecast
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Savings Waterfall")
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute", "relative", "relative", "total"],
        x=["Total Spend", "Dormant + Ex-Employee", "Duplicate Consolidation", "Remaining Optimized Spend"],
        y=[R["total_spend"], -R["category_a_savings"], -R["category_b_savings"], 0],
        text=[INR(R["total_spend"]), f"-{INR(R['category_a_savings'])}", f"-{INR(R['category_b_savings'])}",
              INR(R["total_spend"] - R["total_monthly_savings"])],
        connector={"line": {"color": "rgb(180,180,180)"}},
        decreasing={"marker": {"color": "#2F9E44"}}, totals={"marker": {"color": "#4C6EF5"}},
    ))
    fig.update_layout(title="Monthly spend after confirmed savings", showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Spend Forecast: With vs. Without Action")
    hist = R["recent_trend"]
    forecast = R["forecast_df"]["forecasted_total_spend"].values
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(len(hist))), y=hist.values, mode="lines+markers", name="Actual spend"))
    fig2.add_trace(go.Scatter(x=list(range(len(hist), len(hist)+6)), y=forecast, mode="lines+markers",
                               name="Forecast (no action)", line=dict(dash="dash", color="orange")))
    optimized_forecast = forecast - R["total_monthly_savings"]
    fig2.add_trace(go.Scatter(x=list(range(len(hist), len(hist)+6)), y=optimized_forecast, mode="lines+markers",
                               name="Forecast (after acting on savings)", line=dict(dash="dot", color="green")))
    fig2.update_layout(title="Projected spend, next 6 months", xaxis_title="Month index", yaxis_title="Rs/month")
    st.plotly_chart(fig2, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Implementation cost assumptions")
        st.table(pd.DataFrame({
            "Item": ["One-time total", "Ongoing (per month)"],
            "Amount": [INR(R["one_time_cost"]), INR(R["ongoing_monthly_cost"])],
        }).set_index("Item"))
    with c2:
        st.markdown("#### ROI (calculated, not asserted)")
        m1, m2 = st.columns(2)
        m1.metric("Payback period", f"{R['payback_months']:.1f} months")
        m2.metric("Year-1 ROI", f"{R['roi_multiple']:.2f}x")

# ---------------------------------------------------------------------------
# TAB 7 — Prioritized Actions
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Prioritized, Quantified Action List")
    actions = []

    ex = R["ex_employee_df"]
    for dept, g in ex.groupby("department"):
        actions.append({"Priority": "Critical", "Category": "Compliance / Security",
                         "Action": f"Revoke {len(g)} active license(s) for terminated employees in {dept}",
                         "Monthly Impact (Rs)": int(g["monthly_cost"].sum()), "Owner": f"{dept} manager + IT"})

    ex_ids = set(ex["license_id"])
    dorm_all = R["dormant_df"].merge(R["tools_catalog"][["tool_id", "tool_name"]], on="tool_id", how="left")
    dorm_dedup = dorm_all[~dorm_all["license_id"].isin(ex_ids)]
    for tool_name, g in dorm_dedup.groupby("tool_name"):
        actions.append({"Priority": "High", "Category": "Dormant License Reclamation",
                         "Action": f"Cancel/reassign {len(g)} dormant {tool_name} seat(s)",
                         "Monthly Impact (Rs)": int(g["monthly_cost"].sum()), "Owner": "IT / Procurement"})

    if R["category_b_detail"]:
        d = R["category_b_detail"]
        actions.append({"Priority": "Medium", "Category": "Duplicate Tool Consolidation",
                         "Action": f"Migrate {d['incremental_seats']} user(s) from {d['drop']} to {d['keep']}",
                         "Monthly Impact (Rs)": d["incremental_savings"], "Owner": "IT lead"})

    actions.append({"Priority": "Low (Monitor)", "Category": "Early Warning",
                     "Action": f"Check in with {len(R['watch_list'])} users on the at-risk watch list",
                     "Monthly Impact (Rs)": 0, "Owner": "Department leads"})

    actions_df = pd.DataFrame(actions).sort_values("Monthly Impact (Rs)", ascending=False).reset_index(drop=True)
    actions_df.index += 1

    priority_filter = st.multiselect("Filter by priority", options=actions_df["Priority"].unique().tolist(),
                                      default=actions_df["Priority"].unique().tolist())
    st.dataframe(actions_df[actions_df["Priority"].isin(priority_filter)], width="stretch")
    st.download_button("Download full action list (CSV)",
                        actions_df.to_csv(index=True).encode(), "prioritized_actions.csv", "text/csv")
    st.caption(f"Total quantified impact: {INR(actions_df['Monthly Impact (Rs)'].sum())}/month "
               f"(matches confirmed savings — nothing double-counted).")
