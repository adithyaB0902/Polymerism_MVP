"""Streamlit view for the RSM analysis section (kept out of app.py)."""

import numpy as np
import pandas as pd

from .anova import model_warnings
from .box_behnken import FACTORS
from .quadratic_model import fit_quadratic, regression_diagnostics
from .surfaces import response_surface_grid

RESPONSE = "removal_percent"
LABELS = {
    "chitosan_wt_percent": "Chitosan (wt%)", "biochar_wt_percent": "Biochar (wt%)",
    "pH": "pH", "initial_pb_mg_l": "Initial Pb(II) (mg/L)",
}


def render_rsm_analysis(st, rows):
    """Render the RSM analysis for saved experimental rows (a list of dicts)."""
    factors = list(FACTORS)
    n_terms = 1 + 2 * len(factors) + len(factors) * (len(factors) - 1) // 2
    frame = pd.DataFrame(rows)
    if "data_source" in frame.columns:
        source_counts = frame["data_source"].fillna("experimental").value_counts().to_dict()
        st.caption("Data provenance: " + ", ".join(f"{key}={value}" for key, value in source_counts.items()))
        frame = frame[frame["data_source"].fillna("experimental").eq("experimental")]
        if not frame.empty:
            st.info("Only rows marked experimental are used for RSM fitting. Simulated and predicted rows are excluded.")
    complete = frame[factors + [RESPONSE]].apply(pd.to_numeric, errors="coerce").dropna() if len(frame) else frame
    if len(complete) <= n_terms:
        st.info(f"A full quadratic model in {len(factors)} factors has {n_terms} coefficients. "
                f"Save at least {n_terms + 1} complete measured runs (found {len(complete)}); "
                "the 29-run Box–Behnken design is the intended dataset.")
        return
    try:
        fit = fit_quadratic(frame, RESPONSE, factors)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.caption(f"Quadratic model fitted to {fit['n_runs']} saved experimental rows, in coded units "
               "(−1 / 0 / +1). Model outputs are predictions, not experimental validation.")
    cols = st.columns(5)
    for col, (label, value, fmt) in zip(cols, [
        ("R²", fit["r2"], "{:.4f}"), ("Adjusted R²", fit["adjusted_r2"], "{:.4f}"),
        ("Predicted R²", fit["predicted_r2"], "{:.4f}"),
        ("Adequate precision", fit["adequate_precision"], "{:.2f}"), ("C.V. (%)", fit["cv_percent"], "{:.2f}"),
    ]):
        col.metric(label, fmt.format(value))
    for message in model_warnings(fit):
        st.warning(message)

    st.markdown("**ANOVA**")
    st.dataframe(fit["anova_table"], width="stretch", hide_index=True)
    st.markdown("**Coefficients (coded units)**")
    st.dataframe(fit["coefficients"], width="stretch", hide_index=True)

    st.markdown("**Stationary point**")
    sp = fit["stationary_point"]
    if not sp["exists"]:
        st.write(sp["reason"])
    else:
        where = "inside" if sp["inside_design_region"] else "OUTSIDE"
        st.write(f"Nature: **{sp['nature']}**. Predicted removal there: **{sp['predicted_response']:.2f}%** "
                 f"({where} the studied region).")
        st.dataframe(pd.DataFrame([sp["actual"]]).rename(columns=LABELS), hide_index=True)

    _render_surface(st, fit)
    _render_diagnostics(st, fit)


def _render_surface(st, fit):
    import plotly.graph_objects as go
    factors = fit["factors"]
    st.markdown("**Response surface**")
    c1, c2 = st.columns(2)
    x_factor = c1.selectbox("X axis", factors, 0, format_func=LABELS.get, key="rsm_x")
    y_factor = c2.selectbox("Y axis", factors, 2, format_func=LABELS.get, key="rsm_y")
    if x_factor == y_factor:
        st.error("Choose two different factors.")
        return
    fixed = {}
    others = [f for f in factors if f not in (x_factor, y_factor)]
    for col, f in zip(st.columns(len(others)), others):
        low, center, high = fit["levels"][f]
        fixed[f] = col.slider(LABELS[f], low, high, center, key=f"rsm_fix_{f}")
    grid = response_surface_grid(fit, x_factor, y_factor, fixed)
    kind = st.radio("Plot type", ["Contour", "3D surface"], horizontal=True, key="rsm_plot_kind")
    if kind == "Contour":
        fig = go.Figure(go.Contour(x=grid["x"], y=grid["y"], z=grid["z"], colorscale="Viridis",
                                   colorbar_title="Removal (%)"))
        fig.update_layout(xaxis_title=LABELS[x_factor], yaxis_title=LABELS[y_factor])
    else:
        fig = go.Figure(go.Surface(x=grid["x"], y=grid["y"], z=grid["z"], colorscale="Viridis"))
        fig.update_layout(scene={"xaxis_title": LABELS[x_factor], "yaxis_title": LABELS[y_factor],
                                 "zaxis_title": "Removal (%)"})
    fig.update_layout(height=520, title="Predicted Pb(II) removal (model prediction)")
    st.plotly_chart(fig, width="stretch")


def _render_diagnostics(st, fit):
    import plotly.graph_objects as go
    from scipy import stats
    diag = regression_diagnostics(fit)
    st.markdown("**Diagnostics**")
    c1, c2, c3 = st.columns(3)
    f1 = go.Figure(go.Scatter(x=diag["predicted"], y=diag["studentized_residual"], mode="markers"))
    f1.add_hline(y=0)
    f1.update_layout(title="Studentized residuals vs predicted", height=340,
                     xaxis_title="Predicted", yaxis_title="Residual")
    c1.plotly_chart(f1, width="stretch")
    r = np.sort(diag["studentized_residual"].dropna().to_numpy())
    q = stats.norm.ppf((np.arange(1, len(r) + 1) - 0.5) / len(r))
    f2 = go.Figure(go.Scatter(x=q, y=r, mode="markers"))
    f2.add_shape(type="line", x0=q.min(), x1=q.max(), y0=q.min(), y1=q.max())
    f2.update_layout(title="Normal Q–Q", height=340, xaxis_title="Theoretical", yaxis_title="Sample")
    c2.plotly_chart(f2, width="stretch")
    lo, hi = float(diag[["actual", "predicted"]].min().min()), float(diag[["actual", "predicted"]].max().max())
    f3 = go.Figure(go.Scatter(x=diag["actual"], y=diag["predicted"], mode="markers"))
    f3.add_shape(type="line", x0=lo, x1=hi, y0=lo, y1=hi)
    f3.update_layout(title="Predicted vs actual", height=340, xaxis_title="Actual", yaxis_title="Predicted")
    c3.plotly_chart(f3, width="stretch")
    flagged = diag[diag["flagged"]]
    if len(flagged):
        st.markdown("**Influential or outlying runs**")
        st.dataframe(flagged, width="stretch")
    else:
        st.caption("No runs flagged (|studentized residual| ≤ 3 and Cook's D ≤ 4/n).")
