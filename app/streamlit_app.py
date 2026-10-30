import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from rul.data.load_data import COLUMNS, load_rul, load_test
from rul.models.evaluate import evaluate
from rul.models.predict import load_artifacts, predict_rul
from rul.utils.config import load_config, resolve

st.set_page_config(page_title="RUL Predictive Maintenance", layout="wide")
st.title("Predictive Maintenance: Remaining Useful Life (RUL)")

cfg = load_config()
raw_dir = resolve(cfg, "raw_dir")
model_dir = resolve(cfg, "model_dir")


@st.cache_resource
def get_artifacts():
    return load_artifacts(model_dir)


@st.cache_data
def get_predictions(raw_df, _model, _scaler, _meta):
    return predict_rul(raw_df, _model, _scaler, _meta)


def read_uploaded(file):
    df = pd.read_csv(file, sep=r"\s+", header=None).dropna(axis=1, how="all")
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"Expected {len(COLUMNS)} columns, found {df.shape[1]}.")
    df.columns = COLUMNS
    return df


def status_of(rul, critical, warning):
    return np.select(
        [rul < critical, rul < warning],
        ["Critical", "Warning"],
        default="Healthy",
    )


try:
    model, scaler, meta = get_artifacts()
except FileNotFoundError:
    st.error("No trained model found. Run: python -m rul.pipeline")
    st.stop()

dataset = meta["dataset"]
st.caption(f"Model: {meta['model']} | trained on {dataset} | rul_clip = {meta['rul_clip']}")

st.sidebar.header("Settings")
critical = st.sidebar.slider("Critical if RUL below (cycles)", 5, 60, 20)
warning = st.sidebar.slider("Warning if RUL below (cycles)", 10, 120, 50)
uploaded = st.sidebar.file_uploader("Or upload a C-MAPSS style .txt file", type="txt")
if critical >= warning:
    st.sidebar.error("Critical threshold must be lower than the warning threshold.")
    st.stop()

truth = None
try:
    if uploaded is not None:
        raw_df = read_uploaded(uploaded)
    else:
        raw_df = load_test(raw_dir, dataset)
        truth = load_rul(raw_dir, dataset)
    predictions = get_predictions(raw_df, model, scaler, meta)
except ValueError as error:
    st.error(f"Could not use this data: {error}")
    st.stop()

if set(predictions.index) != set(raw_df["unit"].unique()):
    st.error("Prediction index does not match the engine IDs in the data.")
    st.stop()

fleet = pd.DataFrame({
    "engine": predictions.index.to_numpy(),
    "last_cycle": raw_df.groupby("unit")["cycle"].max().loc[predictions.index].values,
    "predicted_RUL": predictions.round(1).values,
})
fleet["status"] = status_of(fleet["predicted_RUL"], critical, warning)
if truth is not None:
    fleet["true_RUL"] = truth.loc[predictions.index].values
    fleet["error"] = (fleet["predicted_RUL"] - fleet["true_RUL"]).round(1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Engines", len(fleet))
c2.metric("Critical", int((fleet["status"] == "Critical").sum()))
c3.metric("Warning", int((fleet["status"] == "Warning").sum()))
c4.metric("Healthy", int((fleet["status"] == "Healthy").sum()))

left, right = st.columns(2)

with left:
    st.subheader("Fleet (most urgent first)")
    st.dataframe(fleet.sort_values("predicted_RUL"), height=420)

with right:
    st.subheader("Engine detail")
    engine = st.selectbox("Engine", fleet["engine"].tolist())
    sensor = st.selectbox("Sensor trend", meta["sensor_cols"])
    one = raw_df[raw_df["unit"] == engine].sort_values("cycle")
    chart = (
        alt.Chart(one[["cycle", sensor]])
        .mark_line()
        .encode(
            x=alt.X("cycle:Q", title="cycle"),
            y=alt.Y(f"{sensor}:Q", scale=alt.Scale(zero=False), title=sensor),
        )
    )
    st.altair_chart(chart)
    row = fleet[fleet["engine"] == engine].iloc[0]
    st.write(f"Predicted RUL: **{row['predicted_RUL']}** cycles ({row['status']})")

st.subheader("Predicted RUL by engine")
st.bar_chart(fleet.set_index("engine")["predicted_RUL"])

if truth is not None:
    st.subheader("Accuracy on the official test set")
    scores = evaluate(truth.loc[predictions.index], predictions)
    st.dataframe(pd.DataFrame([scores]).round(3))