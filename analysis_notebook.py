"""Marimo companion to analysis.py.

Run with: marimo edit analysis_notebook.py
or:       marimo run analysis_notebook.py

This is ordinary marimo source: opening it does not require a live kernel in
this implementation workflow.  It reads artifacts only and never starts a
benchmark.  Altair facets and labels keep phases and workload kinds separate;
each mark is one observed trial, not a pooled average.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    notebook_dir = Path(__file__).resolve().parent
    if str(notebook_dir) not in sys.path:
        sys.path.insert(0, str(notebook_dir))

    from analysis import load_artifacts, successful_inference

    default_artifact_root = Path(
        os.environ.get("SENTIMENT_BENCHMARK_ARTIFACT_ROOT") or notebook_dir
    ).expanduser()
    return Path, default_artifact_root, load_artifacts, successful_inference


@app.cell
def _(mo, default_artifact_root):
    root_input = mo.ui.text(
        value=str(default_artifact_root),
        label="Artifact root (scans only its immediate */state.json children)",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md(
                """
                # Sentiment energy benchmark analysis

                Figures show **individual measured trials**. They do not pool
                phases, workload kinds, models, batch sizes, repetitions,
                datasets, hosts, or adapters. Hover for full serialized context;
                use the exported `trials.csv` for exact condition metadata.
                Missing measurements are omitted from marks, never replaced by
                zero. Only successful inference trials enter quality figures.
                """
            ),
            root_input,
        ]
    )
    return (root_input,)


@app.cell
def _(Path, load_artifacts, root_input):
    tables = None
    load_error = None
    try:
        tables = load_artifacts(Path(root_input.value).expanduser())
    except (OSError, ValueError) as exc:
        load_error = f"{type(exc).__name__}: {exc}"
    return load_error, tables


@app.cell
def _(mo, successful_inference, tables):
    runs = None if tables is None else tables.runs
    trials = None if tables is None else tables.trials
    inference = None if tables is None else successful_inference(tables.trials)
    eligible = (
        None
        if tables is None
        else tables.trials.loc[tables.trials["analysis_eligible"].fillna(False)].copy()
    )
    return eligible, inference, runs, trials


@app.cell
def _(load_error, mo, runs):
    if load_error is not None:
        status_output = mo.md(f"**Could not load artifacts:** `{load_error}`")
    elif runs is None or runs.empty:
        status_output = mo.md("No immediate `*/state.json` artifacts were found.")
    else:
        non_complete = runs.loc[~runs["analysis_eligible"].fillna(False)]
        status_output = mo.vstack(
            [
                mo.md("## Run status — all discovered states"),
                runs,
                mo.md("## Non-complete status table (including missing or broken results)"),
                non_complete if not non_complete.empty else mo.md("No non-complete or unusable runs."),
            ]
        )
    status_output
    return ()


@app.cell
def _(inference, mo):
    if inference is None:
        inference_output = mo.md("Inference table unavailable.")
    else:
        inference_output = mo.vstack(
            [
                mo.md(
                    """
                    ## Successful inference trials

                    One row per measured trial. `condition_id` incorporates
                    phase, kind, model, batch size, repetition, and serialized
                    trial/dataset/host/adapter metadata. Nulls are not zeros.
                    """
                ),
                inference,
            ]
        )
    inference_output
    return ()


@app.cell
def _():
    import altair as alt

    tooltip = [
        "trial_label:N",
        "condition_id:N",
        "phase:N",
        "kind:N",
        "model:N",
        "batch_size:N",
        "rep:N",
        "run_id:N",
        "dataset_json:N",
        "host_json:N",
        "adapter_json:N",
    ]

    def grouped_bars(data, series, value_column, value_title, title, kinds=False):
        chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X("trial_label:N", title="Observed trial", axis=alt.Axis(labelAngle=-35)),
                xOffset=f"{series}:N",
                y=alt.Y(f"{value_column}:Q", title=value_title),
                color=alt.Color(f"{series}:N", title=series.replace("_", " ")),
                tooltip=tooltip + [f"{series}:N", f"{value_column}:Q"],
            )
            .properties(width=520, height=300, title=title)
        )
        if kinds:
            return chart.facet(row="phase:N", column="kind:N")
        return chart.facet(column="phase:N")

    return alt, grouped_bars, tooltip


@app.cell
def _(alt, grouped_bars, inference, mo):
    if inference is None or inference.empty:
        quality_output = mo.md("No successful inference trials available for quality figures.")
    else:
        quality_long = inference.melt(
            id_vars=[
                "run_id",
                "condition_id",
                "trial_label",
                "phase",
                "kind",
                "model",
                "batch_size",
                "rep",
                "dataset_json",
                "host_json",
                "adapter_json",
            ],
            value_vars=[
                "quality_macro_recall",
                "quality_macro_f1",
                "quality_accuracy",
                "quality_coverage",
            ],
            var_name="quality_metric",
            value_name="value",
        ).dropna(subset=["value"])
        if quality_long.empty:
            quality_output = mo.md("Inference quality fields are present only as nulls.")
        else:
            quality_output = grouped_bars(
                quality_long,
                "quality_metric",
                "value",
                "Score",
                "Inference quality by trial and phase",
            )
    quality_output
    return ()


@app.cell
def _(alt, mo, tables, tooltip):
    if tables is None or tables.class_quality.empty:
        class_output = mo.md("No per-class quality records for successful inference trials.")
    else:
        class_data = tables.class_quality.dropna(subset=["value"])
        class_output = (
            alt.Chart(class_data)
            .mark_rect()
            .encode(
                x=alt.X("class_label:N", title="Gold class"),
                y=alt.Y("trial_label:N", title="Observed inference trial"),
                color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis")),
                facet=alt.Facet("metric:N", title="Per-class metric"),
                tooltip=tooltip + ["class_label:N", "metric:N", "value:Q"],
            )
            .properties(width=180, height=300, title="Per-class inference quality")
        )
    class_output
    return ()


@app.cell
def _(grouped_bars, mo, tables):
    if eligible is None or tables is None:
        energy_output = mo.md("Energy table unavailable.")
    else:
        energy_data = tables.energy_components.loc[
            tables.energy_components["analysis_eligible"].fillna(False)
        ].dropna(subset=["energy_kwh"])
        if energy_data.empty:
            energy_output = mo.md("No CPU, GPU, or RAM energy measurements are available.")
        else:
            energy_output = mo.vstack(
                [
                    mo.md(
                        """
                        ## CodeCarbon component energy

                        CPU, GPU, and RAM are shown separately by phase and kind.
                        An unavailable component is null and produces no bar;
                        it is **not zero**. These components are not whole-system
                        energy. The total `energy_kwh` remains in the trial table.
                        """
                    ),
                    grouped_bars(
                        energy_data,
                        "component",
                        "energy_kwh",
                        "Energy (kWh)",
                        "Energy components by observed trial",
                        kinds=True,
                    ),
                ]
            )
    energy_output
    return ()


@app.cell
def _(grouped_bars, mo, tables):
    if tables is None:
        intensity_output = mo.md("Energy-intensity table unavailable.")
    else:
        intensity_data = tables.energy_intensity.loc[
            tables.energy_intensity["analysis_eligible"].fillna(False)
            & tables.energy_intensity["kind"].eq("inference")
        ].dropna(subset=["j_per_tweet"])
        if intensity_data.empty:
            intensity_output = mo.md("No derived energy-intensity measurements are available.")
        else:
            intensity_output = mo.vstack(
                [
                    mo.md(
                        """
                        ## Energy per submitted tweet

                        Gross, device-idle-adjusted, and loaded-idle-adjusted
                        values remain separate inference measurement bases. No
                        adjustment is inferred when its derived field is missing.
                        Training's legacy per-row derived key is not relabelled
                        as tweet energy and is excluded from this figure.
                        """
                    ),
                    grouped_bars(
                        intensity_data,
                        "basis",
                        "j_per_tweet",
                        "Energy (J / submitted tweet)",
                        "Energy intensity by observed trial",
                        kinds=True,
                    ),
                ]
            )
    intensity_output
    return ()


@app.cell
def _(grouped_bars, mo, tables):
    if tables is None:
        latency_output = mo.md("Latency table unavailable.")
    else:
        latency_data = tables.performance.loc[
            tables.performance["analysis_eligible"].fillna(False)
            & tables.performance["metric"].isin(["latency_p50_s", "latency_p95_s"])
        ].dropna(subset=["value"])
        if latency_data.empty:
            latency_output = mo.md("No latency measurements are available.")
        else:
            latency_output = mo.vstack(
                [
                    mo.md("## End-to-end latency (including queueing)"),
                    grouped_bars(
                        latency_data,
                        "metric",
                        "value",
                        "Latency (s)",
                        "Median and p95 latency by observed trial",
                        kinds=True,
                    ),
                ]
            )
    latency_output
    return ()


@app.cell
def _(alt, mo, tables, tooltip):
    if tables is None:
        throughput_output = mo.md("Throughput table unavailable.")
    else:
        throughput_data = tables.performance.loc[
            tables.performance["analysis_eligible"].fillna(False)
            & tables.performance["kind"].eq("inference")
            & tables.performance["metric"].eq("throughput_tweets_s")
        ].dropna(subset=["value"])
        if throughput_data.empty:
            throughput_output = mo.md("No throughput measurements are available.")
        else:
            throughput_output = (
                alt.Chart(throughput_data)
                .mark_bar()
                .encode(
                    x=alt.X("trial_label:N", title="Observed trial", axis=alt.Axis(labelAngle=-35)),
                    y=alt.Y("value:Q", title="Submitted workload window throughput (tweets/s)"),
                    color=alt.Color("model:N"),
                    tooltip=tooltip + ["value:Q"],
                )
                .properties(width=650, height=320, title="Inference throughput by observed trial")
                .facet(row="phase:N", column="kind:N")
            )
    throughput_output
    return ()


@app.cell
def _(alt, inference, mo, tooltip):
    if inference is None:
        tradeoff_output = mo.md("Quality–energy trade-off unavailable.")
    else:
        tradeoff_data = inference.dropna(
            subset=["quality_macro_recall", "gross_j_per_tweet"]
        )
        if tradeoff_data.empty:
            tradeoff_output = mo.md("No trials have both macro recall and gross energy per tweet.")
        else:
            tradeoff_output = (
                alt.Chart(tradeoff_data)
                .mark_point(size=110, filled=True)
                .encode(
                    x=alt.X("gross_j_per_tweet:Q", title="Gross energy (J / submitted tweet)"),
                    y=alt.Y("quality_macro_recall:Q", title="Macro recall", scale=alt.Scale(zero=False)),
                    color=alt.Color("model:N"),
                    shape=alt.Shape("batch_size:N", title="Batch size"),
                    tooltip=tooltip + ["quality_macro_recall:Q", "gross_j_per_tweet:Q"],
                )
                .properties(width=600, height=360, title="Quality–energy trade-off (one point per inference trial)")
                .facet(column="phase:N")
                .interactive()
            )
    tradeoff_output
    return ()


if __name__ == "__main__":
    app.run()
