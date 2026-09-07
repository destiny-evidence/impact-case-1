import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from ic1.core.config import TASKS, TaskName
    import matplotlib.pyplot as plt

    df = pd.read_csv(TASKS[TaskName.TAXONOMY].frequencies_path)
    df.head()
    return df, mo, pd, plt


@app.cell
def _(df, pd):
    thresholds = range(0, int(df["selected"].max()) + 1)
    coverage = pd.Series(
        {n: int((df["selected"] >= n).sum()) for n in thresholds}
    )
    ax = coverage.plot(
        title="concepts with >= N selections",
        xlabel="N (min selections)",
        ylabel="number of concepts",
        marker="o"
    )
    ax.figure
    return


@app.cell
def _(mo):
    mo.md("""
    ## Structure: how many labels at each element
    """)
    return


@app.cell
def _(df, mo):
    n = mo.ui.slider(1, int(df["selected"].max()), value=1, label="min selections N")
    n
    return (n,)


@app.cell
def _(df, n, pd):
    total = df["depth"].value_counts().sort_index()
    used = (
        df[df["selected"] >= n.value]["depth"]
        .value_counts()
        .reindex(total.index)
        .fillna(0)
        .astype(int)
    )
    counts = pd.DataFrame({"concepts": total, f"≥ {n.value} selected": used})
    counts.plot.barh(title="concepts at each level of depth").figure
    return


@app.cell
def _(df, n, pd, plt):
    scheme_totals = df["scheme_name"].value_counts().sort_index()
    scheme_used = (
        df[df["selected"] >= n.value]["scheme_name"]
        .value_counts()
        .reindex(scheme_totals.index)
        .fillna(0)
        .astype(int)
    )
    _counts = pd.DataFrame({"concepts": scheme_totals, f"≥ {n.value} selected": scheme_used})
    _fig, _ax = plt.subplots(figsize=(5,8))
    _counts.plot.barh(title="concepts within each scheme", ax=_ax).figure
    return


@app.cell
def _(df, mo):
    scheme = mo.ui.dropdown(
        options=sorted(df["scheme_name"].unique()),
        value=sorted(df["scheme_name"].unique())[0],
        label="scheme",
    )
    scheme
    return (scheme,)


@app.cell
def _(df, n, scheme):
    sub = df[df["scheme_name"] == scheme.value]
    uri2label = dict(zip(df["concept_uri"], df["pref_label"]))

    sub = sub.assign(
        parent=sub["broader_uri"].map(uri2label).fillna("(top level)"),
        used=(sub["selected"] >= n.value).astype(int),
    )
    _counts = (
        sub.groupby("parent")
        .agg(concepts=("selected", "size"), used=("used", "sum"))
        .rename(columns={"used": f"≥ {n.value} selected"})
        .sort_index()
    )
    _counts.plot.bar(title=f"{scheme.value}: concepts by parent").figure
    return


if __name__ == "__main__":
    app.run()
