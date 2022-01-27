# License: Apache-2.0
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import get_ipython

from gators.transformers import Transformer


def get_runtime_in_milliseconds(ipynb_benchmark: str) -> float:
    """Return the runtime in seconds.

    Parameters
    ----------
    ipynb_benchmark : str
        Output of the jupyter
        timeit magic command.

    Returns
    -------
    float:
        Runtime in milliseconds.
    """
    if not isinstance(ipynb_benchmark, str):
        raise TypeError("`ipynb_benchmark` should be a str")
    dump = ipynb_benchmark.split(" ± ")[0].split(" ")
    if "s" not in ipynb_benchmark:
        raise ValueError("`ipynb_benchmark` format not supported")
    if "min" in dump[0]:
        return float(dump[0][:-3]) * 1e3

    val = float(dump[0])
    t_unit = dump[1]
    if "ms" in t_unit:
        return val
    if "ms" in t_unit:
        return val
    if "µs" in t_unit:
        return val * 1e-3
    if "ns" in t_unit:
        return val * 1e-6
    if "s" in t_unit:
        return val * 1e3


def generate_per_sample_benchmarking(
    objs: List[Transformer],
    Xs: List[pd.DataFrame],
    extra_info_X_vec: List[str] = None,
    extra_info_O_vec: List[str] = None,
    ys: List[np.ndarray] = None,
    timeit_args="",
) -> pd.DataFrame:
    """Calculate the per-sample benchmarking.

    Parameters
    ----------
    objs List[Transformer]:
        List of transformers.
    Xs List[pd.DataFrame]:
        List of dataFrames.
    ys List[pd.Series], default to None:
        List of target values.


    Returns
    -------
    pd.DataFrame:
        Benchmarking results.
    """
    if ys is None:
        ys = len(Xs) * [None]
    if not extra_info_O_vec:
        index = [
            f"{obj.__class__.__name__}{extra_info_X}"
            for obj, extra_info_X in zip(objs, extra_info_X_vec)
        ]
    elif not extra_info_X_vec:
        index = [
            f"{obj.__class__.__name__}{extra_info_O}"
            for obj, extra_info_O in zip(objs, extra_info_O_vec)
        ]
    else:
        index = [
            f"{obj.__class__.__name__}{extra_info_O}{extra_info_X}"
            for obj, extra_info_O, extra_info_X in zip(
                objs, extra_info_X_vec, extra_info_O_vec
            )
        ]
    columns = ["pandas", "numpy"]
    results = pd.DataFrame(np.nan, columns=columns, index=index)
    if not extra_info_O_vec:
        extra_info_O_vec = len(objs) * [""]
    if not extra_info_X_vec:
        extra_info_X_vec = len(Xs) * [""]
    for i, (obj, extra_info_O) in enumerate(zip(objs, extra_info_O_vec)):
        for X, y, extra_info_X in zip(Xs, ys, extra_info_X_vec):
            mask = pd.Series({c: True for c in X.columns if c.startswith('Dates')})
            X_row = X.iloc[[0]].copy()
            X_row_np = X_row.to_numpy().copy()
            
            if mask.sum():  # ensure that the input datatype is an object for NumPy and a datetime for Pandas
                X = X.astype(object)
                X_row = X.iloc[[0]].copy()
                X_row_np = X_row.to_numpy().copy()
                cols = mask[mask].index
                X[cols] = X[cols].astype('datetime64[ns]')
                X_row[cols] = X_row[cols].astype('datetime64[ns]')
            _ = obj.fit(X.copy(), y)
            dummy = get_ipython().run_line_magic(
                "timeit", f"-o {timeit_args} -q obj.transform(X_row.copy())"
            )
            results.loc[
                f"{obj.__class__.__name__}{extra_info_O}{extra_info_X}", "pandas"
            ] = get_runtime_in_milliseconds(str(dummy))
            dummy = get_ipython().run_line_magic(
                "timeit", f"-o -q {timeit_args} obj.transform_numpy(X_row_np.copy())"
            )
            results.loc[
                f"{obj.__class__.__name__}{extra_info_O}{extra_info_X}", "numpy"
            ] = get_runtime_in_milliseconds(str(dummy))
    return results


def benchmark_with_same_X(
    objs: List[Transformer],
    X: pd.DataFrame,
    info_vec: List[str],
    y: pd.Series = None,
    timeit_args="",
) -> pd.DataFrame:
    index = [f"{obj.__class__.__name__}{info}" for obj, info in zip(objs, info_vec)]
    columns = ["pandas", "numpy"]
    results = pd.DataFrame(np.nan, columns=columns, index=index)
    for obj, info in zip(objs, info_vec):
        idx = f"{obj.__class__.__name__}{info}"
        X_row = X.iloc[[0]].copy()
        X_row_np = X_row.to_numpy()
        _ = obj.fit(X.copy(), y)
        dummy = get_ipython().run_line_magic(
            "timeit", f"-o -q {timeit_args} obj.transform(X_row)"
        )
        results.loc[idx, "pandas"] = get_runtime_in_milliseconds(str(dummy))
        dummy = get_ipython().run_line_magic(
            "timeit", f"-o -q {timeit_args} obj.transform_numpy(X_row_np.copy())"
        )
        results.loc[idx, "numpy"] = get_runtime_in_milliseconds(str(dummy))
    return results


def benchmark(
    objs: object,
    Xs: List[pd.DataFrame],
    info_vec: List[str],
    y: pd.Series = None,
    timeit_args="",
) -> pd.DataFrame:

    index = [f"{obj.__class__.__name__}{info}" for obj, info in zip(objs, info_vec)]
    columns = ["pandas", "numpy"]
    bench = pd.DataFrame(np.nan, columns=columns, index=index)

    for obj, X, info in zip(objs, Xs, info_vec):
        idx = f"{obj.__class__.__name__}{info}"
        X_row = X.iloc[[0]].copy()
        X_row_np = X_row.to_numpy()
        _ = obj.fit(X.copy(), y)
        dummy = get_ipython().run_line_magic(
            "timeit", f"-o -q {timeit_args} obj.transform(X_row)"
        )
        bench.loc[idx, "pandas"] = get_runtime_in_milliseconds(str(dummy))
        dummy = get_ipython().run_line_magic(
            "timeit", f"-o -q {timeit_args} obj.transform_numpy(X_row_np.copy())"
        )
        bench.loc[idx, "numpy"] = get_runtime_in_milliseconds(str(dummy))
    return bench


def benchmark_with_same_obj(
    obj: object,
    Xs: List[pd.DataFrame],
    info_vec: List[str],
    y: pd.Series = None,
    timeit_args="",
) -> pd.DataFrame:

    index = [f"{obj.__class__.__name__}{info}" for info in info_vec]
    columns = ["pandas", "numpy"]
    results = pd.DataFrame(np.nan, columns=columns, index=index)
    for X, info in zip(Xs, info_vec):
        idx = f"{obj.__class__.__name__}{info}"
        X_row = X.iloc[[0]].copy()
        X_row_np = X_row.to_numpy()
        _ = obj.fit(X.copy(), y)
        dummy = get_ipython().run_line_magic(
            "timeit", f"-o -q {timeit_args} obj.transform(X_row)"
        )
        results.loc[idx, "pandas"] = get_runtime_in_milliseconds(str(dummy))
        dummy = get_ipython().run_line_magic(
            "timeit", f"-o -q {timeit_args} obj.transform_numpy(X_row_np.copy())"
        )
        results.loc[idx, "numpy"] = get_runtime_in_milliseconds(str(dummy))
    return results


def plot_comparison(bench_dict):
    for key, val in bench_dict.items():
        column = "column" if key == 1 else "columns"
        (val[columns] * 1e-3).plot.bar(
            logy=True,
            ylabel="runtime (s)",
            xlabel="trasformer",
            rot=90,
            color=["#c73d22", "#0077ea"],
            legend=True,
            figsize=[5 * 1.61, 5],
            title=f"per-sample transform vs transform_numpy\n datetime feature generation - {key} {column}",
            width=0.75,
            fontsize=10,
        )
        plt.show()


def plot_ratios(bench_dict):
    for key, val in bench_dict.items():
        ratio = val["pandas"] / val["numpy"]
        column = "column" if key == 1 else "columns"
        ax = ratio.plot.bar(
            rot=90,
            color=["#0077ea"],
            legend=False,
            figsize=[5 * 1.61, 5],
            title=f"per-sample transform vs transform_numpy\n datetime feature generation - {key} {column}",
            width=0.75,
            fontsize=10,
        )
        for p in ax.patches:
            ax.annotate(
                f"{round(p.get_height())}X",
                (p.get_x() * 1 + 0.25, p.get_height() * 1.02),
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_yaxis().set_ticks([])
        plt.show()


def plot_all(bench_dict):
    # key = list(bench_dict.keys())[-1]
    # val = bench_dict[key]
    # columns = (val['pandas']/val['numpy']).sort_values().index
    for key, val in bench_dict.items():
        fig, ax = plt.subplots(1, 2, figsize=[18, 8])
        column = "column" if key == 1 else "columns"
        val.index = val.index.str.split("_").str[0]
        (val * 1e-3).plot.bar(
            ax=ax[0],
            logy=True,
            ylabel="runtime (s)",
            xlabel="transformers",
            rot=90,
            color=["#c73d22", "#0077ea"],
            legend=True,
            figsize=[5 * 1.61, 5],
            width=0.75,
            fontsize=10,
            ylim=[1e-6, 1],
        )
        (val["pandas"] / val["numpy"]).plot.bar(
            ax=ax[1],
            rot=90,
            color=["#0077ea"],
            legend=False,
            figsize=[5 * 1.61, 5],
            ylabel="runtime speed-up",
            xlabel="transformers",
            width=0.75,
            fontsize=10,
        )
        for p in ax[1].patches:
            ax[1].annotate(
                f"{round(p.get_height())}X",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
            )

        ax[1].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[1].spines["bottom"].set_visible(False)
        ax[1].spines["left"].set_visible(False)
        ax[1].get_yaxis().set_ticks([])

        title = f"""per-sample transform vs transform_numpy - {key} {column}"""
        plt.suptitle(title, fontsize=12)
        plt.tight_layout()
        plt.show()


def run_X(objs, X, columns, n_vec=[1, 10, 100], y=None, timeit_args=""):
    bench_dict = {}
    if y is None:
        ys = None
    else:
        ys = [y]
    for n in n_vec:
        if n == 1:
            bench_dict[1] = generate_per_sample_benchmarking(
                objs, [X], extra_info_X_vec=["_1column"], ys=ys, timeit_args=timeit_args
            )

        else:
            XN = X.copy()
            for col in columns:
                for i in range(n - 1):
                    XN[f"{col}{i}"] = X[col].copy()
            bench_dict[n] = generate_per_sample_benchmarking(
                objs,
                [XN],
                extra_info_X_vec=[f"_{n}column"],
                ys=ys,
                timeit_args=timeit_args,
            )
    return bench_dict
