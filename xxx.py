import ast
import copy
import time
from itertools import combinations
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from gators.data_cleaning import DropHighNaNRatio, Replace
from gators.encoders import WOEEncoder
from gators.feature_generation import OneHot
from gators.feature_generation_str import LowerCase
from gators.feature_selection import VarianceFilter
from gators.imputers import ObjectImputer
from gators.pipeline import Pipeline
from gators.util.iv import compute_iv
from IPython.core.display import display
from joblib import Parallel, delayed

ENTITIES = [
    "card",
    "ccnumberhash",
    "sender_receiver",
    "device",
    "email",
    "ip",
    "phone",
    "bill_country",
    "ship_country",
    "domain",
    "merch_id"
    #     'visitor_id',
]


OBJECT_COLUMNS = [f"custom_string_field_{i}" for i in range(1, 51)] + [
    "card_brand",
    "card_country",
    "card_type",
    "billing_address_country_code",
    "shipping_country_code",
    "sender_email",
    "sender_email_domain",
    "card_hash",
    "sender_ip",
    "sender_device_id",
    "device_country",
    "bt_merchant_customer_id",
    "order_id",
    "cardholder_name",
    "bt_merchant_account_id",
    "card_issuer_name",
    "device_os_type",
    "device_os_version",
    "device_manufacturer",
    "device_model",
    "billing_admin_area1",
    "v41_fl_un_match",
    "cb_date" "is_apm",
    "ip_is_tor",
    "brand_name",
    "is_masked_ip",
    "device_is_vm",
    "sender_phone",
    "is_recurring",
    "proxy_setting",
    "bt_email_length",
    "soft_descriptor",
    "device_is_rooted",
    "transaction_date",
    "device_is_emulator",
    "ship_bill_zip_match",
    "shipping_postal_code",
    "full_billing_address",
    "ip_bill_zip_distance",
    "ip_ship_zip_distance",
    "billing_address_line1",
    "billing_address_line2",
    "shipping_address_line1",
    "shipping_address_line2",
    "cart_item_product_name",
    "full_shipping_address",
    "payment_instrument_detail_type",
    "bank_identification_number",
    "billing_address_postal_code",
]


def save_train_test_on_disk(pickle_file: str):
    # "Kobo Software Ireland Limited_r52z9bzfzhgbwstj_20230301_20230430_raw_data.pkl"
    df = pd.read_pickle(pickle_file)
    df = df.set_index("eid")
    df = df.drop(
        ["original_sim_dc", "sim_dc", "payment_status", "transaction_date_utc"], axis=1
    )

    numerical_columns = [c for c in df if c not in OBJECT_COLUMNS]
    other_columns = [
        c for c in df if c not in numerical_columns and c not in OBJECT_COLUMNS
    ]
    assert not other_columns
    df[numerical_columns] = df[numerical_columns].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("cb_date", axis=1), df["cb_date"].notnull(), random_state=0
    )
    X_train.to_parquet("X_train.parquet")
    pd.DataFrame(y_train).to_parquet("y_train.parquet")
    X_test.to_parquet("X_test.parquet")
    pd.DataFrame(y_test).to_parquet("y_test.parquet")


def get_bool_columns(X: pd.DataFrame):
    return [c for c in X if c.startswith("is_") or "_is_" in c]


def get_columns_to_capitalize(X):
    mask = (((X == "True") | (X == "False")).sum()) != 0
    return list(mask[mask].index)


def get_int_columns(X: pd.DataFrame):
    mask = X.dtypes != object
    numerical_columns = list(mask[mask].index)
    mask = (
        (X[numerical_columns].fillna(0)) == (X[numerical_columns].round(0).fillna(0))
    ).mean() == 1
    int_columns = list(mask[mask].index)
    bool_columns = get_bool_columns(X)
    return [c for c in int_columns if c not in bool_columns]


def get_column_entity_dict(
    X: pd.DataFrame, keyword_entities: Dict[str, str]
) -> Tuple[Dict[str, str], List[str]]:
    column_entity_dict = {}
    columns_without_entity = []
    for c in list(X):
        if "per" in c:
            dummy = c.split("_per_")[1]
            for k, v in keyword_entities.items():
                if f"_{k}_" in dummy or dummy.startswith(k):
                    column_entity_dict[c] = v
                    break
        else:
            for k, v in keyword_entities.items():
                if f"_{k}_" in c or c.startswith(k):
                    column_entity_dict[c] = v
                    break
        if c not in column_entity_dict:
            columns_without_entity.append(c)
    return column_entity_dict, columns_without_entity


def display_info(X: pd.DataFrame, y: pd.Series):
    print(f"samples count: {len(y)}")
    print(f"columns count: {X.shape[1]}")
    print(f"CB rate: {10000 * y.mean():.1f}bp")
    print(f"CB count: {y.sum()}")


def display_entity_analysis(X: pd.DataFrame):
    X_dtypes = X.dtypes
    mask_num = X_dtypes != object
    describe = X[mask_num[mask_num].index].describe().fillna(0)
    mask = describe.loc["std"] != 0
    print(f"Num trivial variables: {len(mask_num)}")
    print(f"Num non-trivial variables: {len(mask[mask].index)}")
    num_vars_per_entity = pd.Series(0, index=ENTITIES)
    num_nonull_vars_per_entity = pd.Series(0, index=ENTITIES)
    ratio_notnull_vars_per_entity = pd.Series(0, index=ENTITIES)
    for entity in ENTITIES:
        cols_entity = [c for c in list(mask.index) if entity in c]
        cols_entity_not_null = [c for c in list(mask[mask].index) if entity in c]
        if len(cols_entity):
            num_vars_per_entity[entity] = len(cols_entity)
            num_nonull_vars_per_entity[entity] = len(cols_entity_not_null)
            ratio_notnull_vars_per_entity[entity] = (
                num_nonull_vars_per_entity[entity] / num_vars_per_entity[entity]
            )

    num_vars_per_entity = num_vars_per_entity.sort_values(ascending=False)
    num_nonull_vars_per_entity = num_nonull_vars_per_entity.sort_values(ascending=False)
    ratio_notnull_vars_per_entity = 100 * ratio_notnull_vars_per_entity.sort_values(
        ascending=False
    )
    print("Num variables per entity:")
    display(num_vars_per_entity)
    print("Num non-trivial variables per entity:")
    display(num_nonull_vars_per_entity)
    print("Ratio non-trivial variables per entity (%):")
    display(ratio_notnull_vars_per_entity.round(2))
    print(
        f"TOTAL num non-trivial entity variables = {num_nonull_vars_per_entity.sum()}"
    )
    print(
        f"""RATIO TOTAL num non-trivial entity variables = {
        (100 * num_nonull_vars_per_entity.sum() / num_vars_per_entity.sum()):.1f}%"""
    )


def get_best_ruleset(series: pd.Series, pct_change_threshold: float = 0):
    pct_change = 100 * series.pct_change().fillna(1)
    mask = pct_change > pct_change_threshold
    for ruleset, m in zip(series.index, mask.values):
        if m == False:  # noqa: E712
            break
        optimized_ruleset = ruleset.split(" ||| ")
    return optimized_ruleset


def compute_perfs(
    X: pd.DataFrame, y: pd.Series, results: pd.DataFrame, betas=[0.5, 1, 2]
):
    from iter_rules.eval import RuleEval
    from iter_rules.frame import RuleFrameGen

    perfs = []
    for scale, rule in results["rule"].items():
        y_pred = RuleFrameGen.compute(X, rule)
        dummy = RuleEval.compute(y, y_pred, betas=betas)
        dummy.name = scale
        perfs.append(dummy)

    perfs = pd.concat(perfs, axis=1).T
    perfs.index.name = results.index.name
    return perfs


def compute_rulesubsets_perf(
    X,
    y,
    ruleset,
    max_num_rules: int = 3,
    betas: List[float] = [0.5, 1, 2],
    tp_multiplier: float = 3,
    fp_multiplier: float = 1,
    normalize: bool = False,
    sample_weight: np.array = None,
    n_jobs: int = 1,
    verbose: bool = False,
):
    from iter_rules.eval import RuleEval
    from joblib import Parallel, delayed

    X_np = X[ruleset].to_numpy()

    def compute(y, combi):
        y_pred = pd.DataFrame(
            X_np[:, combi].sum(2) > 0,
            index=y.index,
            columns=[str(c) for c in combi],
        )
        return RuleEval().compute(
            y,
            y_pred,
            betas=betas,
            tp_multiplier=tp_multiplier,
            fp_multiplier=fp_multiplier,
            normalize=normalize,
            sample_weight=sample_weight,
        )

    results_list = []
    for num_rules in range(1, max_num_rules + 1):
        combi = list(combinations(range(len(ruleset)), num_rules))
        if verbose:
            description = f"\t{len(combi)} combinations of {num_rules} rule."
            if num_rules > 1:
                description = description[:-1] + "s."
            print(description)
        if not len(combi):
            break
        if len(combi) < 64:
            results = [compute(y, combi)]
        else:
            n = len(combi) // n_jobs
            combis = [combi[i * n : (i + 1) * n] for i in range(n_jobs - 1)]
            combis += [combi[(n_jobs - 1) * n :]]
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute)(y, combi) for combi in combis
            )
        results_list.extend(results)
    combined_results = pd.concat(results_list)
    indices = []

    for idx in combined_results.index:
        idx_rules = []
        for i in idx[1:-1].split(","):
            if i:
                idx_rules.append(int(i))
        indices.append(" ||| ".join([ruleset[j] for j in idx_rules]))
    combined_results.index = indices

    combined_results["Nbr of rules"] = combined_results.index.str.count(" \|\|\| ") + 1
    return combined_results


def compute_stats(X, y, regularization=0.01):
    iv, stats = compute_iv(X, y, regularization=regularization)
    stats = stats.drop(["distrib_0", "distrib_1"], axis=1)
    stats["N"] = stats["0"] + stats["1"]
    cols = ["0", "1", "N"]
    stats_ratio = (100 * stats[cols] / stats.groupby("variable").sum()[cols]).rename(
        columns={col: f"{col}(%)" for col in cols}
    )
    stats = pd.concat([stats, stats_ratio], axis=1)
    return iv, stats


def combine_windows_versions_in_device_os_type(X_train, X_test):
    col = "device_os_type"
    if col not in X_train:
        return X_train, X_test
    to_replace_dict = {}
    categories_dict = {}
    cats = X_train[col].unique()
    cats = cats[cats == cats]
    for keyword in ["windows"]:
        cols = [c for c in cats if c.startswith(keyword)]
        d = dict(zip(cols, len(cols) * ["|".join(cols)]))
        categories_dict = {**categories_dict, **d}
    to_replace_dict[col] = categories_dict

    replace = Replace(to_replace_dict=to_replace_dict)
    X_train = replace.fit_transform(X_train)
    X_test = replace.transform(X_test)
    return X_train, X_test


def combine_email_domain_in_sender_email_domain(X_train, X_test):
    col = "sender_email_domain"
    if col not in X_train:
        return X_train, X_test
    to_replace_dict = {}
    categories_dict = {}
    cats = X_train[col].unique()
    cats = cats[cats == cats]
    for keyword in ["hotmail.", "outlook.", "yahoo.", "live.", "icloud.", "gmail."]:
        cols = [c for c in cats if c.startswith(keyword)]
        d = dict(zip(cols, len(cols) * [f"STARTSWITH__{keyword}"]))
        categories_dict = {**categories_dict, **d}

    to_replace_dict[col] = categories_dict
    replace = Replace(to_replace_dict=to_replace_dict)
    X_train = replace.fit_transform(X_train)
    X_test = replace.transform(X_test)
    return X_train, X_test


def combine_device_model_in_device_model(X_train, X_test):
    col = "device_model"
    if col not in X_train:
        return X_train, X_test
    to_replace_dict = {}
    categories_dict = {}

    keywords = [
        "sm",
        "cph",
        "mi",
        "moto",
        "mt",
        "pixel",
        "nokia",
        "redmi",
        "stk",
        "rmx",
        "vivo",
        "huawei",
        "zte",
        "asus",
        "infinix",
        "itel",
        "lenovo",
        "lg",
        "motorola",
        "iphone",
    ]
    cats = X_train[col].unique()
    cats = cats[cats == cats]
    for keyword in keywords:
        cols = [c for c in cats if keyword in c]
        d = dict(zip(cols, len(cols) * [f"CONTAINS__{keyword}"]))
        categories_dict = {**categories_dict, **d}

    to_replace_dict[col] = categories_dict
    replace = Replace(to_replace_dict=to_replace_dict)
    X_train = replace.fit_transform(X_train)
    X_test = replace.transform(X_test)
    return X_train, X_test


def combine_device_os_version_in_device_os_version(X_train, X_test):
    col = "device_os_version"
    if col not in X_train:
        return X_train, X_test
    to_replace_dict = {}
    categories_dict = {}
    cats = X_train[col].unique()
    cats = cats[cats == cats]
    cats_test = X_test[col].unique()
    cats_test = cats_test[cats_test == cats_test]
    for keyword in ["x86_64", "NT", "aarch64"]:
        cols = [c for c in cats if c.startswith(keyword)]
        cols += [c for c in cats_test if c.startswith(keyword)]
        cols = list(set(cols))
        d = dict(zip(cols, len(cols) * [f"STARTSWITH__{keyword}"]))
        categories_dict = {**categories_dict, **d}

    for keyword in np.arange(4, 20).astype(str):
        cols = [c for c in cats if c.startswith(keyword)]
        cols += [c for c in cats_test if c.startswith(keyword)]
        cols = list(set(cols))
        d = dict(zip(cols, len(cols) * [f"STARTSWITH__{keyword}"]))
        categories_dict = {**categories_dict, **d}
    to_replace_dict[col] = categories_dict
    replace = Replace(to_replace_dict=to_replace_dict)
    X_train = replace.fit_transform(X_train)
    X_test = replace.transform(X_test)
    return X_train, X_test


def create_16point1_version_onehot_in_device_os_version(X_train, X_test):
    col = "device_os_version"
    if col not in X_train:
        return X_train, X_test
    onehot = OneHot(categories_dict={col: ["16.1"]})
    X_train = onehot.fit_transform(X_train)
    X_test = onehot.transform(X_test)
    onehot_column = onehot.column_names[0]
    X_train[onehot_column] = X_train[onehot_column].astype(str)
    X_test[onehot_column] = X_test[onehot_column].astype(str)
    return X_train, X_test


def clean_num_columns_after_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame):
    to_drop = [
        c for c in X_train if f"{c}__bin" not in X_train and X_train[c].dtype == float
    ]
    return X_train.drop(to_drop, axis=1), X_test.drop(to_drop, axis=1)


def get_columns_to_onehot_and_to_woe(X_train: pd.DataFrame, max_num_cats_woe: int):
    mask = X_train.dtypes == object
    object_columns = [c for c in list(mask[mask].index) if not c.endswith("bin")]
    n_unique = X_train[object_columns].nunique()
    woe_columns = [c for c in object_columns if n_unique[c] < max_num_cats_woe]
    to_onehot_columns = object_columns
    return to_onehot_columns, woe_columns


def compute_onehot_categories_dict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    to_onehot_columns: List[str],
    min_cat_ratio_onehot=0.01,
    min_woe_onehot=0,
):
    if not to_onehot_columns:
        return {}
    _, stats = compute_stats(X_train[to_onehot_columns], y_train, regularization=0.01)
    mask = (stats["N(%)"] >= 100 * min_cat_ratio_onehot) & (
        stats["woe"].abs() > min_woe_onehot
    )
    stats = stats[mask].sort_values("woe", ascending=False)
    stats = stats.drop_duplicates()
    idx = stats.index
    categories_dict = {var: [] for var, _ in idx}
    _ = {categories_dict[var].append(cat) for var, cat in idx}
    return categories_dict


def get_object_columns_to_drop_after_onehot(
    to_onehot_columns: List[str],
    woe_columns: List[str],
):
    return [c for c in to_onehot_columns if c not in woe_columns]


def compute_stats_train_test(X_train, y_train, X_test, y_test, regularization=1e-3):
    _, stats_train = compute_stats(X_train, y_train, regularization=regularization)
    _, stats_test = compute_stats(X_test, y_test, regularization=regularization)

    stats_columns = list(stats_train.columns)
    columns_dict = dict(zip(stats_columns, [f"{c}_train" for c in stats_columns]))
    stats_train = stats_train.rename(columns=columns_dict)
    columns_dict = dict(zip(stats_columns, [f"{c}_test" for c in stats_columns]))
    stats_test = stats_test.rename(columns=columns_dict)
    stats = pd.concat([stats_train, stats_test], axis=1)
    stats.loc[:, "woe_change(%)"] = (
        100 * (stats["woe_test"] - stats["woe_train"]) / stats["woe_train"]
    )
    stats["woe_sign_change"] = (
        0.5 * (np.sign(stats["woe_train"] * stats["woe_test"]) - 1)
    ).astype(bool)
    return stats


def add_bin_bounds(stats):
    stats["left"] = (
        stats.index.get_level_values(1).str.split(", ").str[0].str[1:].astype(float)
    )
    stats["right"] = (
        stats.index.get_level_values(1).str.split(", ").str[1].str[:-1].astype(float)
    )
    return stats


def drop_samples_with_rare_1s_cats(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    min_ratio: float = 0.03,
    max_ratio_1: float = 0.005,
    verbose=False,
):
    """Drop samples for which a categotegy has a small ratio of 1s.

    Parameters
    ----------
    X_train : pd.DataFrame
        Train data.
    y_train : pd.Series
        Train target values.
    X_test : pd.DataFrame
        Test data.
    y_test : pd.Series
        Test target values.
    min_ratio : float, optional
        Discard the categories present at least `min_ratio`, by default 0.03.
    max_ratio_1 : float, optional
        Drop the categories with a ratio of 1s smaller than `max_ratio_1`, by default 0.005
    verbose : bool, optional
        Verbosity, by default False

    Returns
    -------
    _type_
        _description_
    """
    mask = (X_train.dtypes == object) | (X_train.dtypes == bool)
    object_columns = list(mask[mask].index)
    n_samples_train = len(y_train)
    n_1s_train = y_train.sum()
    _, stats = compute_stats(X_train[object_columns], y_train)
    stats["N(%)"] /= 100
    stats["1(%)"] = stats["1"] / n_1s_train
    stats = stats.drop(["woe", "0(%)"], axis=1)
    mask = (stats["N(%)"] >= min_ratio) & (stats["1(%)"] <= max_ratio_1)
    idx = stats[mask].index
    mask_train = np.zeros(n_samples_train, bool)

    for col, cat in idx:
        mask_dummy_test = X_test[col] == cat
        if y_test[mask_dummy_test].mean() > max_ratio_1:
            continue
        if col.endswith("__bin") and not (
            cat.startswith("(-inf, ") | cat.endswith(", inf)")
        ):
            print("CONTINUE", cat, col)
            continue
        if verbose:
            print(f"Drop `{cat}` from: '{col}'")
        mask_train += X_train[col] == cat

    X_train = X_train[~mask_train]
    y_train = y_train[~mask_train]
    ratio_samples_droppd_train = 100 * (1 - len(y_train) / n_samples_train)
    ratio_1s_dropped_train = 100 * (1 - y_train.sum() / n_1s_train)
    if verbose:
        print("Summary:")
        print(f"{ratio_samples_droppd_train:.1f}% samples dropped in train set")
        print(f"  {ratio_1s_dropped_train:.1f}% of 1s dropped in train set")
    return X_train, y_train, ratio_samples_droppd_train, ratio_1s_dropped_train


def categorical_stability_analysis(stats, X_test, y_test, regularization=1e-2):
    _, stats_test = compute_stats(X_test, y_test, regularization=regularization)
    stability_woe = pd.concat(
        [
            stats.rename(columns={"woe": "woe_train"}),
            stats_test.rename(columns={"woe": "woe_test"}),
        ],
        axis=1,
    )[["woe_train", "woe_test"]].dropna()
    sign_change = np.sign(stability_woe["woe_train"] * stability_woe["woe_test"])
    to_drop = list(set(sign_change[sign_change < 0].index.get_level_values(0)))
    change = (
        100
        * (stability_woe["woe_train"] - stability_woe["woe_test"])
        / stability_woe["woe_train"]
    )[stability_woe["woe_train"] > 0]
    to_drop += list(set(change[change.abs() >= 100].index.get_level_values(0)))
    return list(set(to_drop))


def merge_bins(stats, bins_, regularization=0.01):
    index = [f"({bins_[0]}, {bins_[1]})"] + [
        f"[{l}, {r})" for l, r in zip(bins_[1:-1], bins_[2:])
    ]

    new = pd.DataFrame(columns=["0", "1"], index=index)
    new["left"] = bins_[:-1]
    new["right"] = bins_[1:]

    new["variable"] = stats.index.get_level_values(0)[0]
    new["value"] = new.index

    for i, l, r in zip(index, bins_[:-1], bins_[1:]):
        new.loc[i, ["0", "1"]] = stats[(stats["left"] >= l) & (stats["left"] < r)][
            ["0", "1"]
        ].sum()
    new.loc[:, ["distrib_0", "distrib_1"]] = (
        (new[["0", "1"]] + regularization)
        / (new[["0", "1"]].sum() + 2 * regularization)
    ).to_numpy()

    new.loc[:, "woe"] = np.log(new["distrib_1"] / new["distrib_0"])
    new = new.drop(["distrib_0", "distrib_1"], axis=1)
    new = new.set_index(["variable", "value"])

    return new


def compute_iv_from_stats(stats, regularization=0.01):
    stats[["distrib_0", "distrib_1"]] = (stats[["0", "1"]] + regularization) / (
        stats[["0", "1"]].groupby("variable").sum() + 2 * regularization
    )
    iv = (stats["distrib_1"] - stats["distrib_0"]) * stats["woe"]
    iv.name = "iv"
    return iv.groupby("variable").sum()


def generate_bins_with_constraint(
    stats, monotone_constraint, regularization=0.01, verbose=False
):
    j = 0
    while True:
        stats.loc[:, "split"] = stats.index.get_level_values(1).astype(str)
        stats.loc[:, "left"] = (
            stats["split"].str.split(", ").str[0].str[1:].astype(float)
        )
        stats.loc[:, "right"] = (
            stats["split"].str.split(", ").str[1].str[:-1].astype(float)
        )
        stats = stats.sort_values("left")

        if len(stats) == 1:
            bins_ = np.array([-np.inf, np.inf])
            stats = merge_bins(stats, bins_, regularization)
            break

        stats.loc[:, "sign"] = monotone_constraint
        if monotone_constraint == 1:
            stats["sign"].iloc[:-1] = np.sign(stats["woe"].diff()).values[1:]
        else:
            stats["sign"].iloc[1:] = np.sign(stats["woe"].diff()).values[1:]
        if verbose and j == 0:
            print("Start")
            display(stats)

        stats_ = stats[monotone_constraint * stats["sign"] > 0]
        bins_ = np.concatenate((stats_["left"].values, np.array([+np.inf])))
        bins_[0] = -np.inf
        stats = merge_bins(stats, bins_, regularization)
        stats.loc[:, "sign"] = monotone_constraint
        stats["sign"].iloc[:-1] = np.sign(stats["woe"].diff().iloc[:]).values[1:]

        if (stats["sign"] == monotone_constraint).mean() == 1:
            break
        j += 1
        if j == 20:
            break
        if verbose:
            print(f"Iteration {j}")
            display(stats)
    stats = stats.drop(["sign", "left", "right"], axis=1)
    stats["iv"] = compute_iv_from_stats(stats, regularization=regularization)[0]
    return stats.droplevel(0)


def supervised_feature_selection(estimators, X, y, k_per_model=1):
    to_drop = []
    selected_features = []
    mask = np.ones(len(y), bool)
    for estimator in estimators:
        features = [c for c in X.columns if c not in to_drop]
        if not features:
            break
        estimator.fit(X[mask][features], y[mask])
        feature_importance = pd.Series(
            estimator.feature_importances_, index=features
        ).sort_values(ascending=False)
        feature_importance = feature_importance[feature_importance != 0]
        selected_features_ = list(feature_importance.index[:k_per_model])
        mask[mask] = (y[mask] != 1) | (estimator.predict(X[mask][features]) != 1)
        if y[mask].sum() == 0:
            break

        to_drop += selected_features_
        selected_features += selected_features_

    return list(selected_features)


def generate_numerical_columns_description(
    X_train,
    y_train,
    pos_constraint_columns=None,
    neg_constraint_columns=None,
):
    unknown_constraint_columns = [c for c in X_train if c.endswith("__bin")]

    if pos_constraint_columns:
        _, positive_stats = compute_stats(X_train[pos_constraint_columns], y_train)
        positive_constraint_stats_matching = (
            positive_stats.copy()
            .groupby("variable")
            .apply(generate_bins_with_constraint, **{"monotone_constraint": +1})
        )
        positive_constraint_stats_matching = positive_constraint_stats_matching.drop(
            ["0", "1", "distrib_0", "distrib_1"], axis=1
        )
        positive_constraint_stats_matching = add_bin_bounds(
            positive_constraint_stats_matching
        )
        unknown_constraint_columns = [
            c for c in unknown_constraint_columns if c not in pos_constraint_columns
        ]

    else:
        pos_constraint_columns = []
        positive_constraint_stats_matching = pd.DataFrame([])

    if neg_constraint_columns:
        _, negative_stats = compute_stats(X_train[neg_constraint_columns], y_train)
        negative_constraint_stats_matching = (
            negative_stats.copy()
            .groupby("variable")
            .apply(generate_bins_with_constraint, **{"monotone_constraint": -1})
        )
        negative_constraint_stats_matching = negative_constraint_stats_matching.drop(
            ["0", "1", "distrib_0", "distrib_1"], axis=1
        )
        negative_constraint_stats_matching = add_bin_bounds(
            negative_constraint_stats_matching
        )
        unknown_constraint_columns = [
            c for c in unknown_constraint_columns if c not in neg_constraint_columns
        ]

    else:
        neg_constraint_columns = []
        negative_constraint_stats_matching = pd.DataFrame([])

    if unknown_constraint_columns:
        _, stats = compute_stats(X_train[unknown_constraint_columns], y_train)
        positive_constraint_stats = (
            stats.copy()
            .groupby("variable")
            .apply(generate_bins_with_constraint, **{"monotone_constraint": +1})
            .drop(["0", "1", "distrib_0", "distrib_1"], axis=1)
        )
        positive_constraint_stats = add_bin_bounds(positive_constraint_stats)
        positive_constraint_stats_full = pd.concat(
            [positive_constraint_stats_matching, positive_constraint_stats]
        )

        negative_constraint_stats = (
            stats.copy()
            .groupby("variable")
            .apply(generate_bins_with_constraint, **{"monotone_constraint": -1})
            .drop(["0", "1", "distrib_0", "distrib_1"], axis=1)
        )
        negative_constraint_stats = add_bin_bounds(negative_constraint_stats)
        negative_constraint_stats_full = pd.concat(
            [negative_constraint_stats_matching, negative_constraint_stats]
        )
    else:
        positive_constraint_stats_full = positive_constraint_stats_matching
        negative_constraint_stats_full = negative_constraint_stats_matching

    # WoE & IV
    numerical_columns_description = pd.concat(
        [
            positive_constraint_stats_full.groupby("variable").size(),
            negative_constraint_stats_full.groupby("variable").size(),
            positive_constraint_stats_full.groupby("variable")["iv"]
            .first()
            .rename(index="pos_iv"),
            negative_constraint_stats_full.groupby("variable")["iv"]
            .first()
            .rename(index="neg_iv"),
            positive_constraint_stats_full.groupby("variable")["woe"]
            .last()
            .rename(index="pos_woe"),
            negative_constraint_stats_full.groupby("variable")["woe"]
            .first()
            .rename(index="neg_woe"),
        ],
        axis=1,
    ).rename(columns={0: "pos_num_bins", 1: "neg_num_bins"})

    numerical_columns_description["max_woe"] = numerical_columns_description[
        ["pos_woe", "neg_woe"]
    ].max(1)
    numerical_columns_description["max_iv"] = numerical_columns_description[
        ["pos_iv", "neg_iv"]
    ].max(1)
    numerical_columns_description = numerical_columns_description.sort_values(
        "max_iv", ascending=False
    )

    # optimal bins

    def get_bin_list(x):
        return list(x.to_numpy()) if len(x) == 2 else list(x.to_numpy()) + [np.inf]

    numerical_columns_description = numerical_columns_description.fillna(0)
    mask_pos = (
        numerical_columns_description["pos_iv"]
        >= numerical_columns_description["neg_iv"]
    )
    numerical_columns_description["bins"] = np.nan

    pos_columns = list(mask_pos[mask_pos].index)
    numerical_columns_description["bins"].loc[pos_columns] = (
        positive_constraint_stats_full["left"]
        .loc[pos_columns]
        .groupby("variable")
        .aggregate(get_bin_list)
    )

    mask_neg = (
        numerical_columns_description["pos_iv"]
        < numerical_columns_description["neg_iv"]
    )
    neg_columns = list(mask_neg[mask_neg].index)
    numerical_columns_description["bins"].loc[neg_columns] = (
        negative_constraint_stats_full["left"]
        .loc[neg_columns]
        .groupby("variable")
        .aggregate(get_bin_list)
    )
    # constraints
    numerical_columns_description["constraint"] = 0

    mask_pos_constraint = (numerical_columns_description["pos_num_bins"] > 1) & (
        numerical_columns_description["neg_num_bins"] == 1
    )
    pos_constraint_columns_full = pos_constraint_columns + list(
        mask_pos_constraint[mask_pos_constraint].index
    )
    numerical_columns_description.loc[pos_constraint_columns_full, "constraint"] = 1

    mask_neg_constraint = (numerical_columns_description["pos_num_bins"] == 1) & (
        numerical_columns_description["neg_num_bins"] > 1
    )
    neg_constraint_columns_full = neg_constraint_columns + list(
        mask_neg_constraint[mask_neg_constraint].index
    )
    numerical_columns_description.loc[neg_constraint_columns_full, "constraint"] = -1

    numerical_columns_description[
        ["pos_num_bins", "neg_num_bins"]
    ] = numerical_columns_description[["pos_num_bins", "neg_num_bins"]].astype(int)
    return numerical_columns_description


"""def generate_numerical_columns_description(
    X_train,
    y_train,
    pos_constraint_columns=None,
    neg_constraint_columns=None,
):
    unknown_constraint_columns = [c for c in X_train if c.endswith("__bin")]

    if pos_constraint_columns:
        _, positive_stats = compute_stats(X_train[pos_constraint_columns], y_train)
        positive_constraint_stats_matching = (
            positive_stats.copy()
            .groupby("variable")
            .apply(generate_bins_with_constraint, **{"monotone_constraint": +1})
        )
        positive_constraint_stats_matching = positive_constraint_stats_matching.drop(
            ["0", "1", "distrib_0", "distrib_1"], axis=1
        )
        positive_constraint_stats_matching = add_bin_bounds(
            positive_constraint_stats_matching
        )
        unknown_constraint_columns = [
            c for c in unknown_constraint_columns if c not in pos_constraint_columns
        ]

    else:
        pos_constraint_columns = []
        positive_constraint_stats_matching = pd.DataFrame([])

    if neg_constraint_columns:
        _, negative_stats = compute_stats(X_train[neg_constraint_columns], y_train)
        negative_constraint_stats_matching = (
            negative_stats.copy()
            .groupby("variable")
            .apply(generate_bins_with_constraint, **{"monotone_constraint": -1})
        )
        negative_constraint_stats_matching = negative_constraint_stats_matching.drop(
            ["0", "1", "distrib_0", "distrib_1"], axis=1
        )
        negative_constraint_stats_matching = add_bin_bounds(
            negative_constraint_stats_matching
        )
        unknown_constraint_columns = [
            c for c in unknown_constraint_columns if c not in neg_constraint_columns
        ]

    else:
        neg_constraint_columns = []
        negative_constraint_stats_matching = pd.DataFrame([])

    if unknown_constraint_columns:
        _, stats = compute_stats(X_train[unknown_constraint_columns], y_train)
        positive_constraint_stats = (
            stats.copy()
            .groupby("variable")
            .apply(generate_bins_with_constraint, **{"monotone_constraint": +1})
            .drop(["0", "1", "distrib_0", "distrib_1"], axis=1)
        )
        positive_constraint_stats = add_bin_bounds(positive_constraint_stats)
        positive_constraint_stats_full = pd.concat(
            [positive_constraint_stats_matching, positive_constraint_stats]
        )

        negative_constraint_stats = (
            stats.copy()
            .groupby("variable")
            .apply(generate_bins_with_constraint, **{"monotone_constraint": -1})
            .drop(["0", "1", "distrib_0", "distrib_1"], axis=1)
        )
        negative_constraint_stats = add_bin_bounds(negative_constraint_stats)
        negative_constraint_stats_full = pd.concat(
            [negative_constraint_stats_matching, negative_constraint_stats]
        )
    else:
        positive_constraint_stats_full = positive_constraint_stats_matching
        negative_constraint_stats_full = negative_constraint_stats_matching

    # WoE & IV
    numerical_columns_description = pd.concat(
        [
            positive_constraint_stats_full.groupby("variable").size(),
            negative_constraint_stats_full.groupby("variable").size(),
            positive_constraint_stats_full.groupby("variable")["iv"]
            .first()
            .rename(index="pos_iv"),
            negative_constraint_stats_full.groupby("variable")["iv"]
            .first()
            .rename(index="neg_iv"),
            positive_constraint_stats_full.groupby("variable")["woe"]
            .last()
            .rename(index="pos_woe"),
            negative_constraint_stats_full.groupby("variable")["woe"]
            .first()
            .rename(index="neg_woe"),
        ],
        axis=1,
    ).rename(columns={0: "pos_num_bins", 1: "neg_num_bins"})

    numerical_columns_description["max_woe"] = numerical_columns_description[
        ["pos_woe", "neg_woe"]
    ].max(1)
    numerical_columns_description["max_iv"] = numerical_columns_description[
        ["pos_iv", "neg_iv"]
    ].max(1)
    numerical_columns_description = numerical_columns_description.sort_values(
        "max_iv", ascending=False
    )

    # optimal bins

    def get_bin_list(x):
        return list(x.to_numpy()) if len(x) == 2 else list(x.to_numpy()) + [np.inf]

    numerical_columns_description = numerical_columns_description.fillna(0)
    mask_pos = (
        numerical_columns_description["pos_iv"]
        >= numerical_columns_description["neg_iv"]
    )
    numerical_columns_description["bins"] = np.nan

    pos_columns = list(mask_pos[mask_pos].index)
    numerical_columns_description["bins"].loc[pos_columns] = (
        positive_constraint_stats_full["left"]
        .loc[pos_columns]
        .groupby("variable")
        .aggregate(get_bin_list)
    )

    mask_neg = (
        numerical_columns_description["pos_iv"]
        < numerical_columns_description["neg_iv"]
    )
    neg_columns = list(mask_neg[mask_neg].index)
    numerical_columns_description["bins"].loc[neg_columns] = (
        negative_constraint_stats_full["left"]
        .loc[neg_columns]
        .groupby("variable")
        .aggregate(get_bin_list)
    )
    # constraints
    numerical_columns_description["constraint"] = 0

    mask_pos_constraint = (numerical_columns_description["pos_num_bins"] > 1) & (
        numerical_columns_description["neg_num_bins"] == 1
    )
    pos_constraint_columns_full = pos_constraint_columns + list(
        mask_pos_constraint[mask_pos_constraint].index
    )
    numerical_columns_description.loc[pos_constraint_columns_full, "constraint"] = 1

    mask_neg_constraint = (numerical_columns_description["pos_num_bins"] == 1) & (
        numerical_columns_description["neg_num_bins"] > 1
    )
    neg_constraint_columns_full = neg_constraint_columns + list(
        mask_neg_constraint[mask_neg_constraint].index
    )
    numerical_columns_description.loc[neg_constraint_columns_full, "constraint"] = -1

    numerical_columns_description[
        ["pos_num_bins", "neg_num_bins"]
    ] = numerical_columns_description[["pos_num_bins", "neg_num_bins"]].astype(int)
    return numerical_columns_description
    
        """
