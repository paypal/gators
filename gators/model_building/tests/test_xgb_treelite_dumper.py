# License: Apache-2.0
import glob
import os
import platform

import numpy as np
import pytest
import xgboost

from gators.model_building.xgb_treelite_dumper import XGBTreeliteDumper


def test():
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 0])
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    model = xgboost.train({"max_depth": 1}, dtrain, num_boost_round=1)
    if platform.system() == "Linux":
        XGBTreeliteDumper.dump(
            model=model,
            toolchain="gcc",
            parallel_comp=1,
            model_path=".",
            model_name="dummy",
        )
    elif platform.system() == "Darwin":
        XGBTreeliteDumper.dump(
            model=model,
            toolchain="clang",
            parallel_comp=1,
            model_path=".",
            model_name="dummy",
        )
    elif platform.system() == "Windows":
        XGBTreeliteDumper.dump(
            model=model,
            toolchain="msvc",
            parallel_comp=1,
            model_path=".",
            model_name="dummy",
        )
    else:
        pass
    [os.remove(f) for f in glob.glob("*") if f.startswith("dummy")]


def test_input():
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 0])
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    model = xgboost.train({"max_depth": 1}, dtrain, num_boost_round=1)
    with pytest.raises(TypeError):
        XGBTreeliteDumper.dump(
            model=0,
            toolchain="gcc",
            parallel_comp=1,
            model_path=".",
            model_name="dummy",
        )
    with pytest.raises(TypeError):
        XGBTreeliteDumper.dump(
            model=model,
            toolchain=0,
            parallel_comp=1,
            model_path=".",
            model_name="dummy",
        )
    with pytest.raises(TypeError):
        XGBTreeliteDumper.dump(
            model=model,
            toolchain="gcc",
            parallel_comp="a",
            model_path=".",
            model_name="dummy",
        )
    with pytest.raises(TypeError):
        XGBTreeliteDumper.dump(
            model=model,
            toolchain="gcc",
            parallel_comp=1,
            model_path=0,
            model_name="dummy",
        )
    with pytest.raises(TypeError):
        XGBTreeliteDumper.dump(
            model=model, toolchain="gcc", parallel_comp=1, model_path=".", model_name=0
        )
    with pytest.raises(ValueError):
        XGBTreeliteDumper.dump(
            model=model,
            toolchain="a",
            parallel_comp=1,
            model_path=".",
            model_name="dummy",
        )
