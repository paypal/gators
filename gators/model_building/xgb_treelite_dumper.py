# License: Apache-2.0
import os

import numpy as np
import treelite
import xgboost
from xgboost.sklearn import XGBClassifier


class XGBTreeliteDumper:
    """XGBoost Treelite Dumper class.

    Examples
    --------
    >>> import numpy as np
    >>> import xgboost as xgb
    >>> from gators.model_building import XGBTreeliteDumper
    >>> X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    >>> y_train = np.array([0, 1, 1, 0])
    >>> dtrain = xgb.DMatrix(X_train, label=y_train)
    >>> model = xgb.train({'max_depth': 1}, dtrain, num_boost_round=1)
    >>> XGBTreeliteDumper.dump(
    ...     model=model,
    ...     toolchain='gcc',
    ...     parallel_comp=1,
    ...     model_path='.',
    ...     model_name='dummy')
    [00:00:00] ../src/compiler/ast/split.cc:31: Parallel compilation enabled; member trees will be divided into 1 translation units.
    [00:00:01] ../src/compiler/ast_native.cc:45: Using ASTNativeCompiler
    [00:00:01] ../src/compiler/ast/split.cc:31: Parallel compilation enabled; member trees will be divided into 1 translation units.
    [00:00:01] ../src/c_api/c_api.cc:121: Code generation finished. Writing code to files...
    [00:00:01] ../src/c_api/c_api.cc:126: Writing file tu0.c...
    [00:00:01] ../src/c_api/c_api.cc:126: Writing file header.h...
    [00:00:01] ../src/c_api/c_api.cc:126: Writing file recipe.json...
    [00:00:01] ../src/c_api/c_api.cc:126: Writing file main.c...
    """

    @staticmethod
    def dump(
        model: xgboost.core.Booster,
        toolchain: str,
        parallel_comp: int,
        model_path: str,
        model_name: str,
    ):
        """Dump the XGBoost treelite as a .so and a
        .dylib file.

        Parameters
        ----------
        model: xgboost.core.Booster.
            Trained model.
        toolchain: str
            Compiler. List of available treelite compiler.
            * gcc
            * clang
            * msvc
        parallel_comp: int
            Treelite parallel compilation.
        model_path : str
            Model path.
        model_name : str
            Model name.
        """
        if not isinstance(model, xgboost.core.Booster):
            raise TypeError("`model` should be an xgboost.core.Booster.")
        if not isinstance(toolchain, str):
            raise TypeError("`toolchain` should be a str.")
        if toolchain not in ["gcc", "clang", "msvc"]:
            raise ValueError("`toolchain` should be `gcc`, `clang`, or `msvc`.")
        if not isinstance(parallel_comp, int):
            raise TypeError("`parallel_comp` should be an int.")
        if not isinstance(model_path, str):
            raise TypeError("`model_path` should be a str.")
        if not isinstance(model_name, str):
            raise TypeError("`model_name` should be a str.")
        model_ = treelite.Model.from_xgboost(model)
        platform_dict = {"gcc": "unix", "clang": "osx", "msvc": "windows"}
        format_dict = {"gcc": ".so", "clang": ".dylib", "msvc": ".dll"}
        model_.export_lib(
            toolchain=toolchain,
            libpath=os.path.join(model_path, model_name + format_dict[toolchain]),
            verbose=False,
            params={"parallel_comp": parallel_comp},
            nthread=1,
        )
        model_.export_srcpkg(
            platform=platform_dict[toolchain],
            toolchain=toolchain,
            params={"parallel_comp": parallel_comp},
            pkgpath=os.path.join(model_path, model_name + ".zip"),
            libname=os.path.join(model_path, model_name + format_dict[toolchain]),
            verbose=True,
        )
