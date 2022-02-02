# License: Apache-2.0
import os
from typing import Union

import treelite
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor


class LGBMTreeliteDumper:
    """LightGBM Treelite Dumper class.

    Examples
    --------
    >>> import numpy as np
    >>> from lightgbm import LGBMClassifier
    >>> from gators.model_building import LGBMTreeliteDumper
    >>> X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    >>> y_train = np.array([0, 1, 1, 0])
    >>> model = LGBMClassifier(max_depth=1, n_estimators=1).fit(X_train, y_train)
    >>> LGBMTreeliteDumper.dump(
    ... model=model,
    ... toolchain='gcc',
    ... parallel_comp=1,
    ... model_path='.',
    ... model_name='dummy')
    [00:00:00] /Users/travis/build/dmlc/treelite/src/compiler/ast/split.cc:29: Parallel compilation enabled; member trees will be divided into 1 translation units.
    [00:00:01] /Users/travis/build/dmlc/treelite/src/compiler/ast/split.cc:29: Parallel compilation enabled; member trees will be divided into 1 translation units.
    """

    @staticmethod
    def dump(
        model: Union[LGBMClassifier, LGBMRegressor],
        toolchain: str,
        parallel_comp: int,
        model_path: str,
        model_name: str,
        verbose: bool = False,
    ):
        """Dump the XGBoost treelite as a .so and a
        .dylib file.

        Parameters
        ----------
        model: Union[LGBMClassifier, LGBMRegressor].
            LightGBM trained model.
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
        verbose: bool, default False.
            Verbosity.
        """
        if not isinstance(model, (LGBMClassifier, LGBMRegressor)):
            raise TypeError("`model` should be a LGBMClassifier or LGBMRegressor.")
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
        file_name = "dummy_lgbm_model.txt"
        model.booster_.save_model(file_name)
        model_ = treelite.Model.load(file_name, model_format="lightgbm")
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
            verbose=verbose,
        )
