# License: Apache-2.0
import pytest
import os
import numpy as np
from lightgbm import LGBMClassifier
from gators.model_building.lgbm_treelite_dumper import LGBMTreeliteDumper


def test():
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 0])
    model = LGBMClassifier(max_depth=1, n_estimators=1).fit(X_train, y_train)
    LGBMTreeliteDumper.dump(
        model=model,
        toolchain='gcc',
        parallel_comp=1,
        model_path='.',
        model_name='dummy'
    )
    LGBMTreeliteDumper.dump(
        model=model,
        toolchain='clang',
        parallel_comp=1,
        model_path='.',
        model_name='dummy'
    )
    os.remove('dummy.zip')
    os.remove('dummy.so')
    os.remove('dummy.dylib')
    os.remove('dummy_lgbm_model.txt')


def test_input():
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 0])
    model = LGBMClassifier(max_depth=1, n_estimators=1).fit(X_train, y_train)
    with pytest.raises(TypeError):
        LGBMTreeliteDumper.dump(
            model=0,
            toolchain='gcc',
            parallel_comp=1,
            model_path='.',
            model_name='dummy'
        )
    with pytest.raises(TypeError):
        LGBMTreeliteDumper.dump(
            model=model,
            toolchain=0,
            parallel_comp=1,
            model_path='.',
            model_name='dummy'
        )
    with pytest.raises(TypeError):
        LGBMTreeliteDumper.dump(
            model=model,
            toolchain='gcc',
            parallel_comp='a',
            model_path='.',
            model_name='dummy'
        )
    with pytest.raises(TypeError):
        LGBMTreeliteDumper.dump(
            model=model,
            toolchain='gcc',
            parallel_comp=1,
            model_path=0,
            model_name='dummy'
        )
    with pytest.raises(TypeError):
        LGBMTreeliteDumper.dump(
            model=model,
            toolchain='gcc',
            parallel_comp=1,
            model_path='.',
            model_name=0
        )
    with pytest.raises(ValueError):
        LGBMTreeliteDumper.dump(
            model=model,
            toolchain='a',
            parallel_comp=1,
            model_path='.',
            model_name='dummy'
        )
