ONNX Export
===========

Export a fitted ``Pipeline`` (or single transformer) to ONNX and run inference with ONNX Runtime.

Every converter follows the same tensor naming convention: inputs are named ``{col}__in``
and outputs use the resulting Gators column name. Dtypes are derived from each transformer's
``_input_dtypes``/``_output_dtypes`` (Polars ``String`` -> ONNX ``STRING``, ``Boolean`` -> ``BOOL``,
integer/temporal dtypes -> ``INT64``, floats -> ``FLOAT``/``DOUBLE`` per ``float_datatype``).

Graph Export
------------

.. autofunction:: gators.onnx_converters.to_onnx_graph

.. autofunction:: gators.onnx_converters.pipeline_to_onnx

.. autofunction:: gators.onnx_converters.pipeline_to_scoring_onnx

Compatibility Check
--------------------

.. autofunction:: gators.onnx_converters.check_pipeline_onnx_compatibility

Inference
---------

.. autofunction:: gators.onnx_converters.create_session

.. autofunction:: gators.onnx_converters.run_session

Exceptions
----------

.. autoclass:: gators.onnx_converters.OnnxNotSupportedError
   :show-inheritance:
