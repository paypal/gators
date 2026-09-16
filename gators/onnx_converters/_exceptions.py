# SPDX-License-Identifier: Apache-2.0
class OnnxNotSupportedError(Exception):
    """Raised when a transformer or strategy cannot be expressed in a standard ONNX graph."""
