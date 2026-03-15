"""
Model validation and summary computation.
Validates layer compatibility and computes output shapes and parameter counts.
"""

from typing import List, Tuple, Optional
from models import LayerConfig, LayerType, NetworkConfig


def _compute_conv_output(input_size, kernel_size, strides, padding):
    """Compute output size for a conv/pool operation."""
    if padding == "same":
        return (input_size + strides - 1) // strides
    return (input_size - kernel_size) // strides + 1


def compute_layer_info(layer: LayerConfig, input_shape: Optional[Tuple]) -> dict:
    """Compute output shape and parameter count for a single layer."""
    lt = layer.layer_type
    params = 0
    output_shape = input_shape

    if lt == LayerType.INPUT:
        shape = tuple(layer.input_shape) if layer.input_shape else (784,)
        output_shape = shape
        params = 0

    elif lt == LayerType.DENSE:
        units = layer.units or 64
        if input_shape:
            in_features = input_shape[-1]
            params = in_features * units
            if layer.use_bias:
                params += units
            output_shape = input_shape[:-1] + (units,)
        else:
            output_shape = (units,)

    elif lt == LayerType.ACTIVATION:
        params = 0

    elif lt == LayerType.DROPOUT:
        params = 0

    elif lt == LayerType.FLATTEN:
        if input_shape:
            flat_size = 1
            for d in input_shape:
                flat_size *= d
            output_shape = (flat_size,)
        params = 0

    elif lt == LayerType.RESHAPE:
        output_shape = tuple(layer.target_shape) if layer.target_shape else input_shape
        params = 0

    elif lt in (LayerType.CONV1D,):
        filters = layer.filters or 32
        ks = layer.kernel_size if isinstance(layer.kernel_size, int) else (layer.kernel_size[0] if layer.kernel_size else 3)
        if input_shape and len(input_shape) >= 2:
            in_channels = input_shape[-1]
            params = ks * in_channels * filters
            if layer.use_bias:
                params += filters
            stride = layer.strides if isinstance(layer.strides, int) else (layer.strides[0] if (layer.strides and len(layer.strides) > 0) else 1)
            padding_val = layer.padding.value if hasattr(layer.padding, 'value') else (layer.padding or "valid")
            seq_len = _compute_conv_output(input_shape[0], ks, stride, padding_val)
            output_shape = (seq_len, filters)
        else:
            output_shape = ("?", filters)

    elif lt in (LayerType.CONV2D, LayerType.SEPARABLE_CONV2D, LayerType.CONV2D_TRANSPOSE):
        filters = layer.filters or 32
        ks = layer.kernel_size or [3, 3]
        if isinstance(ks, int):
            ks = [ks, ks]
        if input_shape and len(input_shape) >= 3:
            in_channels = input_shape[-1]
            if lt == LayerType.SEPARABLE_CONV2D:
                params = ks[0] * ks[1] * in_channels + in_channels * filters
            elif lt == LayerType.DEPTHWISE_CONV2D:
                params = ks[0] * ks[1] * in_channels
                filters = in_channels
            else:
                params = ks[0] * ks[1] * in_channels * filters
            if layer.use_bias:
                params += filters
            strides = layer.strides or [1, 1]
            if isinstance(strides, int):
                strides = [strides, strides]
            
            padding_val = layer.padding.value if hasattr(layer.padding, 'value') else (layer.padding or "valid")
            h = _compute_conv_output(input_shape[0], ks[0], strides[0], padding_val)
            w = _compute_conv_output(input_shape[1], ks[1], strides[1], padding_val)
            output_shape = (h, w, filters)
        else:
            output_shape = ("?", "?", filters)

    elif lt == LayerType.CONV3D:
        filters = layer.filters or 32
        ks = layer.kernel_size or [3, 3, 3]
        if isinstance(ks, int):
            ks = [ks, ks, ks]
        if input_shape and len(input_shape) >= 4:
            in_channels = input_shape[-1]
            params = ks[0] * ks[1] * ks[2] * in_channels * filters
            if layer.use_bias:
                params += filters
            output_shape = ("?", "?", "?", filters)
        else:
            output_shape = ("?", "?", "?", filters)

    elif lt == LayerType.DEPTHWISE_CONV2D:
        ks = layer.kernel_size or [3, 3]
        if isinstance(ks, int):
            ks = [ks, ks]
        if input_shape and len(input_shape) >= 3:
            in_channels = input_shape[-1]
            params = ks[0] * ks[1] * in_channels
            if layer.use_bias:
                params += in_channels
            strides = layer.strides or [1, 1]
            if isinstance(strides, int):
                strides = [strides, strides]
            
            padding_val = layer.padding.value if hasattr(layer.padding, 'value') else (layer.padding or "valid")
            h = _compute_conv_output(input_shape[0], ks[0], strides[0], padding_val)
            w = _compute_conv_output(input_shape[1], ks[1], strides[1], padding_val)
            output_shape = (h, w, in_channels)
        else:
            output_shape = ("?", "?", "?")

    elif lt in (LayerType.MAX_POOL_1D, LayerType.AVG_POOL_1D):
        ps = layer.pool_size or 2
        if isinstance(ps, list):
            ps = ps[0]
        if input_shape and len(input_shape) >= 2:
            seq_len = input_shape[0] // ps
            output_shape = (seq_len,) + input_shape[1:]
        params = 0

    elif lt in (LayerType.MAX_POOL_2D, LayerType.AVG_POOL_2D):
        ps = layer.pool_size or [2, 2]
        if isinstance(ps, int):
            ps = [ps, ps]
        if input_shape and len(input_shape) >= 3:
            h = input_shape[0] // ps[0]
            w = input_shape[1] // ps[1]
            output_shape = (h, w, input_shape[-1])
        params = 0

    elif lt in (LayerType.MAX_POOL_3D, LayerType.AVG_POOL_3D):
        params = 0
        if input_shape:
            output_shape = tuple("?" for _ in input_shape)

    elif lt in (LayerType.GLOBAL_AVG_POOL_1D, LayerType.GLOBAL_MAX_POOL_1D):
        if input_shape and len(input_shape) >= 2:
            output_shape = (input_shape[-1],)
        params = 0

    elif lt in (LayerType.GLOBAL_AVG_POOL_2D, LayerType.GLOBAL_MAX_POOL_2D):
        if input_shape and len(input_shape) >= 3:
            output_shape = (input_shape[-1],)
        params = 0

    elif lt in (LayerType.SIMPLE_RNN, LayerType.LSTM, LayerType.GRU):
        units = layer.units or 64
        if input_shape and len(input_shape) >= 2:
            in_features = input_shape[-1]
            if lt == LayerType.SIMPLE_RNN:
                params = (in_features + units + 1) * units
            elif lt == LayerType.LSTM:
                params = 4 * ((in_features + units + 1) * units)
            elif lt == LayerType.GRU:
                params = 3 * ((in_features + units + 1) * units)
            if layer.return_sequences:
                output_shape = (input_shape[0], units)
            else:
                output_shape = (units,)
        else:
            output_shape = (units,)

    elif lt == LayerType.BIDIRECTIONAL:
        units = layer.units or 64
        wrapped = layer.wrapped_layer_type or "LSTM"
        if input_shape and len(input_shape) >= 2:
            in_features = input_shape[-1]
            if wrapped == "LSTM":
                single_params = 4 * ((in_features + units + 1) * units)
            elif wrapped == "GRU":
                single_params = 3 * ((in_features + units + 1) * units)
            else:
                single_params = (in_features + units + 1) * units
            params = single_params * 2  # forward + backward
            if layer.return_sequences:
                output_shape = (input_shape[0], units * 2)
            else:
                output_shape = (units * 2,)
        else:
            output_shape = (units * 2,)

    elif lt == LayerType.BATCH_NORM:
        if input_shape:
            params = 4 * input_shape[-1]  # gamma, beta, moving_mean, moving_var

    elif lt == LayerType.LAYER_NORM:
        if input_shape:
            params = 2 * input_shape[-1]  # gamma, beta

    elif lt == LayerType.EMBEDDING:
        input_dim = layer.input_dim or 10000
        output_dim = layer.output_dim or 128
        params = input_dim * output_dim
        if input_shape:
            output_shape = input_shape + (output_dim,)
        else:
            output_shape = ("?", output_dim)

    return {
        "output_shape": output_shape,
        "params": params
    }


def validate_network(config: NetworkConfig) -> dict:
    """Validate the network architecture for common issues."""
    errors = []
    warnings = []
    layers = config.layers

    if not layers:
        errors.append("Network has no layers.")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Check if first layer defines input shape
    first = layers[0]
    if first.layer_type != LayerType.INPUT:
        if first.layer_type == LayerType.DENSE and not first.input_shape:
            warnings.append(
                "First layer is Dense without an Input layer. "
                "Consider adding an Input layer or specifying input_shape on the first Dense layer."
            )

    # Track dimensionality through the network
    current_shape = None
    for i, layer in enumerate(layers):
        lt = layer.layer_type
        info = compute_layer_info(layer, current_shape)
        prev_shape = current_shape
        current_shape = info["output_shape"]

        # Conv after Dense without reshape
        if lt in (LayerType.CONV1D, LayerType.CONV2D, LayerType.CONV3D,
                  LayerType.SEPARABLE_CONV2D, LayerType.DEPTHWISE_CONV2D):
            if prev_shape and len(prev_shape) == 1:
                errors.append(
                    f"Layer {i} ({lt.value}): convolutional layer requires multi-dimensional input, "
                    f"but previous layer outputs 1D shape {prev_shape}. Add a Reshape layer."
                )

        # Dense after multi-dim without flatten
        if lt == LayerType.DENSE:
            if prev_shape and len(prev_shape) > 1 and i > 0:
                # Check if there's a conv-like layer before without flatten
                prev_lt = layers[i-1].layer_type
                if prev_lt not in (LayerType.FLATTEN, LayerType.GLOBAL_AVG_POOL_1D,
                                   LayerType.GLOBAL_AVG_POOL_2D, LayerType.GLOBAL_MAX_POOL_1D,
                                   LayerType.GLOBAL_MAX_POOL_2D, LayerType.DENSE,
                                   LayerType.SIMPLE_RNN, LayerType.LSTM, LayerType.GRU,
                                   LayerType.BIDIRECTIONAL):
                    warnings.append(
                        f"Layer {i} (Dense): previous layer outputs shape {prev_shape}. "
                        f"You might need a Flatten or GlobalPooling layer before Dense."
                    )

        # RNN needs 3D input
        if lt in (LayerType.SIMPLE_RNN, LayerType.LSTM, LayerType.GRU, LayerType.BIDIRECTIONAL):
            if prev_shape and len(prev_shape) < 2:
                errors.append(
                    f"Layer {i} ({lt.value}): recurrent layers require 2D+ input (sequence_length, features), "
                    f"but previous layer outputs shape {prev_shape}."
                )

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def get_model_summary(config: NetworkConfig) -> dict:
    """Compute per-layer summary with shapes and parameter counts."""
    layers_info = []
    current_shape = None
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for i, layer in enumerate(config.layers):
        info = compute_layer_info(layer, current_shape)
        current_shape = info["output_shape"]
        p = info["params"]

        layer_name = layer.name or f"{layer.layer_type.value}_{i}"

        # BatchNorm has non-trainable params (moving_mean, moving_var)
        if layer.layer_type == LayerType.BATCH_NORM:
            trainable_p = p // 2
            non_trainable_p = p // 2
        else:
            trainable_p = p
            non_trainable_p = 0

        total_params += p
        trainable_params += trainable_p
        non_trainable_params += non_trainable_p

        shape_str = str(current_shape) if current_shape else "?"

        layers_info.append({
            "name": layer_name,
            "layer_type": layer.layer_type.value,
            "output_shape": f"(None, {', '.join(str(s) for s in current_shape)})" if current_shape else "?",
            "param_count": p
        })

    # Also validate
    validation = validate_network(config)

    return {
        "layers": layers_info,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "warnings": validation.get("warnings", [])
    }
