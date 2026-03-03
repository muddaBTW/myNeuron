"""
TensorFlow/Keras code generation from network configuration.
Generates clean, runnable Python code.
"""

from models import (
    NetworkConfig, LayerConfig, LayerType, ActivationType,
    OptimizerType, LossType, MetricType
)


def _activation_str(act: ActivationType) -> str:
    """Convert activation enum to code string."""
    if act is None or act == ActivationType.NONE:
        return "None"
    return f"'{act.value}'"


def _format_shape(shape) -> str:
    """Format a shape value for code generation."""
    if isinstance(shape, list):
        if len(shape) == 1:
            return str(shape[0])
        return str(tuple(shape))
    return str(shape)


def _gen_layer_code(layer: LayerConfig, idx: int) -> str:
    """Generate code for a single layer."""
    lt = layer.layer_type
    parts = []

    if lt == LayerType.INPUT:
        shape = tuple(layer.input_shape) if layer.input_shape else (784,)
        return f"    tf.keras.layers.Input(shape={shape})"

    elif lt == LayerType.DENSE:
        parts.append(f"units={layer.units or 64}")
        parts.append(f"activation={_activation_str(layer.activation)}")
        if not layer.use_bias:
            parts.append("use_bias=False")
        if layer.kernel_initializer and layer.kernel_initializer.value != "glorot_uniform":
            parts.append(f"kernel_initializer='{layer.kernel_initializer.value}'")
        return f"    tf.keras.layers.Dense({', '.join(parts)})"

    elif lt == LayerType.ACTIVATION:
        act = layer.activation.value if layer.activation and layer.activation != ActivationType.NONE else "relu"
        return f"    tf.keras.layers.Activation('{act}')"

    elif lt == LayerType.DROPOUT:
        return f"    tf.keras.layers.Dropout(rate={layer.rate or 0.5})"

    elif lt == LayerType.FLATTEN:
        return f"    tf.keras.layers.Flatten()"

    elif lt == LayerType.RESHAPE:
        shape = tuple(layer.target_shape) if layer.target_shape else (-1,)
        return f"    tf.keras.layers.Reshape(target_shape={shape})"

    elif lt in (LayerType.CONV1D, LayerType.CONV2D, LayerType.CONV3D):
        parts.append(f"filters={layer.filters or 32}")
        ks = layer.kernel_size or (3 if lt == LayerType.CONV1D else [3, 3] if lt == LayerType.CONV2D else [3, 3, 3])
        parts.append(f"kernel_size={_format_shape(ks)}")
        if layer.strides:
            parts.append(f"strides={_format_shape(layer.strides)}")
        parts.append(f"padding='{layer.padding.value}'")
        parts.append(f"activation={_activation_str(layer.activation)}")
        return f"    tf.keras.layers.{lt.value}({', '.join(parts)})"

    elif lt == LayerType.SEPARABLE_CONV2D:
        parts.append(f"filters={layer.filters or 32}")
        ks = layer.kernel_size or [3, 3]
        parts.append(f"kernel_size={_format_shape(ks)}")
        parts.append(f"padding='{layer.padding.value}'")
        parts.append(f"activation={_activation_str(layer.activation)}")
        return f"    tf.keras.layers.SeparableConv2D({', '.join(parts)})"

    elif lt == LayerType.DEPTHWISE_CONV2D:
        ks = layer.kernel_size or [3, 3]
        parts.append(f"kernel_size={_format_shape(ks)}")
        parts.append(f"padding='{layer.padding.value}'")
        parts.append(f"activation={_activation_str(layer.activation)}")
        return f"    tf.keras.layers.DepthwiseConv2D({', '.join(parts)})"

    elif lt == LayerType.CONV2D_TRANSPOSE:
        parts.append(f"filters={layer.filters or 32}")
        ks = layer.kernel_size or [3, 3]
        parts.append(f"kernel_size={_format_shape(ks)}")
        parts.append(f"padding='{layer.padding.value}'")
        parts.append(f"activation={_activation_str(layer.activation)}")
        return f"    tf.keras.layers.Conv2DTranspose({', '.join(parts)})"

    elif lt in (LayerType.MAX_POOL_1D, LayerType.AVG_POOL_1D):
        ps = layer.pool_size or 2
        lname = "MaxPooling1D" if lt == LayerType.MAX_POOL_1D else "AveragePooling1D"
        return f"    tf.keras.layers.{lname}(pool_size={ps})"

    elif lt in (LayerType.MAX_POOL_2D, LayerType.AVG_POOL_2D):
        ps = layer.pool_size or [2, 2]
        lname = "MaxPooling2D" if lt == LayerType.MAX_POOL_2D else "AveragePooling2D"
        return f"    tf.keras.layers.{lname}(pool_size={_format_shape(ps)})"

    elif lt in (LayerType.MAX_POOL_3D, LayerType.AVG_POOL_3D):
        ps = layer.pool_size or [2, 2, 2]
        lname = "MaxPooling3D" if lt == LayerType.MAX_POOL_3D else "AveragePooling3D"
        return f"    tf.keras.layers.{lname}(pool_size={_format_shape(ps)})"

    elif lt in (LayerType.GLOBAL_AVG_POOL_1D, LayerType.GLOBAL_AVG_POOL_2D,
                LayerType.GLOBAL_MAX_POOL_1D, LayerType.GLOBAL_MAX_POOL_2D):
        return f"    tf.keras.layers.{lt.value}()"

    elif lt == LayerType.SIMPLE_RNN:
        parts.append(f"units={layer.units or 64}")
        parts.append(f"activation={_activation_str(layer.activation or ActivationType.TANH)}")
        parts.append(f"return_sequences={layer.return_sequences}")
        if layer.recurrent_dropout and layer.recurrent_dropout > 0:
            parts.append(f"recurrent_dropout={layer.recurrent_dropout}")
        return f"    tf.keras.layers.SimpleRNN({', '.join(parts)})"

    elif lt == LayerType.LSTM:
        parts.append(f"units={layer.units or 64}")
        parts.append(f"activation={_activation_str(layer.activation or ActivationType.TANH)}")
        parts.append(f"return_sequences={layer.return_sequences}")
        if layer.recurrent_dropout and layer.recurrent_dropout > 0:
            parts.append(f"recurrent_dropout={layer.recurrent_dropout}")
        return f"    tf.keras.layers.LSTM({', '.join(parts)})"

    elif lt == LayerType.GRU:
        parts.append(f"units={layer.units or 64}")
        parts.append(f"activation={_activation_str(layer.activation or ActivationType.TANH)}")
        parts.append(f"return_sequences={layer.return_sequences}")
        if layer.recurrent_dropout and layer.recurrent_dropout > 0:
            parts.append(f"recurrent_dropout={layer.recurrent_dropout}")
        return f"    tf.keras.layers.GRU({', '.join(parts)})"

    elif lt == LayerType.BIDIRECTIONAL:
        wrapped = layer.wrapped_layer_type or "LSTM"
        inner_parts = [f"units={layer.units or 64}"]
        inner_parts.append(f"return_sequences={layer.return_sequences}")
        return f"    tf.keras.layers.Bidirectional(tf.keras.layers.{wrapped}({', '.join(inner_parts)}))"

    elif lt == LayerType.BATCH_NORM:
        return f"    tf.keras.layers.BatchNormalization()"

    elif lt == LayerType.LAYER_NORM:
        return f"    tf.keras.layers.LayerNormalization()"

    elif lt == LayerType.EMBEDDING:
        return f"    tf.keras.layers.Embedding(input_dim={layer.input_dim or 10000}, output_dim={layer.output_dim or 128})"

    return f"    # Unknown layer type: {lt}"


def _gen_optimizer_code(config) -> str:
    """Generate optimizer instantiation code."""
    opt = config.optimizer
    otype = opt.optimizer_type
    lr = opt.learning_rate

    if otype == OptimizerType.ADAM:
        return f"tf.keras.optimizers.Adam(learning_rate={lr}, beta_1={opt.beta_1}, beta_2={opt.beta_2})"
    elif otype == OptimizerType.SGD:
        return f"tf.keras.optimizers.SGD(learning_rate={lr}, momentum={opt.momentum})"
    elif otype == OptimizerType.RMSPROP:
        return f"tf.keras.optimizers.RMSprop(learning_rate={lr})"
    elif otype == OptimizerType.ADAGRAD:
        return f"tf.keras.optimizers.Adagrad(learning_rate={lr})"
    elif otype == OptimizerType.ADAMAX:
        return f"tf.keras.optimizers.Adamax(learning_rate={lr})"
    elif otype == OptimizerType.NADAM:
        return f"tf.keras.optimizers.Nadam(learning_rate={lr})"
    elif otype == OptimizerType.ADAMW:
        return f"tf.keras.optimizers.AdamW(learning_rate={lr}, weight_decay={opt.weight_decay or 0.01})"
    return f"'{otype.value}'"


def _gen_metrics_code(metrics) -> str:
    """Generate metrics list code."""
    metric_strs = []
    for m in metrics:
        if m in (MetricType.ACCURACY, MetricType.MAE, MetricType.MSE):
            metric_strs.append(f"'{m.value}'")
        else:
            metric_strs.append(f"tf.keras.metrics.{m.value}()")
    return f"[{', '.join(metric_strs)}]"


def generate_code(config: NetworkConfig) -> str:
    """Generate complete TensorFlow/Keras code from network configuration."""

    lines = []

    # ── Header ──
    lines.append('"""')
    lines.append(f"Neural Network: {config.model_name}")
    lines.append(f"Generated by myNeuron — Visual Neural Network Designer")
    lines.append('"""')
    lines.append("")

    # ── Imports ──
    lines.append("import tensorflow as tf")
    lines.append("import numpy as np")
    lines.append("")
    lines.append("")

    # ── Check for Input layer usage → Functional vs Sequential ──
    has_input_layer = any(l.layer_type == LayerType.INPUT for l in config.layers)
    has_bidirectional = any(l.layer_type == LayerType.BIDIRECTIONAL for l in config.layers)

    if has_input_layer:
        # Functional API
        lines.append("# ─── Model Architecture (Functional API) ─────────────────────")
        lines.append("")

        non_input_layers = []
        input_layer = None
        for l in config.layers:
            if l.layer_type == LayerType.INPUT:
                input_layer = l
            else:
                non_input_layers.append(l)

        shape = tuple(input_layer.input_shape) if input_layer and input_layer.input_shape else (784,)
        lines.append(f"inputs = tf.keras.Input(shape={shape})")
        lines.append("")

        if non_input_layers:
            first = non_input_layers[0]
            first_code = _gen_layer_code(first, 0).strip()
            lines.append(f"x = {first_code}(inputs)")

            for i, layer in enumerate(non_input_layers[1:], 1):
                code = _gen_layer_code(layer, i).strip()
                lines.append(f"x = {code}(x)")

            lines.append("")
            lines.append(f"model = tf.keras.Model(inputs=inputs, outputs=x, name='{config.model_name}')")
        else:
            lines.append(f"model = tf.keras.Model(inputs=inputs, outputs=inputs, name='{config.model_name}')")
    else:
        # Sequential API
        lines.append("# ─── Model Architecture (Sequential API) ─────────────────────")
        lines.append("")
        lines.append(f"model = tf.keras.Sequential([")

        for i, layer in enumerate(config.layers):
            code = _gen_layer_code(layer, i)
            comma = "," if i < len(config.layers) - 1 else ""
            lines.append(f"{code}{comma}")

        lines.append(f"], name='{config.model_name}')")

    lines.append("")
    lines.append("")

    # ── Compile ──
    lines.append("# ─── Compile ─────────────────────────────────────────────────")
    lines.append("")
    optimizer_code = _gen_optimizer_code(config.compile_config)
    metrics_code = _gen_metrics_code(config.compile_config.metrics)
    lines.append("model.compile(")
    lines.append(f"    optimizer={optimizer_code},")
    lines.append(f"    loss='{config.compile_config.loss.value}',")
    lines.append(f"    metrics={metrics_code}")
    lines.append(")")
    lines.append("")
    lines.append("")

    # ── Summary ──
    lines.append("# ─── Model Summary ───────────────────────────────────────────")
    lines.append("")
    lines.append("model.summary()")
    lines.append("")
    lines.append("")

    # ── Training Template ──
    tc = config.training_config
    lines.append("# ─── Training ────────────────────────────────────────────────")
    lines.append("# Replace X_train and y_train with your actual data")
    lines.append("")
    lines.append("# X_train = np.random.randn(1000, *model.input_shape[1:]).astype(np.float32)")
    lines.append("# y_train = np.random.randint(0, 10, size=(1000,))")
    lines.append("")
    lines.append("# history = model.fit(")
    lines.append(f"#     X_train, y_train,")
    lines.append(f"#     epochs={tc.epochs},")
    lines.append(f"#     batch_size={tc.batch_size},")
    lines.append(f"#     validation_split={tc.validation_split},")
    lines.append(f"#     shuffle={tc.shuffle},")
    lines.append("#     verbose=1")
    lines.append("# )")
    lines.append("")
    lines.append("")

    # ── Save ──
    lines.append("# ─── Save Model ──────────────────────────────────────────────")
    lines.append("")
    lines.append(f"# model.save('{config.model_name}.keras')")
    lines.append(f"# print(f'Model saved as {config.model_name}.keras')")
    lines.append("")

    return "\n".join(lines)
