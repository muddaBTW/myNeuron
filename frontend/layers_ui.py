"""
Layer configuration UI components for Streamlit.
Provides dynamic forms for configuring each layer type.
"""

import streamlit as st


# ─── Layer Parameter Definitions ─────────────────────────────────────────────

LAYER_CATEGORIES = {
    "🧠 Core": ["Input", "Dense", "Activation", "Dropout", "Flatten", "Reshape"],
    "📐 Convolutional": ["Conv1D", "Conv2D", "Conv3D", "SeparableConv2D", "DepthwiseConv2D", "Conv2DTranspose"],
    "🔽 Pooling": ["MaxPooling1D", "MaxPooling2D", "MaxPooling3D",
                    "AveragePooling1D", "AveragePooling2D", "AveragePooling3D",
                    "GlobalAveragePooling1D", "GlobalAveragePooling2D",
                    "GlobalMaxPooling1D", "GlobalMaxPooling2D"],
    "🔁 Recurrent": ["SimpleRNN", "LSTM", "GRU", "Bidirectional"],
    "📊 Normalization": ["BatchNormalization", "LayerNormalization"],
    "📝 Embedding": ["Embedding"],
}

ACTIVATIONS = ["None", "relu", "sigmoid", "tanh", "softmax", "leaky_relu",
               "elu", "selu", "swish", "gelu", "softplus", "linear"]

PADDINGS = ["valid", "same"]

INITIALIZERS = ["glorot_uniform", "glorot_normal", "he_uniform", "he_normal",
                "lecun_uniform", "lecun_normal", "zeros", "ones",
                "random_normal", "random_uniform"]


def render_layer_config(layer_type: str, idx: int, current_config: dict = None) -> dict:
    """Render configuration widgets for a specific layer type and return config dict."""
    config = {"layer_type": layer_type}
    if current_config:
        config.update(current_config)

    if layer_type == "Input":
        default_shape = ",".join(map(str, current_config.get("input_shape", [784]))) if current_config and "input_shape" in current_config else "784"
        shape_str = st.text_input(
            "Input Shape", value=default_shape,
            key=f"input_shape_{idx}",
            help="Comma-separated dimensions, e.g. '28,28,1' for images or '784' for flat"
        )
        try:
            config["input_shape"] = [int(x.strip()) for x in shape_str.split(",")]
        except ValueError:
            config["input_shape"] = [784]

    elif layer_type == "Dense":
        col1, col2 = st.columns(2)
        with col1:
            default_units = current_config.get("units", 64) if current_config else 64
            config["units"] = st.number_input("Units", min_value=1, value=default_units,
                                               key=f"units_{idx}")
        with col2:
            default_act = current_config.get("activation", "relu") if current_config else "relu"
            if isinstance(default_act, str) and default_act in ACTIVATIONS:
                act_idx = ACTIVATIONS.index(default_act)
            else:
                act_idx = ACTIVATIONS.index("relu")
            config["activation"] = st.selectbox("Activation", ACTIVATIONS, index=act_idx,
                                                 key=f"act_{idx}")
        default_bias = current_config.get("use_bias", True) if current_config else True
        config["use_bias"] = st.checkbox("Use Bias", value=default_bias, key=f"bias_{idx}")
        default_init = current_config.get("kernel_initializer", "glorot_uniform") if current_config else "glorot_uniform"
        if default_init in INITIALIZERS:
            init_idx = INITIALIZERS.index(default_init)
        else:
            init_idx = 0
        config["kernel_initializer"] = st.selectbox("Initializer", INITIALIZERS, index=init_idx,
                                                     key=f"init_{idx}")

    elif layer_type == "Activation":
        default_act = current_config.get("activation", "relu") if current_config else "relu"
        act_idx = (ACTIVATIONS[1:].index(default_act)) if (isinstance(default_act, str) and default_act in ACTIVATIONS[1:]) else 0
        config["activation"] = st.selectbox("Activation", ACTIVATIONS[1:],
                                             index=act_idx,
                                             key=f"act_{idx}")

    elif layer_type == "Dropout":
        default_rate = current_config.get("rate", 0.5) if current_config else 0.5
        config["rate"] = st.slider("Dropout Rate", 0.0, 1.0, default_rate, 0.05,
                                    key=f"rate_{idx}")

    elif layer_type == "Flatten":
        st.caption("Flattens input — no parameters to configure.")

    elif layer_type == "Reshape":
        default_ts = ",".join(map(str, current_config.get("target_shape", [28,28,1]))) if current_config and "target_shape" in current_config else "28,28,1"
        shape_str = st.text_input("Target Shape", value=default_ts,
                                   key=f"target_shape_{idx}")
        try:
            config["target_shape"] = [int(x.strip()) for x in shape_str.split(",")]
        except ValueError:
            config["target_shape"] = [28, 28, 1]

    elif layer_type in ("Conv1D", "Conv2D", "Conv3D", "SeparableConv2D",
                         "Conv2DTranspose"):
        col1, col2 = st.columns(2)
        with col1:
            default_filters = current_config.get("filters", 32) if current_config else 32
            config["filters"] = st.number_input("Filters", min_value=1, value=default_filters,
                                                 key=f"filters_{idx}")
        with col2:
            if layer_type == "Conv1D":
                default_ks = current_config.get("kernel_size", 3) if current_config else 3
                config["kernel_size"] = st.number_input("Kernel Size", min_value=1,
                                                         value=default_ks, key=f"ks_{idx}")
            elif layer_type == "Conv3D":
                default_ks = ",".join(map(str, current_config.get("kernel_size", [3,3,3]))) if current_config and "kernel_size" in current_config else "3,3,3"
                ks_str = st.text_input("Kernel Size", default_ks, key=f"ks_{idx}")
                try:
                    config["kernel_size"] = [int(x.strip()) for x in ks_str.split(",")]
                except ValueError:
                    config["kernel_size"] = [3, 3, 3]
            else:
                default_ks = ",".join(map(str, current_config.get("kernel_size", [3,3]))) if current_config and "kernel_size" in current_config else "3,3"
                ks_str = st.text_input("Kernel Size", default_ks, key=f"ks_{idx}")
                try:
                    config["kernel_size"] = [int(x.strip()) for x in ks_str.split(",")]
                except ValueError:
                    config["kernel_size"] = [3, 3]

        col3, col4 = st.columns(2)
        with col3:
            default_pad = current_config.get("padding", "same") if current_config else "same"
            pad_idx = PADDINGS.index(default_pad) if default_pad in PADDINGS else 1
            config["padding"] = st.selectbox("Padding", PADDINGS, index=pad_idx, key=f"pad_{idx}")
        with col4:
            default_act = current_config.get("activation", "relu") if current_config else "relu"
            act_idx = ACTIVATIONS.index(default_act) if (isinstance(default_act, str) and default_act in ACTIVATIONS) else 1
            config["activation"] = st.selectbox("Activation", ACTIVATIONS, index=act_idx,
                                                 key=f"act_{idx}")

    elif layer_type == "DepthwiseConv2D":
        ks_str = st.text_input("Kernel Size", "3,3", key=f"ks_{idx}")
        try:
            config["kernel_size"] = [int(x.strip()) for x in ks_str.split(",")]
        except ValueError:
            config["kernel_size"] = [3, 3]
        col1, col2 = st.columns(2)
        with col1:
            config["padding"] = st.selectbox("Padding", PADDINGS, key=f"pad_{idx}")
        with col2:
            config["activation"] = st.selectbox("Activation", ACTIVATIONS,
                                                 key=f"act_{idx}")

    elif layer_type in ("MaxPooling1D", "AveragePooling1D"):
        config["pool_size"] = st.number_input("Pool Size", min_value=1, value=2,
                                               key=f"ps_{idx}")

    elif layer_type in ("MaxPooling2D", "AveragePooling2D"):
        default_ps = ",".join(map(str, current_config.get("pool_size", [2,2]))) if current_config and "pool_size" in current_config else "2,2"
        ps_str = st.text_input("Pool Size", default_ps, key=f"ps_{idx}")
        try:
            config["pool_size"] = [int(x.strip()) for x in ps_str.split(",")]
        except ValueError:
            config["pool_size"] = [2, 2]

    elif layer_type in ("MaxPooling3D", "AveragePooling3D"):
        ps_str = st.text_input("Pool Size", "2,2,2", key=f"ps_{idx}")
        try:
            config["pool_size"] = [int(x.strip()) for x in ps_str.split(",")]
        except ValueError:
            config["pool_size"] = [2, 2, 2]

    elif layer_type.startswith("Global"):
        st.caption("Global pooling — no parameters to configure.")

    elif layer_type in ("SimpleRNN", "LSTM", "GRU"):
        col1, col2 = st.columns(2)
        with col1:
            default_units = current_config.get("units", 64) if current_config else 64
            config["units"] = st.number_input("Units", min_value=1, value=default_units,
                                               key=f"units_{idx}")
        with col2:
            default_act = current_config.get("activation", "tanh") if current_config else "tanh"
            act_idx = ACTIVATIONS.index(default_act) if (isinstance(default_act, str) and default_act in ACTIVATIONS) else ACTIVATIONS.index("tanh")
            config["activation"] = st.selectbox("Activation", ACTIVATIONS,
                                                 index=act_idx,
                                                 key=f"act_{idx}")
        col3, col4 = st.columns(2)
        with col3:
            default_ret = current_config.get("return_sequences", False) if current_config else False
            config["return_sequences"] = st.checkbox("Return Sequences", value=default_ret,
                                                      key=f"ret_seq_{idx}")
        with col4:
            default_drop = current_config.get("recurrent_dropout", 0.0) if current_config else 0.0
            config["recurrent_dropout"] = st.slider("Recurrent Dropout", 0.0, 1.0,
                                                     default_drop, 0.05, key=f"rec_drop_{idx}")

    elif layer_type == "Bidirectional":
        col1, col2 = st.columns(2)
        with col1:
            default_wrapped = current_config.get("wrapped_layer_type", "LSTM") if current_config else "LSTM"
            wrapped_types = ["LSTM", "GRU", "SimpleRNN"]
            wrapped_idx = wrapped_types.index(default_wrapped) if default_wrapped in wrapped_types else 0
            config["wrapped_layer_type"] = st.selectbox("RNN Type",
                                                         wrapped_types,
                                                         index=wrapped_idx,
                                                         key=f"wrapped_{idx}")
        with col2:
            default_units = current_config.get("units", 64) if current_config else 64
            config["units"] = st.number_input("Units", min_value=1, value=default_units,
                                               key=f"units_{idx}")
        default_ret = current_config.get("return_sequences", False) if current_config else False
        config["return_sequences"] = st.checkbox("Return Sequences", value=default_ret,
                                                  key=f"ret_seq_{idx}")

    elif layer_type in ("BatchNormalization", "LayerNormalization"):
        st.caption("Normalization — using default parameters.")

    elif layer_type == "Embedding":
        col1, col2 = st.columns(2)
        with col1:
            default_dim = current_config.get("input_dim", 10000) if current_config else 10000
            config["input_dim"] = st.number_input("Vocab Size (input_dim)",
                                                    min_value=1, value=default_dim,
                                                    key=f"input_dim_{idx}")
        with col2:
            default_out = current_config.get("output_dim", 128) if current_config else 128
            config["output_dim"] = st.number_input("Embedding Dim (output_dim)",
                                                    min_value=1, value=default_out,
                                                    key=f"output_dim_{idx}")

    return config
