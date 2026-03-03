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


def render_layer_config(layer_type: str, idx: int) -> dict:
    """Render configuration widgets for a specific layer type and return config dict."""
    config = {"layer_type": layer_type}

    if layer_type == "Input":
        shape_str = st.text_input(
            "Input Shape", value="784",
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
            config["units"] = st.number_input("Units", min_value=1, value=64,
                                               key=f"units_{idx}")
        with col2:
            config["activation"] = st.selectbox("Activation", ACTIVATIONS,
                                                 key=f"act_{idx}")
        config["use_bias"] = st.checkbox("Use Bias", value=True, key=f"bias_{idx}")
        config["kernel_initializer"] = st.selectbox("Initializer", INITIALIZERS,
                                                     key=f"init_{idx}")

    elif layer_type == "Activation":
        config["activation"] = st.selectbox("Activation", ACTIVATIONS[1:],
                                             key=f"act_{idx}")

    elif layer_type == "Dropout":
        config["rate"] = st.slider("Dropout Rate", 0.0, 1.0, 0.5, 0.05,
                                    key=f"rate_{idx}")

    elif layer_type == "Flatten":
        st.caption("Flattens input — no parameters to configure.")

    elif layer_type == "Reshape":
        shape_str = st.text_input("Target Shape", value="28,28,1",
                                   key=f"target_shape_{idx}")
        try:
            config["target_shape"] = [int(x.strip()) for x in shape_str.split(",")]
        except ValueError:
            config["target_shape"] = [28, 28, 1]

    elif layer_type in ("Conv1D", "Conv2D", "Conv3D", "SeparableConv2D",
                         "Conv2DTranspose"):
        col1, col2 = st.columns(2)
        with col1:
            config["filters"] = st.number_input("Filters", min_value=1, value=32,
                                                 key=f"filters_{idx}")
        with col2:
            if layer_type == "Conv1D":
                config["kernel_size"] = st.number_input("Kernel Size", min_value=1,
                                                         value=3, key=f"ks_{idx}")
            elif layer_type == "Conv3D":
                ks_str = st.text_input("Kernel Size", "3,3,3", key=f"ks_{idx}")
                try:
                    config["kernel_size"] = [int(x.strip()) for x in ks_str.split(",")]
                except ValueError:
                    config["kernel_size"] = [3, 3, 3]
            else:
                ks_str = st.text_input("Kernel Size", "3,3", key=f"ks_{idx}")
                try:
                    config["kernel_size"] = [int(x.strip()) for x in ks_str.split(",")]
                except ValueError:
                    config["kernel_size"] = [3, 3]

        col3, col4 = st.columns(2)
        with col3:
            config["padding"] = st.selectbox("Padding", PADDINGS, key=f"pad_{idx}")
        with col4:
            config["activation"] = st.selectbox("Activation", ACTIVATIONS,
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
        ps_str = st.text_input("Pool Size", "2,2", key=f"ps_{idx}")
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
            config["units"] = st.number_input("Units", min_value=1, value=64,
                                               key=f"units_{idx}")
        with col2:
            config["activation"] = st.selectbox("Activation", ACTIVATIONS,
                                                 index=ACTIVATIONS.index("tanh"),
                                                 key=f"act_{idx}")
        col3, col4 = st.columns(2)
        with col3:
            config["return_sequences"] = st.checkbox("Return Sequences", False,
                                                      key=f"ret_seq_{idx}")
        with col4:
            config["recurrent_dropout"] = st.slider("Recurrent Dropout", 0.0, 1.0,
                                                     0.0, 0.05, key=f"rec_drop_{idx}")

    elif layer_type == "Bidirectional":
        col1, col2 = st.columns(2)
        with col1:
            config["wrapped_layer_type"] = st.selectbox("RNN Type",
                                                         ["LSTM", "GRU", "SimpleRNN"],
                                                         key=f"wrapped_{idx}")
        with col2:
            config["units"] = st.number_input("Units", min_value=1, value=64,
                                               key=f"units_{idx}")
        config["return_sequences"] = st.checkbox("Return Sequences", False,
                                                  key=f"ret_seq_{idx}")

    elif layer_type in ("BatchNormalization", "LayerNormalization"):
        st.caption("Normalization — using default parameters.")

    elif layer_type == "Embedding":
        col1, col2 = st.columns(2)
        with col1:
            config["input_dim"] = st.number_input("Vocab Size (input_dim)",
                                                    min_value=1, value=10000,
                                                    key=f"input_dim_{idx}")
        with col2:
            config["output_dim"] = st.number_input("Embedding Dim (output_dim)",
                                                    min_value=1, value=128,
                                                    key=f"output_dim_{idx}")

    return config
