"""
myNeuron — Visual Neural Network Designer
Streamlit Frontend Application
"""

import streamlit as st
import requests
import json
from layers_ui import render_layer_config, LAYER_CATEGORIES, ACTIVATIONS
from visualizer import draw_network, draw_legend

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="myNeuron — Neural Network Designer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Dark theme enhancements */
    .stApp {
        background-color: #0E1117;
    }

    /* Layer card styling */
    .layer-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
        transition: all 0.3s ease;
    }
    .layer-card:hover {
        border-color: #4FC3F7;
        box-shadow: 0 0 15px rgba(79, 195, 247, 0.2);
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, #1e1e30 0%, #2d1b4e 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #3d2d5c;
    }
    .stat-value {
        font-size: 28px;
        font-weight: bold;
        color: #BB86FC;
    }
    .stat-label {
        font-size: 12px;
        color: #78909C;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Header */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sub-header {
        color: #78909C;
        font-size: 1rem;
        margin-top: -10px;
    }

    /* Code block */
    .code-container {
        background: #1a1a2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 5px;
    }

    /* Summary table */
    .summary-table {
        width: 100%;
        border-collapse: collapse;
    }
    .summary-table th {
        background: #1e1e30;
        color: #BB86FC;
        padding: 10px;
        text-align: left;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .summary-table td {
        padding: 8px 10px;
        border-bottom: 1px solid #2d3748;
        color: #e0e0e0;
        font-size: 13px;
    }
    .summary-table tr:hover td {
        background: rgba(79, 195, 247, 0.05);
    }

    div[data-testid="stExpander"] {
        border: 1px solid #2d3748;
        border-radius: 10px;
        margin-bottom: 8px;
    }

    /* Sidebar category selector */
    .sidebar .stSelectbox label {
        color: #BB86FC;
    }
</style>
""", unsafe_allow_html=True)


# ─── Backend URL ─────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000"


# ─── Session State Init ─────────────────────────────────────────────────────

if "layers" not in st.session_state:
    st.session_state.layers = []
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""
if "model_summary" not in st.session_state:
    st.session_state.model_summary = None
if "validation_result" not in st.session_state:
    st.session_state.validation_result = None


# ─── Helper Functions ────────────────────────────────────────────────────────

def build_network_config():
    """Build the network config dict from session state."""
    layer_configs = []
    for layer in st.session_state.layers:
        lc = {"layer_type": layer["layer_type"]}
        config = layer.get("config", {})

        # Map config to expected API fields
        for key in ["units", "activation", "use_bias", "kernel_initializer",
                     "filters", "kernel_size", "strides", "padding",
                     "pool_size", "rate", "input_shape", "target_shape",
                     "input_dim", "output_dim", "return_sequences",
                     "recurrent_dropout", "wrapped_layer_type"]:
            if key in config:
                val = config[key]
                if key == "activation" and val == "None":
                    val = "None"
                lc[key] = val

        layer_configs.append(lc)

    return {
        "layers": layer_configs,
        "compile_config": {
            "optimizer": {
                "optimizer_type": st.session_state.get("optimizer", "adam"),
                "learning_rate": st.session_state.get("learning_rate", 0.001),
            },
            "loss": st.session_state.get("loss", "sparse_categorical_crossentropy"),
            "metrics": st.session_state.get("metrics", ["accuracy"]),
        },
        "training_config": {
            "epochs": st.session_state.get("epochs", 10),
            "batch_size": st.session_state.get("batch_size", 32),
            "validation_split": st.session_state.get("val_split", 0.2),
            "shuffle": True,
        },
        "model_name": st.session_state.get("model_name", "MyModel"),
    }


def call_api(endpoint, data=None, method="POST"):
    """Call the FastAPI backend."""
    try:
        if method == "POST":
            resp = requests.post(f"{API_URL}{endpoint}", json=data, timeout=10)
        else:
            resp = requests.get(f"{API_URL}{endpoint}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Could not connect to backend. Make sure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        return None


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 myNeuron")
    st.markdown("---")

    # Model name
    st.text_input("Model Name", value="MyModel", key="model_name")

    st.markdown("### ➕ Add Layer")

    # Category selector
    category = st.selectbox("Category", list(LAYER_CATEGORIES.keys()))
    layer_type = st.selectbox("Layer Type", LAYER_CATEGORIES[category])

    if st.button("Add Layer", use_container_width=True, type="primary"):
        st.session_state.layers.append({
            "layer_type": layer_type,
            "config": {"layer_type": layer_type}
        })
        st.rerun()

    st.markdown("---")

    # Quick templates
    st.markdown("### 📋 Templates")
    template = st.selectbox("Load Template", [
        "— Select —",
        "Simple MLP (MNIST)",
        "CNN (Image Classification)",
        "LSTM (Sequence)",
        "Autoencoder"
    ])

    if template != "— Select —" and st.button("Load Template", use_container_width=True):
        if template == "Simple MLP (MNIST)":
            st.session_state.layers = [
                {"layer_type": "Input", "config": {"layer_type": "Input", "input_shape": [784]}},
                {"layer_type": "Dense", "config": {"layer_type": "Dense", "units": 256, "activation": "relu"}},
                {"layer_type": "BatchNormalization", "config": {"layer_type": "BatchNormalization"}},
                {"layer_type": "Dropout", "config": {"layer_type": "Dropout", "rate": 0.3}},
                {"layer_type": "Dense", "config": {"layer_type": "Dense", "units": 128, "activation": "relu"}},
                {"layer_type": "Dropout", "config": {"layer_type": "Dropout", "rate": 0.2}},
                {"layer_type": "Dense", "config": {"layer_type": "Dense", "units": 10, "activation": "softmax"}},
            ]
        elif template == "CNN (Image Classification)":
            st.session_state.layers = [
                {"layer_type": "Input", "config": {"layer_type": "Input", "input_shape": [28, 28, 1]}},
                {"layer_type": "Conv2D", "config": {"layer_type": "Conv2D", "filters": 32, "kernel_size": [3, 3], "padding": "same", "activation": "relu"}},
                {"layer_type": "BatchNormalization", "config": {"layer_type": "BatchNormalization"}},
                {"layer_type": "MaxPooling2D", "config": {"layer_type": "MaxPooling2D", "pool_size": [2, 2]}},
                {"layer_type": "Conv2D", "config": {"layer_type": "Conv2D", "filters": 64, "kernel_size": [3, 3], "padding": "same", "activation": "relu"}},
                {"layer_type": "BatchNormalization", "config": {"layer_type": "BatchNormalization"}},
                {"layer_type": "MaxPooling2D", "config": {"layer_type": "MaxPooling2D", "pool_size": [2, 2]}},
                {"layer_type": "Flatten", "config": {"layer_type": "Flatten"}},
                {"layer_type": "Dense", "config": {"layer_type": "Dense", "units": 128, "activation": "relu"}},
                {"layer_type": "Dropout", "config": {"layer_type": "Dropout", "rate": 0.5}},
                {"layer_type": "Dense", "config": {"layer_type": "Dense", "units": 10, "activation": "softmax"}},
            ]
        elif template == "LSTM (Sequence)":
            st.session_state.layers = [
                {"layer_type": "Input", "config": {"layer_type": "Input", "input_shape": [100, 50]}},
                {"layer_type": "LSTM", "config": {"layer_type": "LSTM", "units": 128, "return_sequences": True}},
                {"layer_type": "Dropout", "config": {"layer_type": "Dropout", "rate": 0.2}},
                {"layer_type": "LSTM", "config": {"layer_type": "LSTM", "units": 64, "return_sequences": False}},
                {"layer_type": "Dense", "config": {"layer_type": "Dense", "units": 32, "activation": "relu"}},
                {"layer_type": "Dense", "config": {"layer_type": "Dense", "units": 10, "activation": "softmax"}},
            ]
        elif template == "Autoencoder":
            st.session_state.layers = [
                {"layer_type": "Input", "config": {"layer_type": "Input", "input_shape": [784]}},
                {"layer_type": "Dense", "config": {"layer_type": "Dense", "units": 256, "activation": "relu"}},
                {"layer_type": "Dense", "config": {"layer_type": "Dense", "units": 64, "activation": "relu"}},
                {"layer_type": "Dense", "config": {"layer_type": "Dense", "units": 32, "activation": "relu"}},
                {"layer_type": "Dense", "config": {"layer_type": "Dense", "units": 64, "activation": "relu"}},
                {"layer_type": "Dense", "config": {"layer_type": "Dense", "units": 256, "activation": "relu"}},
                {"layer_type": "Dense", "config": {"layer_type": "Dense", "units": 784, "activation": "sigmoid"}},
            ]
        st.rerun()

    st.markdown("---")

    # Compile Config
    st.markdown("### ⚙️ Compile Settings")

    st.selectbox("Optimizer", ["adam", "sgd", "rmsprop", "adagrad", "adamax", "nadam", "adamw"],
                 key="optimizer")

    st.number_input("Learning Rate", min_value=0.00001, max_value=1.0,
                    value=0.001, step=0.0001, format="%.5f", key="learning_rate")

    st.selectbox("Loss Function", [
        "sparse_categorical_crossentropy", "categorical_crossentropy",
        "binary_crossentropy", "mse", "mae", "huber",
        "cosine_similarity", "kl_divergence"
    ], key="loss")

    st.multiselect("Metrics", ["accuracy", "Precision", "Recall", "AUC",
                                "F1Score", "mae", "mse", "RootMeanSquaredError"],
                   default=["accuracy"], key="metrics")

    st.markdown("---")

    # Training Config
    st.markdown("### 🏋️ Training Settings")
    st.number_input("Epochs", min_value=1, value=10, key="epochs")
    st.number_input("Batch Size", min_value=1, value=32, key="batch_size")
    st.slider("Validation Split", 0.0, 0.5, 0.2, 0.05, key="val_split")

    st.markdown("---")

    # Clear all
    if st.button("🗑️ Clear All Layers", use_container_width=True):
        st.session_state.layers = []
        st.session_state.generated_code = ""
        st.session_state.model_summary = None
        st.rerun()


# ─── Main Content ────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">myNeuron</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Design • Visualize • Export Neural Networks</p>',
            unsafe_allow_html=True)

# Quick stats
if st.session_state.layers:
    cols = st.columns(4)
    n_layers = len(st.session_state.layers)

    # Calculate rough param count
    total_params = 0
    config_data = build_network_config()

    with cols[0]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{n_layers}</div>
            <div class="stat-label">Layers</div>
        </div>
        """, unsafe_allow_html=True)

    # Unique layer types
    unique_types = len(set(l["layer_type"] for l in st.session_state.layers))
    with cols[1]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{unique_types}</div>
            <div class="stat-label">Layer Types</div>
        </div>
        """, unsafe_allow_html=True)

    # Has input layer?
    has_input = any(l["layer_type"] == "Input" for l in st.session_state.layers)
    with cols[2]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{"✅" if has_input else "⚠️"}</div>
            <div class="stat-label">Input Layer</div>
        </div>
        """, unsafe_allow_html=True)

    # Last layer
    last_type = st.session_state.layers[-1]["layer_type"] if st.session_state.layers else "—"
    with cols[3]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="font-size: 18px;">{last_type}</div>
            <div class="stat-label">Output Layer</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab_arch, tab_viz, tab_code, tab_summary = st.tabs([
    "🏗️ Architecture", "📊 Visualization", "💻 Code", "📋 Summary"
])


# ── Architecture Tab ──

with tab_arch:
    if not st.session_state.layers:
        st.info("👈 Add layers from the sidebar or load a template to get started!")
    else:
        for i, layer in enumerate(st.session_state.layers):
            col_main, col_actions = st.columns([6, 1])

            with col_main:
                with st.expander(f"**Layer {i}** — {layer['layer_type']}", expanded=False):
                    new_config = render_layer_config(layer["layer_type"], i)
                    st.session_state.layers[i]["config"] = new_config

            with col_actions:
                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    if i > 0 and st.button("⬆", key=f"up_{i}", help="Move up"):
                        st.session_state.layers[i], st.session_state.layers[i - 1] = \
                            st.session_state.layers[i - 1], st.session_state.layers[i]
                        st.rerun()
                with c2:
                    if i < len(st.session_state.layers) - 1 and st.button("⬇", key=f"down_{i}", help="Move down"):
                        st.session_state.layers[i], st.session_state.layers[i + 1] = \
                            st.session_state.layers[i + 1], st.session_state.layers[i]
                        st.rerun()
                with c3:
                    if st.button("🗑", key=f"del_{i}", help="Delete"):
                        st.session_state.layers.pop(i)
                        st.rerun()


# ── Visualization Tab ──

with tab_viz:
    if not st.session_state.layers:
        st.info("Add layers to see the network visualization.")
    else:
        # Build visualization data
        viz_data = []
        # Try to get model summary for shape info
        config_data = build_network_config()
        summary_result = call_api("/api/model-summary", config_data)

        for i, layer in enumerate(st.session_state.layers):
            entry = {
                "layer_type": layer["layer_type"],
                "config": layer.get("config", {}),
                "output_shape": "",
                "params": 0
            }
            if summary_result and i < len(summary_result.get("layers", [])):
                entry["output_shape"] = summary_result["layers"][i]["output_shape"]
                entry["params"] = summary_result["layers"][i]["param_count"]
            viz_data.append(entry)

        col_viz, col_legend = st.columns([4, 1])

        with col_viz:
            fig = draw_network(viz_data)
            st.pyplot(fig, use_container_width=True)

        with col_legend:
            used_types = list(set(l["layer_type"] for l in st.session_state.layers))
            legend_fig = draw_legend(used_types)
            if legend_fig:
                st.pyplot(legend_fig, use_container_width=True)

        # Show warnings
        if summary_result and summary_result.get("warnings"):
            for w in summary_result["warnings"]:
                st.warning(f"⚠️ {w}")

        # Total params footer
        if summary_result:
            total = summary_result.get("total_params", 0)
            trainable = summary_result.get("trainable_params", 0)
            non_trainable = summary_result.get("non_trainable_params", 0)
            st.markdown(f"""
            **Total Parameters:** {total:,} | 
            **Trainable:** {trainable:,} | 
            **Non-trainable:** {non_trainable:,}
            """)


# ── Code Tab ──

with tab_code:
    if not st.session_state.layers:
        st.info("Add layers to generate TensorFlow code.")
    else:
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            generate_clicked = st.button("⚡ Generate Code", use_container_width=True,
                                          type="primary")
        with col_btn2:
            validate_clicked = st.button("✅ Validate Architecture", use_container_width=True)

        if generate_clicked:
            config_data = build_network_config()
            with st.spinner("Generating TensorFlow code..."):
                result = call_api("/api/generate-code", config_data)
                if result:
                    if result["success"]:
                        st.session_state.generated_code = result["code"]
                        st.success("✅ Code generated successfully!")
                    else:
                        for err in result.get("errors", []):
                            st.error(f"❌ {err}")

        if validate_clicked:
            config_data = build_network_config()
            with st.spinner("Validating..."):
                result = call_api("/api/validate", config_data)
                if result:
                    if result["valid"]:
                        st.success("✅ Architecture is valid!")
                    else:
                        for err in result.get("errors", []):
                            st.error(f"❌ {err}")
                    for w in result.get("warnings", []):
                        st.warning(f"⚠️ {w}")

        if st.session_state.generated_code:
            st.markdown("---")
            st.code(st.session_state.generated_code, language="python")

            st.download_button(
                label="📥 Download Code",
                data=st.session_state.generated_code,
                file_name=f"{st.session_state.get('model_name', 'model')}.py",
                mime="text/x-python",
                use_container_width=True,
            )


# ── Summary Tab ──

with tab_summary:
    if not st.session_state.layers:
        st.info("Add layers to see the model summary.")
    else:
        config_data = build_network_config()
        summary = call_api("/api/model-summary", config_data)

        if summary:
            # Build summary data for native Streamlit table
            import pandas as pd
            rows = []
            for i, layer in enumerate(summary.get("layers", [])):
                params = layer["param_count"]
                rows.append({
                    "#": i,
                    "Layer Name": layer["name"],
                    "Type": layer["layer_type"],
                    "Output Shape": layer["output_shape"],
                    "Parameters": f"{params:,}" if params > 0 else "0"
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Totals
            cols = st.columns(3)
            with cols[0]:
                st.metric("Total Parameters", f"{summary['total_params']:,}")
            with cols[1]:
                st.metric("Trainable", f"{summary['trainable_params']:,}")
            with cols[2]:
                st.metric("Non-trainable", f"{summary['non_trainable_params']:,}")

            # Warnings
            if summary.get("warnings"):
                st.markdown("### ⚠️ Warnings")
                for w in summary["warnings"]:
                    st.warning(w)


# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #78909C; font-size: 12px;">'
    'myNeuron — Visual Neural Network Designer | Built with Streamlit + FastAPI'
    '</div>',
    unsafe_allow_html=True
)
