"""
Neural Network Visualizer
Renders publication-quality network diagrams using matplotlib.
Each layer is shown as a column of nodes with connections between layers.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import BytesIO


# ─── Color Schemes ───────────────────────────────────────────────────────────

LAYER_COLORS = {
    "Input":                "#4FC3F7",
    "Dense":                "#7E57C2",
    "Activation":           "#AB47BC",
    "Dropout":              "#78909C",
    "Flatten":              "#8D6E63",
    "Reshape":              "#8D6E63",
    # Conv
    "Conv1D":               "#66BB6A",
    "Conv2D":               "#43A047",
    "Conv3D":               "#2E7D32",
    "SeparableConv2D":      "#81C784",
    "DepthwiseConv2D":      "#A5D6A7",
    "Conv2DTranspose":      "#4CAF50",
    # Pooling
    "MaxPooling1D":         "#FFA726",
    "MaxPooling2D":         "#FB8C00",
    "MaxPooling3D":         "#EF6C00",
    "AveragePooling1D":     "#FFCC80",
    "AveragePooling2D":     "#FFB74D",
    "AveragePooling3D":     "#FFA000",
    "GlobalAveragePooling1D": "#FFB300",
    "GlobalAveragePooling2D": "#FFA000",
    "GlobalMaxPooling1D":     "#FF8F00",
    "GlobalMaxPooling2D":     "#FF6F00",
    # Recurrent
    "SimpleRNN":            "#EF5350",
    "LSTM":                 "#E53935",
    "GRU":                  "#C62828",
    "Bidirectional":        "#FF7043",
    # Normalization
    "BatchNormalization":   "#26C6DA",
    "LayerNormalization":   "#00ACC1",
    # Embedding
    "Embedding":            "#EC407A",
}

LAYER_CATEGORIES = {
    "Input": "Core",
    "Dense": "Core",
    "Activation": "Core",
    "Dropout": "Regularization",
    "Flatten": "Core",
    "Reshape": "Core",
    "Conv1D": "Convolutional",
    "Conv2D": "Convolutional",
    "Conv3D": "Convolutional",
    "SeparableConv2D": "Convolutional",
    "DepthwiseConv2D": "Convolutional",
    "Conv2DTranspose": "Convolutional",
    "MaxPooling1D": "Pooling",
    "MaxPooling2D": "Pooling",
    "MaxPooling3D": "Pooling",
    "AveragePooling1D": "Pooling",
    "AveragePooling2D": "Pooling",
    "AveragePooling3D": "Pooling",
    "GlobalAveragePooling1D": "Pooling",
    "GlobalAveragePooling2D": "Pooling",
    "GlobalMaxPooling1D": "Pooling",
    "GlobalMaxPooling2D": "Pooling",
    "SimpleRNN": "Recurrent",
    "LSTM": "Recurrent",
    "GRU": "Recurrent",
    "Bidirectional": "Recurrent",
    "BatchNormalization": "Normalization",
    "LayerNormalization": "Normalization",
    "Embedding": "Embedding",
}


def _get_display_nodes(layer_type, layer_config):
    """Determine how many nodes to display for a layer."""
    if layer_type == "Input":
        total = 1
        if layer_config.get("input_shape"):
            shape = layer_config["input_shape"]
            total = shape[-1] if isinstance(shape, list) else shape
        return min(total, 8), total

    elif layer_type == "Dense":
        total = layer_config.get("units", 64)
        return min(total, 8), total

    elif layer_type in ("Conv1D", "Conv2D", "Conv3D", "SeparableConv2D",
                         "DepthwiseConv2D", "Conv2DTranspose"):
        total = layer_config.get("filters", 32)
        return min(total, 6), total

    elif layer_type in ("SimpleRNN", "LSTM", "GRU"):
        total = layer_config.get("units", 64)
        return min(total, 8), total

    elif layer_type == "Bidirectional":
        total = (layer_config.get("units", 64)) * 2
        return min(total, 8), total

    elif layer_type == "Embedding":
        total = layer_config.get("output_dim", 128)
        return min(total, 6), total

    elif layer_type in ("Dropout", "Activation", "BatchNormalization",
                         "LayerNormalization"):
        return 4, None

    elif layer_type in ("Flatten",):
        return 5, None

    elif layer_type == "Reshape":
        return 4, None

    elif "Pooling" in layer_type:
        return 4, None

    return 4, None


def _draw_nodes(ax, x, y_positions, color, node_radius=0.15, alpha=0.9):
    """Draw nodes at specified positions."""
    for y in y_positions:
        circle = plt.Circle((x, y), node_radius, color=color,
                            ec='white', linewidth=1.5, alpha=alpha, zorder=3)
        ax.add_patch(circle)


def _draw_connections(ax, x1, y1_positions, x2, y2_positions, alpha=0.08):
    """Draw connections between two layers."""
    for y1 in y1_positions:
        for y2 in y2_positions:
            ax.plot([x1, x2], [y1, y2], color='#B0BEC5', linewidth=0.5,
                    alpha=alpha, zorder=1)


def draw_network(layers_data, figsize=None):
    """
    Draw a neural network diagram.

    Args:
        layers_data: list of dicts with keys:
            - layer_type: str
            - config: dict (layer params)
            - output_shape: str
            - params: int
        figsize: optional (width, height) tuple

    Returns:
        matplotlib figure
    """
    n_layers = len(layers_data)
    if n_layers == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Add layers to visualize your network",
                ha='center', va='center', fontsize=14, color='#78909C',
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        fig.patch.set_facecolor('#0E1117')
        return fig

    # Calculate layout
    layer_spacing = 2.5
    total_width = (n_layers - 1) * layer_spacing + 4
    if figsize is None:
        figsize = (max(total_width, 8), 8)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    all_node_positions = []
    node_radius = 0.18

    for i, layer in enumerate(layers_data):
        x = 2 + i * layer_spacing
        lt = layer["layer_type"]
        config = layer.get("config", {})
        color = LAYER_COLORS.get(lt, "#90A4AE")

        n_display, total_count = _get_display_nodes(lt, config)

        # Compute y-positions for nodes
        max_height = 5
        if n_display <= 1:
            y_positions = [0]
        else:
            y_positions = np.linspace(-max_height / 2, max_height / 2, n_display)

        # If total exceeds display, add ellipsis indicator
        show_ellipsis = total_count is not None and total_count > n_display

        if show_ellipsis and n_display > 2:
            # Remove middle node and replace with dots
            mid = n_display // 2
            y_with_gap = list(y_positions[:mid]) + list(y_positions[mid + 1:])
            ellipsis_y = y_positions[mid]
        else:
            y_with_gap = list(y_positions)
            ellipsis_y = None

        _draw_nodes(ax, x, y_with_gap, color, node_radius)

        if ellipsis_y is not None:
            ax.text(x, ellipsis_y, "⋮", ha='center', va='center',
                    fontsize=16, color='white', fontweight='bold', zorder=4)

        all_node_positions.append(y_with_gap)

        # Draw connections to previous layer
        if i > 0:
            prev_x = 2 + (i - 1) * layer_spacing
            conn_alpha = max(0.03, 0.15 / max(len(all_node_positions[i - 1]) * len(y_with_gap) / 16, 1))
            _draw_connections(ax, prev_x, all_node_positions[i - 1],
                            x, y_with_gap, alpha=conn_alpha)

        # Layer label (below)
        display_name = lt
        if len(display_name) > 12:
            # Abbreviate long names
            display_name = display_name.replace("Pooling", "Pool").replace("Average", "Avg").replace("Global", "G.")
            display_name = display_name.replace("Normalization", "Norm").replace("Bidirectional", "Bidir.")

        ax.text(x, -max_height / 2 - 0.8, display_name,
                ha='center', va='top', fontsize=8, color='white',
                fontweight='bold', zorder=5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                         alpha=0.8, edgecolor='none'))

        # Shape label (further below)
        shape_str = layer.get("output_shape", "")
        if shape_str:
            ax.text(x, -max_height / 2 - 1.5, shape_str,
                    ha='center', va='top', fontsize=6.5, color='#B0BEC5',
                    zorder=5)

        # Param count (above)
        params = layer.get("params", 0)
        if params > 0:
            if params >= 1_000_000:
                param_str = f"{params / 1_000_000:.1f}M"
            elif params >= 1_000:
                param_str = f"{params / 1_000:.1f}K"
            else:
                param_str = str(params)
            ax.text(x, max_height / 2 + 0.6, f"{param_str} params",
                    ha='center', va='bottom', fontsize=6.5, color='#78909C',
                    zorder=5)

        # Total count label (inside top node)
        if total_count is not None and total_count > 1:
            ax.text(x, max(y_with_gap) + 0.5, str(total_count),
                    ha='center', va='bottom', fontsize=7, color=color,
                    fontweight='bold', zorder=5)

    # Axis settings
    x_min = 0.5
    x_max = 2 + (n_layers - 1) * layer_spacing + 1.5
    y_min = -max_height / 2 - 2.5
    y_max = max_height / 2 + 1.8

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text((x_min + x_max) / 2, y_max - 0.2, "Network Architecture",
            ha='center', va='top', fontsize=13, color='white',
            fontweight='bold', zorder=5)

    plt.tight_layout()
    return fig


def draw_legend(layer_types_used):
    """Draw a color legend for the layer types used."""
    categories = {}
    for lt in layer_types_used:
        cat = LAYER_CATEGORIES.get(lt, "Other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(lt)

    n_items = sum(len(v) for v in categories.values())
    if n_items == 0:
        return None

    fig, ax = plt.subplots(figsize=(3, max(n_items * 0.35 + len(categories) * 0.4, 2)))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    ax.axis('off')

    y = 1.0
    for cat, items in categories.items():
        ax.text(0.05, y, cat.upper(), ha='left', va='top',
                fontsize=8, color='#78909C', fontweight='bold',
                transform=ax.transAxes)
        y -= 0.06
        for lt in items:
            color = LAYER_COLORS.get(lt, "#90A4AE")
            ax.add_patch(plt.Rectangle((0.05, y - 0.02), 0.06, 0.035,
                                       transform=ax.transAxes,
                                       facecolor=color, edgecolor='none',
                                       alpha=0.9))
            name = lt.replace("Normalization", "Norm").replace("Pooling", "Pool")
            ax.text(0.14, y, name, ha='left', va='top',
                    fontsize=7, color='white', transform=ax.transAxes)
            y -= 0.05
        y -= 0.02

    plt.tight_layout()
    return fig
