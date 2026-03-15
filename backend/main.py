"""
myNeuron — FastAPI Backend
Neural Network Code Generation & Validation API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from models import (
    NetworkConfig, LayerCatalogItem, CodeGenerationResponse,
    ValidationResponse, ModelSummaryResponse, LayerSummary, LayerType
)
from code_generator import generate_code
from validators import validate_network, get_model_summary


app = FastAPI(
    title="myNeuron API",
    description="Neural Network Design & TensorFlow Code Generation API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Layer Catalog ───────────────────────────────────────────────────────────

LAYER_CATALOG: List[LayerCatalogItem] = [
    # Core
    LayerCatalogItem(layer_type="Input", category="Core", description="Input layer defining the shape of input data", parameters=["input_shape"]),
    LayerCatalogItem(layer_type="Dense", category="Core", description="Fully connected layer", parameters=["units", "activation", "use_bias", "kernel_initializer"]),
    LayerCatalogItem(layer_type="Activation", category="Core", description="Applies an activation function", parameters=["activation"]),
    LayerCatalogItem(layer_type="Dropout", category="Core", description="Randomly sets input units to 0 for regularization", parameters=["rate"]),
    LayerCatalogItem(layer_type="Flatten", category="Core", description="Flattens multi-dimensional input to 1D", parameters=[]),
    LayerCatalogItem(layer_type="Reshape", category="Core", description="Reshapes input to a target shape", parameters=["target_shape"]),
    # Convolutional
    LayerCatalogItem(layer_type="Conv1D", category="Convolutional", description="1D convolution (temporal)", parameters=["filters", "kernel_size", "strides", "padding", "activation"]),
    LayerCatalogItem(layer_type="Conv2D", category="Convolutional", description="2D convolution (spatial)", parameters=["filters", "kernel_size", "strides", "padding", "activation"]),
    LayerCatalogItem(layer_type="Conv3D", category="Convolutional", description="3D convolution (volumetric)", parameters=["filters", "kernel_size", "strides", "padding", "activation"]),
    LayerCatalogItem(layer_type="SeparableConv2D", category="Convolutional", description="Depthwise separable 2D convolution", parameters=["filters", "kernel_size", "padding", "activation"]),
    LayerCatalogItem(layer_type="DepthwiseConv2D", category="Convolutional", description="Depthwise 2D convolution", parameters=["kernel_size", "padding", "activation"]),
    LayerCatalogItem(layer_type="Conv2DTranspose", category="Convolutional", description="Transposed 2D convolution (deconvolution)", parameters=["filters", "kernel_size", "padding", "activation"]),
    # Pooling
    LayerCatalogItem(layer_type="MaxPooling1D", category="Pooling", description="Max pooling for 1D data", parameters=["pool_size"]),
    LayerCatalogItem(layer_type="MaxPooling2D", category="Pooling", description="Max pooling for 2D data", parameters=["pool_size"]),
    LayerCatalogItem(layer_type="MaxPooling3D", category="Pooling", description="Max pooling for 3D data", parameters=["pool_size"]),
    LayerCatalogItem(layer_type="AveragePooling1D", category="Pooling", description="Average pooling for 1D data", parameters=["pool_size"]),
    LayerCatalogItem(layer_type="AveragePooling2D", category="Pooling", description="Average pooling for 2D data", parameters=["pool_size"]),
    LayerCatalogItem(layer_type="AveragePooling3D", category="Pooling", description="Average pooling for 3D data", parameters=["pool_size"]),
    LayerCatalogItem(layer_type="GlobalAveragePooling1D", category="Pooling", description="Global average pooling for 1D data", parameters=[]),
    LayerCatalogItem(layer_type="GlobalAveragePooling2D", category="Pooling", description="Global average pooling for 2D data", parameters=[]),
    LayerCatalogItem(layer_type="GlobalMaxPooling1D", category="Pooling", description="Global max pooling for 1D data", parameters=[]),
    LayerCatalogItem(layer_type="GlobalMaxPooling2D", category="Pooling", description="Global max pooling for 2D data", parameters=[]),
    # Recurrent
    LayerCatalogItem(layer_type="SimpleRNN", category="Recurrent", description="Fully-connected RNN", parameters=["units", "activation", "return_sequences", "recurrent_dropout"]),
    LayerCatalogItem(layer_type="LSTM", category="Recurrent", description="Long Short-Term Memory layer", parameters=["units", "activation", "return_sequences", "recurrent_dropout"]),
    LayerCatalogItem(layer_type="GRU", category="Recurrent", description="Gated Recurrent Unit layer", parameters=["units", "activation", "return_sequences", "recurrent_dropout"]),
    LayerCatalogItem(layer_type="Bidirectional", category="Recurrent", description="Bidirectional wrapper for RNN layers", parameters=["wrapped_layer_type", "units", "return_sequences"]),
    # Normalization
    LayerCatalogItem(layer_type="BatchNormalization", category="Normalization", description="Batch normalization layer", parameters=[]),
    LayerCatalogItem(layer_type="LayerNormalization", category="Normalization", description="Layer normalization layer", parameters=[]),
    # Embedding
    LayerCatalogItem(layer_type="Embedding", category="Embedding", description="Turns positive integers into dense vectors", parameters=["input_dim", "output_dim"]),
]


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "myNeuron API"}


@app.get("/api/layer-catalog", response_model=List[LayerCatalogItem])
async def get_layer_catalog():
    """Return all available layer types with their parameters."""
    return LAYER_CATALOG


@app.post("/api/generate-code", response_model=CodeGenerationResponse)
async def generate_network_code(config: NetworkConfig):
    """Generate TensorFlow/Keras code from network configuration."""
    try:
        # Validate first
        validation = validate_network(config)
        if not validation["valid"]:
            return CodeGenerationResponse(
                code="",
                success=False,
                errors=validation["errors"]
            )

        code = generate_code(config)
        return CodeGenerationResponse(
            code=code,
            success=True,
            errors=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")


@app.post("/api/validate", response_model=ValidationResponse)
async def validate_architecture(config: NetworkConfig):
    """Validate the network architecture for compatibility issues."""
    try:
        result = validate_network(config)
        return ValidationResponse(
            valid=result["valid"],
            errors=result["errors"],
            warnings=result["warnings"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.post("/api/model-summary", response_model=ModelSummaryResponse)
async def get_model_summary_endpoint(config: NetworkConfig):
    """Get per-layer model summary with shapes and parameter counts."""
    try:
        summary = get_model_summary(config)
        return ModelSummaryResponse(
            layers=[LayerSummary(**l) for l in summary["layers"]],
            total_params=summary["total_params"],
            trainable_params=summary["trainable_params"],
            non_trainable_params=summary["non_trainable_params"],
            warnings=summary["warnings"]
        )
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        print(f"ERROR in /api/model-summary: {error_details}")
        raise HTTPException(status_code=500, detail=f"Summary computation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
