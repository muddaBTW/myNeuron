"""
Pydantic models for neural network layer definitions and configurations.
Supports all major TensorFlow/Keras layer types.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Union, Literal
from enum import Enum


# ─── Enums ───────────────────────────────────────────────────────────────────

class ActivationType(str, Enum):
    NONE = "None"
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SELU = "selu"
    SWISH = "swish"
    GELU = "gelu"
    SOFTPLUS = "softplus"
    LINEAR = "linear"


class PaddingType(str, Enum):
    VALID = "valid"
    SAME = "same"


class InitializerType(str, Enum):
    GLOROT_UNIFORM = "glorot_uniform"
    GLOROT_NORMAL = "glorot_normal"
    HE_UNIFORM = "he_uniform"
    HE_NORMAL = "he_normal"
    LECUN_UNIFORM = "lecun_uniform"
    LECUN_NORMAL = "lecun_normal"
    ZEROS = "zeros"
    ONES = "ones"
    RANDOM_NORMAL = "random_normal"
    RANDOM_UNIFORM = "random_uniform"


class OptimizerType(str, Enum):
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADAMAX = "adamax"
    NADAM = "nadam"
    ADAMW = "adamw"


class LossType(str, Enum):
    CATEGORICAL_CROSSENTROPY = "categorical_crossentropy"
    SPARSE_CATEGORICAL_CROSSENTROPY = "sparse_categorical_crossentropy"
    BINARY_CROSSENTROPY = "binary_crossentropy"
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    COSINE_SIMILARITY = "cosine_similarity"
    KL_DIVERGENCE = "kl_divergence"


class MetricType(str, Enum):
    ACCURACY = "accuracy"
    PRECISION = "Precision"
    RECALL = "Recall"
    AUC = "AUC"
    F1_SCORE = "F1Score"
    MAE = "mae"
    MSE = "mse"
    RMSE = "RootMeanSquaredError"


class LayerType(str, Enum):
    # Core
    INPUT = "Input"
    DENSE = "Dense"
    ACTIVATION = "Activation"
    DROPOUT = "Dropout"
    FLATTEN = "Flatten"
    RESHAPE = "Reshape"
    # Convolutional
    CONV1D = "Conv1D"
    CONV2D = "Conv2D"
    CONV3D = "Conv3D"
    SEPARABLE_CONV2D = "SeparableConv2D"
    DEPTHWISE_CONV2D = "DepthwiseConv2D"
    CONV2D_TRANSPOSE = "Conv2DTranspose"
    # Pooling
    MAX_POOL_1D = "MaxPooling1D"
    MAX_POOL_2D = "MaxPooling2D"
    MAX_POOL_3D = "MaxPooling3D"
    AVG_POOL_1D = "AveragePooling1D"
    AVG_POOL_2D = "AveragePooling2D"
    AVG_POOL_3D = "AveragePooling3D"
    GLOBAL_AVG_POOL_1D = "GlobalAveragePooling1D"
    GLOBAL_AVG_POOL_2D = "GlobalAveragePooling2D"
    GLOBAL_MAX_POOL_1D = "GlobalMaxPooling1D"
    GLOBAL_MAX_POOL_2D = "GlobalMaxPooling2D"
    # Recurrent
    SIMPLE_RNN = "SimpleRNN"
    LSTM = "LSTM"
    GRU = "GRU"
    BIDIRECTIONAL = "Bidirectional"
    # Normalization
    BATCH_NORM = "BatchNormalization"
    LAYER_NORM = "LayerNormalization"
    # Embedding
    EMBEDDING = "Embedding"


# ─── Layer Config ────────────────────────────────────────────────────────────

class LayerConfig(BaseModel):
    """Universal layer configuration model. 
    Fields are optional - only relevant ones are used per layer_type."""
    layer_type: LayerType
    name: Optional[str] = None

    # Input
    input_shape: Optional[List[int]] = None

    # Dense / RNN
    units: Optional[int] = None
    activation: Optional[ActivationType] = ActivationType.NONE
    use_bias: Optional[bool] = True
    kernel_initializer: Optional[InitializerType] = InitializerType.GLOROT_UNIFORM

    # Conv
    filters: Optional[int] = None
    kernel_size: Optional[Union[int, List[int]]] = None
    strides: Optional[Union[int, List[int]]] = None
    padding: Optional[PaddingType] = PaddingType.VALID

    # Pooling
    pool_size: Optional[Union[int, List[int]]] = None

    # Dropout
    rate: Optional[float] = None

    # Reshape
    target_shape: Optional[List[int]] = None

    # Embedding
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None

    # RNN specific
    return_sequences: Optional[bool] = False
    recurrent_dropout: Optional[float] = 0.0

    # Bidirectional
    wrapped_layer_type: Optional[Literal["LSTM", "GRU", "SimpleRNN"]] = None


# ─── Compile & Training Config ──────────────────────────────────────────────

class OptimizerConfig(BaseModel):
    optimizer_type: OptimizerType = OptimizerType.ADAM
    learning_rate: float = 0.001
    momentum: Optional[float] = 0.0       # SGD
    beta_1: Optional[float] = 0.9         # Adam
    beta_2: Optional[float] = 0.999       # Adam
    weight_decay: Optional[float] = None  # AdamW

class CompileConfig(BaseModel):
    optimizer: OptimizerConfig = OptimizerConfig()
    loss: LossType = LossType.SPARSE_CATEGORICAL_CROSSENTROPY
    metrics: List[MetricType] = [MetricType.ACCURACY]

class TrainingConfig(BaseModel):
    epochs: int = 10
    batch_size: int = 32
    validation_split: float = 0.2
    shuffle: bool = True

class NetworkConfig(BaseModel):
    """Complete neural network definition."""
    layers: List[LayerConfig]
    compile_config: CompileConfig = CompileConfig()
    training_config: TrainingConfig = TrainingConfig()
    model_name: str = "MyModel"


# ─── Response Models ─────────────────────────────────────────────────────────

class LayerSummary(BaseModel):
    name: str
    layer_type: str
    output_shape: str
    param_count: int

class ModelSummaryResponse(BaseModel):
    layers: List[LayerSummary]
    total_params: int
    trainable_params: int
    non_trainable_params: int
    warnings: List[str] = []

class CodeGenerationResponse(BaseModel):
    code: str
    success: bool
    errors: List[str] = []

class ValidationResponse(BaseModel):
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []

class LayerCatalogItem(BaseModel):
    layer_type: str
    category: str
    description: str
    parameters: List[str]
