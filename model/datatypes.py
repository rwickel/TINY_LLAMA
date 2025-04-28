
from dataclasses import dataclass
from typing import List, Optional, Union
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from typing_extensions import Annotated
import torch
import base64

class Role(Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class BuiltinTool(Enum):
    brave_search = "brave_search"
    wolfram_alpha = "wolfram_alpha"
    photogen = "photogen"
    code_interpreter = "code_interpreter"


Primitive = Union[str, int, float, bool, None]
RecursiveType = Union[Primitive, List[Primitive], Dict[str, Primitive]]


class ToolCall(BaseModel):
    call_id: str
    tool_name: Union[BuiltinTool, str]
    # Plan is to deprecate the Dict in favor of a JSON string
    # that is parsed on the client side instead of trying to manage
    # the recursive type here.
    # Making this a union so that client side can start prepping for this change.
    # Eventually, we will remove both the Dict and arguments_json field,
    # and arguments will just be a str
    arguments: Union[str, Dict[str, RecursiveType]]
    arguments_json: Optional[str] = None

    @field_validator("tool_name", mode="before")
    @classmethod
    def validate_field(cls, v):
        if isinstance(v, str):
            try:
                return BuiltinTool(v)
            except ValueError:
                return v
        return v


class ToolPromptFormat(Enum):
    """Prompt format for calling custom / zero shot tools.

    :cvar json: JSON format for calling tools. It takes the form:
        {
            "type": "function",
            "function" : {
                "name": "function_name",
                "description": "function_description",
                "parameters": {...}
            }
        }
    :cvar function_tag: Function tag format, pseudo-XML. This looks like:
        <function=function_name>(parameters)</function>

    :cvar python_list: Python list. The output is a valid Python expression that can be
        evaluated to a list. Each element in the list is a function call. Example:
        ["function_name(param1, param2)", "function_name(param1, param2)"]
    """

    json = "json"
    function_tag = "function_tag"
    python_list = "python_list"


class StopReason(Enum):
    end_of_turn = "end_of_turn"
    end_of_message = "end_of_message"
    out_of_tokens = "out_of_tokens"


class ToolParamDefinition(BaseModel):
    param_type: str
    description: Optional[str] = None
    required: Optional[bool] = True
    default: Optional[Any] = None


class ToolDefinition(BaseModel):
    tool_name: Union[BuiltinTool, str]
    description: Optional[str] = None
    parameters: Optional[Dict[str, ToolParamDefinition]] = None

    @field_validator("tool_name", mode="before")
    @classmethod
    def validate_field(cls, v):
        if isinstance(v, str):
            try:
                return BuiltinTool(v)
            except ValueError:
                return v
        return v


class RawMediaItem(BaseModel):
    type: Literal["image"] = "image"
    data: bytes | BytesIO

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("data")
    def serialize_data(self, data: Optional[bytes], _info):
        if data is None:
            return None
        return base64.b64encode(data).decode("utf-8")

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, v):
        if isinstance(v, str):
            return base64.b64decode(v)
        return v


class RawTextItem(BaseModel):
    type: Literal["text"] = "text"
    text: str


RawContentItem = Annotated[Union[RawTextItem, RawMediaItem], Field(discriminator="type")]

RawContent = str | RawContentItem | List[RawContentItem]


class RawMessage(BaseModel):
    role: Literal["user"] | Literal["system"] | Literal["tool"] | Literal["assistant"]
    content: RawContent

    # This is for RAG but likely should be absorbed into content
    context: Optional[RawContent] = None

    # These are for the output message coming from the assistant
    stop_reason: Optional[StopReason] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)


class GenerationResult(BaseModel):
    token: int
    text: str
    logprobs: Optional[List[float]] = None

    source: Literal["input"] | Literal["output"]

    # index within the batch
    batch_idx: int
    # whether generation for this item is already finished. note that tokens can
    # get returned even afterwards since other items in the batch can still be generating tokens
    finished: bool
    # because a batch is parallel processed, useful decoding for one item can correspond to processing
    # pad tokens or tokens beyond EOS for other items. we could have decided to return None for this case
    # but it's more convenient to return a list of GenerationResult and filter out the ignored tokens
    ignore_token: bool


class QuantizationMode(str, Enum):
    none = "none"
    fp8_mixed = "fp8_mixed"
    int4_mixed = "int4_mixed"

class Size(BaseModel):
    height: int
    width: int    

class VisionArgs(BaseModel):
    image_size: Size
    patch_size: Size

    # parameters for the encoder transformer
    dim: int
    n_layers: int
    n_heads: int
    mlp_ratio: float
    output_dim: int

    pixel_shuffle_ratio: float


@dataclass
class MaskedEmbedding:
    embedding: torch.Tensor
    mask: torch.Tensor


@dataclass
class LLMInput:
    """
    This is the input to the LLM from the "user" -- the user in this case views the
    Llama4 model holistically and does not care or know about its inner workings (e.g.,
    whether it has an encoder or if it is early fusion or not.)

    This is distinct from the "TransformerInput" class which is really the Llama4
    backbone operating on early fused modalities and producing text output
    """

    tokens: torch.Tensor

    # images are already pre-processed (resized, tiled, etc.)
    images: Optional[List[torch.Tensor]] = None


@dataclass
class TransformerInput:
    """
    This is the "core" backbone transformer of the Llama4 model. Inputs for other modalities
    are expected to be "embedded" via encoders sitting before this layer in the model.
    """

    tokens: torch.Tensor

    # tokens_position defines the position of the tokens in each batch,
    # - when it is a tensor ([batch_size,]), it is the start position of the tokens in each batch
    # - when it is an int, the start position are the same for all batches
    tokens_position: Union[torch.Tensor, int]
    image_embedding: Optional[MaskedEmbedding] = None


@dataclass
class LLMOutput:
    logits: torch.Tensor   


TransformerOutput = LLMOutput