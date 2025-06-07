# File: apps/smgrid/signal_models.py

from pydantic import BaseModel, ValidationError, TypeAdapter
from typing import List, Union, Literal, Optional

class BaseOrder(BaseModel):
    type: Literal["limit", "market"]
    direction: Literal["buy", "sell"]
    price: float
    tp: Optional[float]
    sl: Optional[float]
    tag: Optional[str]
    spacing: Optional[float]

class GridSignal(BaseModel):
    mode: Literal["grid"]
    orders: List[BaseOrder]

class TrendOrder(BaseOrder):
    type: Literal["market"] # type: ignore
    spacing: float # type: ignore

class TrendSignal(BaseModel):
    mode: Literal["trend"]
    orders: TrendOrder

SignalAdapter = TypeAdapter(Union[GridSignal, TrendSignal])

def validate_signal(raw: dict) -> Union[GridSignal, TrendSignal, None]:
    try:
        return SignalAdapter.validate_python(raw)
    except ValidationError as e:
        from core.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.warning(f"[SignalValidator] Invalid signal format: {e}")
        return None
