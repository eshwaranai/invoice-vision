"""Pre-label suggestion module for automated bounding box proposal."""

from invoice_vision.prelabel.suggest import (
    CLASS_NAMES,
    SuggestedBox,
    WordBox,
    suggest_document_boxes,
)

__all__ = [
    "CLASS_NAMES",
    "SuggestedBox",
    "WordBox",
    "suggest_document_boxes",
]
