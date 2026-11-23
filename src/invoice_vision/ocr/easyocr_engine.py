"""EasyOCR engine module with JSON caching and low-memory execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class OCRItem:
    """Represents a single recognized text bounding region."""

    box: List[List[float]]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    text: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert item to dictionary for JSON serialization."""
        return {
            "box": self.box,
            "text": self.text,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRItem":
        """Instantiate item from serialized dictionary."""
        return cls(
            box=data["box"],
            text=str(data["text"]),
            confidence=float(data["confidence"]),
        )


class EasyOCREngine:
    """Wrapper around EasyOCR providing disk caching and low-footprint inference."""

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        gpu: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        model_storage_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.cache_dir = Path(cache_dir or "data/processed/ocr_cache")
        self.model_storage_dir = Path(model_storage_dir or "data/models/easyocr")
        self._reader: Any = None

    def _get_reader(self) -> Any:
        """Lazily initialize the EasyOCR reader singleton."""
        if self._reader is None:
            import easyocr

            self.model_storage_dir.mkdir(parents=True, exist_ok=True)
            user_net_dir = self.model_storage_dir / "user_network"
            user_net_dir.mkdir(parents=True, exist_ok=True)
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                model_storage_directory=str(self.model_storage_dir),
                user_network_directory=str(user_net_dir),
            )
        return self._reader

    def recognize_image(
        self,
        image_path: Union[str, Path],
        force_reload: bool = False,
    ) -> List[OCRItem]:
        """Run OCR on a single image file or load from cache if previously processed.

        Processes one image at a time to maintain a minimal memory footprint.
        """
        img_path = Path(image_path)
        cache_file = self.cache_dir / f"{img_path.name}.json"

        # 1. Check disk cache
        if not force_reload and cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [OCRItem.from_dict(d) for d in data]

        # 2. Run inference on image
        reader = self._get_reader()
        raw_results = reader.readtext(str(img_path))

        ocr_items: List[OCRItem] = []
        for bbox, text, prob in raw_results:
            # Normalize bounding box points to floats
            box_points = [[float(coord[0]), float(coord[1])] for coord in bbox]
            ocr_items.append(
                OCRItem(
                    box=box_points,
                    text=str(text).strip(),
                    confidence=float(prob),
                )
            )

        # 3. Save to disk cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump([item.to_dict() for item in ocr_items], f, indent=2, ensure_ascii=False)

        # Force garbage collection of internal image tensors
        gc.collect()

        return ocr_items

    @staticmethod
    def get_full_text(items: List[OCRItem]) -> str:
        """Concatenate all OCR text tokens in top-to-bottom spatial order."""
        # Sort items primarily by top-y coordinate, then by left-x coordinate
        sorted_items = sorted(
            items,
            key=lambda item: (item.box[0][1] if item.box else 0, item.box[0][0] if item.box else 0),
        )
        return "\n".join(item.text for item in sorted_items if item.text)
