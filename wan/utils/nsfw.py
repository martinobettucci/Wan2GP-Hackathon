from typing import Optional
from PIL import Image
import torch
from transformers import pipeline

_nsfw_pipeline: Optional[object] = None


def load_nsfw_pipeline() -> object:
    """Lazy load the NSFW detection pipeline."""
    global _nsfw_pipeline
    if _nsfw_pipeline is None:
        _nsfw_pipeline = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=0 if torch.cuda.is_available() else -1,
        )
    return _nsfw_pipeline


def contains_nsfw(frames: torch.Tensor) -> bool:
    """Return True if any frame is flagged NSFW."""
    classifier = load_nsfw_pipeline()
    for idx in range(frames.shape[1]):
        frame = frames[:, idx]
        frame = ((frame + 1) / 2).clamp(0, 1)
        img = Image.fromarray((frame.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
        result = classifier(img, top_k=1)[0]
        label = result.get("label")
        score = result.get("score", 0)
        if label in {"porn", "sexy", "hentai"} and score > 0.5:
            return True
    return False
