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
            use_fast=True,
            device=0 if torch.cuda.is_available() else -1,
        )
    return _nsfw_pipeline


def contains_nsfw(frames: torch.Tensor) -> bool:
    """
    Retourne True si au moins une frame est détectée NSFW
    Accepte (N_frames, C, H, W) ou (C, N_frames, H, W)
    """
    classifier = load_nsfw_pipeline()
    
    # Auto-détecte le format
    if frames.shape[0] in [1, 3]:  # (C, N, H, W)
        N = frames.shape[1]
        get_frame = lambda idx: frames[:, idx]
    elif frames.shape[1] in [1, 3]:  # (N, C, H, W)
        N = frames.shape[0]
        get_frame = lambda idx: frames[idx]
    else:
        raise ValueError("Format de tensor inattendu")
    
    for idx in range(N):
        frame = get_frame(idx).detach().cpu()
        # Normalisation
        if frame.min() < 0:
            frame = ((frame + 1) / 2).clamp(0, 1)
        else:
            frame = frame.clamp(0, 1)
        img = Image.fromarray((frame.permute(1, 2, 0).numpy() * 255).astype("uint8"))

        # DEBUG : Sauvegarde et inspection
        # img.save(f"debug_frame_{idx}.png")
        result = classifier(img, top_k=5)
        print(f"Frame {idx}: {result}")  # Voir les scores

        # Accepte le "top1" OU tout label "NSFW" > 0.4 (exemple seuil bas)
        for entry in result:
            label = entry["label"].lower()
            score = entry["score"]
            if label in {"porn", "sexy", "hentai"} and score > 0.4:
                return True

    return False

