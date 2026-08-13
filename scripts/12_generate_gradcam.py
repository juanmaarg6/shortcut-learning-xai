"""Generate side-by-side Grad-CAM comparisons for fixed Waterbirds cases."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from shortcut_learning.models import build_resnet50_classifier
from shortcut_learning.transforms import IMAGENET_MEAN, IMAGENET_STD

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "waterbird_complete95_forest2water2"

SELECTED_CASES_PATH = PROJECT_ROOT / "results" / "xai" / "selected_cases.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures" / "xai"

REFERENCE_SEED = 123
IMAGE_SIZE = 224
RESIZE_SIZE = 232

CLASS_NAMES = {
    0: "Landbird",
    1: "Waterbird",
}

GROUP_NAMES = {
    0: "Landbird on land",
    1: "Landbird on water",
    2: "Waterbird on land",
    3: "Waterbird on water",
}


def load_model(
    method: str,
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    """Load one frozen trained checkpoint."""
    checkpoint_path = (
        PROJECT_ROOT / "artifacts" / "checkpoints" / method / f"seed_{seed}" / "best.pt"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = build_resnet50_classifier(
        num_classes=2,
        pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def prepare_image(
    image_path: Path,
) -> tuple[np.ndarray, torch.Tensor]:
    """Create aligned RGB visualization and normalized model input."""
    spatial_transform = transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
        ]
    )
    tensor_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        cropped = spatial_transform(image)

    rgb = np.asarray(cropped, dtype=np.float32) / 255.0
    tensor = tensor_transform(cropped).unsqueeze(0)

    return rgb, tensor


def predict(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
) -> tuple[int, float]:
    """Return predicted class and confidence."""
    tensor = tensor.to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits.float(), dim=1)[0]

    prediction = int(probabilities.argmax())
    confidence = float(probabilities[prediction])

    return prediction, confidence


def generate_cam(
    cam: GradCAM,
    tensor: torch.Tensor,
    target_class: int,
    rgb: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Generate a Grad-CAM overlay for a specified class target."""
    targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(
        input_tensor=tensor.to(device),
        targets=targets,
    )[0]

    return show_cam_on_image(
        rgb,
        grayscale_cam,
        use_rgb=True,
        image_weight=0.55,
    )


def main() -> None:
    """Create one comparison figure using cases fixed before CAM generation."""
    if not SELECTED_CASES_PATH.is_file():
        raise FileNotFoundError(
            "Selected XAI cases were not found. "
            "Run scripts/11_select_xai_cases.py first."
        )

    selected = pd.read_csv(SELECTED_CASES_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Grad-CAM comparison")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Reference seed: {REFERENCE_SEED}")
    print(f"Cases: {len(selected)}")
    print("Target policy: each CAM explains that model's predicted class.")

    erm_model = load_model(
        method="erm",
        seed=REFERENCE_SEED,
        device=device,
    )
    balanced_model = load_model(
        method="group_balanced",
        seed=REFERENCE_SEED,
        device=device,
    )

    figure, axes = plt.subplots(
        nrows=len(selected),
        ncols=3,
        figsize=(11, 3.4 * len(selected)),
    )

    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    with (
        GradCAM(
            model=erm_model,
            target_layers=[erm_model.layer4[-1]],
        ) as erm_cam,
        GradCAM(
            model=balanced_model,
            target_layers=[balanced_model.layer4[-1]],
        ) as balanced_cam,
    ):
        for row_index, row in enumerate(selected.itertuples(index=False)):
            image_path = DATASET_DIR / row.img_filename
            rgb, tensor = prepare_image(image_path)

            erm_prediction, erm_confidence = predict(
                model=erm_model,
                tensor=tensor,
                device=device,
            )
            balanced_prediction, balanced_confidence = predict(
                model=balanced_model,
                tensor=tensor,
                device=device,
            )

            erm_overlay = generate_cam(
                cam=erm_cam,
                tensor=tensor,
                target_class=erm_prediction,
                rgb=rgb,
                device=device,
            )
            balanced_overlay = generate_cam(
                cam=balanced_cam,
                tensor=tensor,
                target_class=balanced_prediction,
                rgb=rgb,
                device=device,
            )

            original_ax = axes[row_index, 0]
            erm_ax = axes[row_index, 1]
            balanced_ax = axes[row_index, 2]

            original_ax.imshow(rgb)
            erm_ax.imshow(erm_overlay)
            balanced_ax.imshow(balanced_overlay)

            for ax in (original_ax, erm_ax, balanced_ax):
                ax.axis("off")

            true_name = CLASS_NAMES[int(row.label)]
            group_name = GROUP_NAMES[int(row.group)]

            original_ax.set_title(
                f"{row.xai_category}\nTrue: {true_name}\n{group_name}",
                fontsize=9,
            )

            erm_ax.set_title(
                "ERM\n"
                f"Pred: {CLASS_NAMES[erm_prediction]} "
                f"({100.0 * erm_confidence:.1f}%)",
                fontsize=9,
            )

            balanced_ax.set_title(
                "Group-balanced\n"
                f"Pred: {CLASS_NAMES[balanced_prediction]} "
                f"({100.0 * balanced_confidence:.1f}%)",
                fontsize=9,
            )

    figure.suptitle(
        "Grad-CAM: ERM vs Group-Balanced ERM",
        fontsize=15,
    )
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "gradcam_erm_vs_group_balanced.png"

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Saved: {output_path}")
    print(
        "\nInterpretation reminder: Grad-CAM provides attribution evidence, "
        "not proof of causal feature use."
    )


if __name__ == "__main__":
    main()
