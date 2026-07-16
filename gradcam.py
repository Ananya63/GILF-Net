# ============================================================
# Grad-CAM++ Visualization (multi-class safe)
# ============================================================
# Generates Grad-CAM++ heatmaps for the fused feature map produced
# by `model.fusion`, and overlays them on the original input image
# alongside the true and predicted class labels.
# ============================================================

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class GradCAMPlusPlus:
    """Grad-CAM++ using forward/backward hooks on a target module."""

    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module

        self.gradients = None
        self.activations = None

        self.target_module.register_forward_hook(self.forward_hook)
        self.target_module.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()

        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward(retain_graph=True)

        gradients = self.gradients[0]      # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Grad-CAM++ weights
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3

        eps = 1e-8
        denominator = 2 * grad_2 + torch.sum(
            activations * grad_3, dim=(1, 2), keepdim=True
        )
        denominator = torch.where(denominator != 0.0, denominator, torch.tensor(eps, device=denominator.device))

        alpha = grad_2 / denominator
        weights = torch.sum(alpha * F.relu(gradients), dim=(1, 2))

        cam = torch.sum(weights.view(-1, 1, 1) * activations, dim=0)

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy(), class_idx


def visualize_gradcam(model, loader, device, class_names, num_images=6,
                       mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                       target_layer=None):
    """
    Runs Grad-CAM++ on `num_images` samples drawn from `loader` and
    plots Original / Grad-CAM++ / Overlay side by side.

    `target_layer` defaults to `model.fusion` (the GILF module's output),
    which is the natural attribution point for this architecture.
    """
    model.eval()

    if target_layer is None:
        target_layer = model.fusion

    gradcam = GradCAMPlusPlus(model, target_layer)

    mean = np.array(mean)
    std = np.array(std)

    images_shown = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        for i in range(images.size(0)):
            if images_shown >= num_images:
                return

            img_tensor = images[i].unsqueeze(0)
            cam, pred_class = gradcam.generate(img_tensor)
            true_class = labels[i].item()

            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = std * img + mean
            img = np.clip(img, 0, 1)

            cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

            overlay = 0.6 * img + 0.4 * heatmap
            overlay = np.clip(overlay, 0, 1)

            plt.figure(figsize=(10, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title(f"Original\nTrue: {class_names[true_class]}")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(cam, cmap='jet')
            plt.title("Grad-CAM++")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title(f"Pred: {class_names[pred_class]}")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

            images_shown += 1


if __name__ == "__main__":
    # Example usage (assumes a trained model, test_loader, and
    # class_names are already available):
    #
    # from models.fusion_model import LiftingFusionNet
    # model = LiftingFusionNet(num_classes=8)
    # model.load_state_dict(torch.load("best_model.pth"))
    # model.to(device)
    #
    # visualize_gradcam(model, test_loader, device, class_names, num_images=6)
    pass
