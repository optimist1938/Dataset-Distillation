import torch
import os


def save_distilled_grid(distilled_x, distilled_y, out_path, mean=0.1307, std=0.3081):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    num_classes = int(distilled_y.max().item()) + 1
    num_per_class = (distilled_y == 0).sum().item()
    imgs = distilled_x.detach().cpu()
    imgs = imgs * std + mean
    imgs = imgs.clamp(0, 1)
    fig, axes = plt.subplots(num_per_class, num_classes,
                              figsize=(num_classes * 1.2, num_per_class * 1.2))
    if num_per_class == 1:
        axes = axes[None, :]   

    for c in range(num_classes):
        idx = (distilled_y == c).nonzero(as_tuple=True)[0]
        for row, i in enumerate(idx):
            ax = axes[row, c]
            ax.imshow(imgs[i].squeeze(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(str(c), fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Distilled image grid saved to {out_path}")
