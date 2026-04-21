import json
import os
import torch
import torchvision
import torchvision.transforms as T
from src.model import ConvNet
from src.distill import distill
from src.evaluate import evaluate_distilled, evaluate_random_init, evaluate_full_dataset
from src.visualize import save_distilled_grid

CONFIG = {
    "num_per_class": 1,      
    "inner_lr_init": 0.02,   
    "outer_lr": 0.01,         
    "inner_steps": 1,         
    "num_steps": 2000,      
    "batch_size": 256,

    "eval_inner_steps": 1,    
    "num_eval_trials": 10,    

    "full_dataset_epochs": 5,
    "full_dataset_lr": 0.001,
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_set = torchvision.datasets.MNIST(
        "/kaggle/working/data", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        "/kaggle/working/data", train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1000, shuffle=False, num_workers=2
    )

    model = ConvNet().to(device)
    n_total = CONFIG["num_per_class"] * 10

    print("Deriving distilled dataset")

    distilled_x, distilled_y, distilled_lr = distill(
        model, train_loader, CONFIG, device
    )
    print(f"Learned LR: {distilled_lr.item():.5f}")
    print(f"Evaluating distilled data")
    dist_mean, dist_std, dist_all = evaluate_distilled(
        model, distilled_x, distilled_y, distilled_lr, test_loader, CONFIG, device
    )
    print(f"Distilled ({n_total} imgs): {dist_mean*100:.2f}% ± {dist_std*100:.2f}%")
    print("Baselines")
    rand_mean, rand_std = evaluate_random_init(model, test_loader, device)
    print(f"Random init:  {rand_mean*100:.2f}% ± {rand_std*100:.2f}%")
    print("Training on full dataset")
    full_acc = evaluate_full_dataset(
        model, train_loader, test_loader, device,
        epochs=CONFIG["full_dataset_epochs"],
        lr=CONFIG["full_dataset_lr"],
    )
    print(f"Full dataset: {full_acc*100:.2f}%")
    results_dir = "/kaggle/working/results"
    os.makedirs(results_dir, exist_ok=True)
    results = {
        "config": CONFIG,
        "distilled": {
            "num_images": n_total,
            "mean_acc": round(dist_mean, 6),
            "std_acc": round(dist_std, 6),
            "all_accs": [round(a, 6) for a in dist_all],
            "learned_lr": round(distilled_lr.item(), 6),
        },
        "baselines": {
            "random_init_mean": round(rand_mean, 6),
            "random_init_std": round(rand_std, 6),
            "full_dataset_acc": round(full_acc, 6),
        },
    }
    json_path = os.path.join(results_dir, "mnist_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to  {json_path}")

    pt_path = os.path.join(results_dir, "distilled_mnist.pt")
    torch.save({"x": distilled_x, "y": distilled_y, "lr": distilled_lr}, pt_path)
    print(f"Distilled data saved to  {pt_path}")

    grid_path = os.path.join(results_dir, "distilled_mnist_grid.png")
    save_distilled_grid(distilled_x, distilled_y, grid_path)
    print(f"Random init: {rand_mean*100:.1f}% ± {rand_std*100:.1f}%")
    print(f"Distilled ({n_total} imgs): {dist_mean*100:.1f}% ± {dist_std*100:.1f}%")
    print(f"Full dataset: {full_acc*100:.1f}%")
if __name__ == "__main__":
    main()