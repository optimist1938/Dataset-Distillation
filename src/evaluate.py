import torch
import torch.nn.functional as F
import math
def _test_accuracy(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

def _mean_std(values):
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return mean, std

def evaluate_distilled(model, distilled_x, distilled_y, learned_lr,
                        test_loader, config, device):
    accs = []
    for _ in range(config["num_eval_trials"]):
        model.reset_parameters()
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=learned_lr.item())
        for _ in range(config["eval_inner_steps"]):
            loss = F.cross_entropy(model(distilled_x), distilled_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        accs.append(_test_accuracy(model, test_loader, device))
    mean, std = _mean_std(accs)
    return mean, std, accs


def evaluate_random_init(model, test_loader, device, num_trials=10):
    accs = []
    for _ in range(num_trials):
        model.reset_parameters()
        accs.append(_test_accuracy(model, test_loader, device))
    return _mean_std(accs)


def evaluate_full_dataset(model, train_loader, test_loader, device,
                           epochs=5, lr=0.01):
    model.reset_parameters()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return _test_accuracy(model, test_loader, device)