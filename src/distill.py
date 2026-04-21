import torch
import torch.nn.functional as F
from torch.func import functional_call          


#чек alogirthm 1 из статьи, https://arxiv.org/pdf/1811.10959, стр.5

# эта функция для инициализации дистилированного датасета. 
# можно было взять рандомную инициализацию, но тут https://github.com/ssnl/dataset-distillation
# предложили взять средние пиксели по всему датасету (а-ля средняя матрица)
def _init_from_class_means(train_loader, num_per_class, num_classes, img_shape, device):
    sums = torch.zeros(num_classes, *img_shape, device=device)
    counts = torch.zeros(num_classes, device=device)
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        for c in range(num_classes):
            mask = y == c
            if mask.any():
                sums[c] += x[mask].sum(0)
                counts[c] += mask.sum()
    means = sums / counts.view(num_classes, 1, 1, 1).clamp(min=1)
    distilled_x = means.unsqueeze(1).expand(-1, num_per_class, -1, -1, -1).reshape(
        num_classes * num_per_class, *img_shape
    )
    return distilled_x


# Line 6 алгоритма
def _inner_step(model, params, distilled_x, distilled_y, lr):
    logits = functional_call(model, params, distilled_x)
    loss = F.cross_entropy(logits, distilled_y)
    grads = torch.autograd.grad(loss, list(params.values()), create_graph=True)
    return {name: p - lr * g for (name, p), g in zip(params.items(), grads)}

# Line 9 алгоритма
def distill(model, train_loader, config: dict, device: torch.device):
    img_shape = (1, 28, 28)
    num_classes = 10
    n = config["num_per_class"] * num_classes
    distilled_x = _init_from_class_means(
        train_loader, config["num_per_class"], num_classes, img_shape, device
    ).detach().requires_grad_(True)
    distilled_y = torch.arange(num_classes, device=device).repeat_interleave(
        config["num_per_class"]
    )
    distilled_lr = torch.tensor(
        config["inner_lr_init"], dtype=torch.float32, device=device, requires_grad=True
    )
    outer_opt = torch.optim.Adam([distilled_x, distilled_lr], lr=config["outer_lr"])
    real_iter = iter(train_loader)
    for step in range(1, config["num_steps"] + 1):
        # Здесь сэмплируем параметры
        model.reset_parameters()
        params = {
            name: p.detach().clone().requires_grad_(True)
            for name, p in model.named_parameters()
        }
        # Здесь делаем сколько-то шагов по дистилированном датасету
        for _ in range(config["inner_steps"]):
            params = _inner_step(model, params, distilled_x, distilled_y, distilled_lr)

        # Line 7 алгоритма
        try:
            real_x, real_y = next(real_iter)
        except StopIteration:
            real_iter = iter(train_loader)
            real_x, real_y = next(real_iter)
        real_x, real_y = real_x.to(device), real_y.to(device)

        logits = functional_call(model, params, real_x)
        outer_loss = F.cross_entropy(logits, real_y)
        outer_opt.zero_grad()
        outer_loss.backward()
        outer_opt.step()
        with torch.no_grad():
            distilled_lr.clamp_(min=1e-5)
        if step % config["num_steps"] // 20 == 0:
            print(
                f"step{step:>5}/{config['num_steps']} | "
                f"outer_loss={outer_loss.item():.4f} | "
                f"learned_lr={distilled_lr.item():.5f}"
            )

    return distilled_x.detach(), distilled_y.detach(), distilled_lr.detach()
