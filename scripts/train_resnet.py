"""Pretrained ResNet-18: head only vs fine-tuning layer4, compared with the same network trained from scratch."""
import torch
from torch import nn
from torchvision.models import resnet18

from museum_guide.data import class_names, device, load_metadata, loaders, split
from museum_guide.models import pretrained_resnet18
from museum_guide.train import REPORTS, cpu_latency_ms, fit, n_params, plot_confusion, plot_history, predict, save_results

SIZE = 128
FIG = REPORTS / "figures"


def scratch_resnet18(n_classes):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def main():
    dev = device()
    df = load_metadata()
    parts, classes = split(df), class_names(df)
    data = loaders(parts, SIZE, augment=True, batch_size=32)
    variants = {
        "scratch": (lambda: scratch_resnet18(5), 1e-3, 40),
        "pretrained_head_only": (lambda: pretrained_resnet18(5, train_layer4=False), 1e-3, 20),
        "pretrained_layer4": (lambda: pretrained_resnet18(5, train_layer4=True), 3e-4, 20),
    }
    results, best = {}, (None, -1, None, None)
    for name, (build, lr, epochs) in variants.items():
        torch.manual_seed(0)
        model = build()
        hist = fit(model, data, dev, epochs=epochs, lr=lr, weight_decay=1e-4, patience=8, verbose=False)
        val_acc = max(hist.val_acc)
        results[name] = {
            "best_val_acc": val_acc,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "epochs_run": len(hist.val_acc),
        }
        plot_history(hist, f"ResNet-18 {name}", FIG / f"resnet_{name}.png")
        print(name, results[name])
        if name != "scratch" and val_acc > best[1]:
            best = (name, val_acc, model, hist)

    name, val_acc, model, hist = best
    y, pred = predict(model, data["test"], dev)
    results["final"] = {
        "variant": name,
        "params": n_params(model),
        "val_acc": val_acc,
        "test_acc": float((y == pred).mean()),
        "cpu_latency_ms": cpu_latency_ms(model, (3, SIZE, SIZE)),
    }
    print("final", results["final"])
    plot_confusion(y, pred, classes, f"ResNet-18 ({name}) - test", FIG / "cm_resnet.png")
    torch.save(model.state_dict(), REPORTS / "resnet18.pt")
    save_results("resnet", results, hist)


if __name__ == "__main__":
    main()
