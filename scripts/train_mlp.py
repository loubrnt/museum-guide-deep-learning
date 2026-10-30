"""Feed-forward network: sanity check, learning-rate study, regularisation study, final model.

Each experiment writes its curves to reports/figures and its numbers to reports/mlp.json.
"""
import torch
from torch.utils.data import DataLoader, Subset

from museum_guide.data import class_names, device, load_metadata, loaders, split
from museum_guide.models import FeedForward
from museum_guide.train import (
    REPORTS,
    cpu_latency_ms,
    fit,
    n_params,
    plot_confusion,
    plot_history,
    predict,
    save_results,
)

SIZE = 64  # 64x64x3 = 12 288 inputs: an MLP has one weight per pixel and per hidden unit
HIDDEN = (512, 128)
FIG = REPORTS / "figures"


def new_model(dropout=0.0):
    torch.manual_seed(0)
    return FeedForward(3 * SIZE * SIZE, HIDDEN, 5, dropout)


def main():
    dev = device()
    df = load_metadata()
    parts, classes = split(df), class_names(df)
    plain = loaders(parts, SIZE, augment=False)
    results = {}

    # 1. Sanity check: the network must be able to memorise 64 images. If not, the code is broken.
    tiny = DataLoader(Subset(plain["train"].dataset, range(64)), batch_size=64, shuffle=True)
    hist = fit(new_model(), {"train": tiny, "val": tiny}, dev, epochs=100, lr=1e-3, verbose=False)
    results["sanity_overfit_64_images"] = {"final_train_acc": hist.train_acc[-1], "final_loss": hist.train_loss[-1]}
    plot_history(hist, "Sanity check: memorising 64 images", FIG / "mlp_sanity.png")
    print("sanity", results["sanity_overfit_64_images"])

    # 2. Learning rate: too high diverges or oscillates, too low barely moves.
    results["lr_study"] = {}
    for lr in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        hist = fit(new_model(), plain, dev, epochs=25, lr=lr, verbose=False)
        results["lr_study"][str(lr)] = {"best_val_acc": max(hist.val_acc), "final_train_loss": hist.train_loss[-1]}
        plot_history(hist, f"MLP, lr={lr}", FIG / f"mlp_lr_{lr}.png")
        print("lr", lr, results["lr_study"][str(lr)])

    # 3. Regularisation: without it the train/val gap explodes (overfitting).
    augmented = loaders(parts, SIZE, augment=True)
    configs = {
        "none": dict(dropout=0.0, wd=0.0, data=plain),
        "dropout_wd": dict(dropout=0.5, wd=1e-4, data=plain),
        "dropout_wd_augment": dict(dropout=0.5, wd=1e-4, data=augmented),
    }
    results["regularisation_study"] = {}
    best_name, best_model, best_acc, best_hist = None, None, -1, None
    for name, c in configs.items():
        model = new_model(c["dropout"])
        hist = fit(model, c["data"], dev, epochs=60, lr=1e-4, weight_decay=c["wd"], verbose=False)
        i = hist.val_acc.index(max(hist.val_acc))
        results["regularisation_study"][name] = {
            "best_val_acc": hist.val_acc[i],
            "train_acc_at_best": hist.train_acc[i],
            "best_epoch": i + 1,
        }
        plot_history(hist, f"MLP, regularisation: {name}", FIG / f"mlp_reg_{name}.png")
        print("reg", name, results["regularisation_study"][name])
        if hist.val_acc[i] > best_acc:
            best_name, best_model, best_acc, best_hist = name, model, hist.val_acc[i], hist

    # 4. Final model on the test set (artists never seen during training or model selection).
    y, pred = predict(best_model, plain["test"], dev)
    results["final"] = {
        "config": best_name,
        "params": n_params(best_model),
        "val_acc": best_acc,
        "test_acc": float((y == pred).mean()),
        "cpu_latency_ms": cpu_latency_ms(best_model, (3, SIZE, SIZE)),
    }
    print("final", results["final"])
    plot_confusion(y, pred, classes, "MLP - test", FIG / "cm_mlp.png")

    # 5. Failure diagnosis: an MLP ties each weight to one pixel position, so a small shift should hurt it.
    shifted = []
    for x, _ in plain["test"]:
        shifted.append(torch.roll(x, shifts=(8, 8), dims=(2, 3)))
    xs = torch.cat(shifted)
    with torch.no_grad():
        best_model.eval().to(dev)
        pred_shift = best_model(xs.to(dev)).argmax(1).cpu().numpy()
    results["final"]["test_acc_shifted_8px"] = float((y == pred_shift).mean())
    results["final"]["prediction_change_rate_shifted"] = float((pred != pred_shift).mean())
    print("shift", results["final"]["test_acc_shifted_8px"], results["final"]["prediction_change_rate_shifted"])

    torch.save(best_model.state_dict(), REPORTS / "mlp.pt")
    save_results("mlp", results, best_hist)


if __name__ == "__main__":
    main()
