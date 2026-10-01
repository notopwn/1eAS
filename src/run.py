"""Run the whole thing: load, split, train, measure, draw, save.

This file is complete. It calls the functions you wrote. If it crashes, the
problem is almost always in one of your functions, not here.

Usage:
    python src/run.py                          the defaults: ImageNet weights,
                                               every layer trained, 10 epochs
    python src/run.py --scratch --tag scratch  random weights instead
    python src/run.py --freeze --lr 1e-3 --tag frozen
                                               train only the last layer
    python src/run.py --epochs 20 --lr 3e-4    train longer, or faster

Every run writes its pictures and its model into results/, named after --tag,
and prints one line at the end that you can paste straight into the
experiment table in your report. Everything it prints is also saved in
results/<tag>_output.txt, for the long report.

It does not touch your web page. When a run gives you the model you want to
publish, export it:
    python src/export_web.py --tag <that tag>
"""

import argparse
import json
import sys
import time
from pathlib import Path

from data import load_folder, split_train_test
from evaluate import (accuracy, confusion_matrix, plot_confusion, plot_history,
                      plot_worst, worst_examples)
from train import build_model, count_trainable, save, train

ROOT = Path(__file__).resolve().parents[1]


class Tee:
    """Print to the screen and into a file at the same time."""

    def __init__(self, screen, path):
        self.screen = screen
        self.file = open(path, "w", encoding="utf-8")

    def write(self, text):
        self.screen.write(text)
        self.file.write(text)

    def flush(self):
        self.screen.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.screen, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "clean"))
    parser.add_argument("--scratch", action="store_true",
                        help="start from random weights instead of ImageNet")
    parser.add_argument("--freeze", action="store_true",
                        help="train only the new last layer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="run", help="name for the output files")
    args = parser.parse_args()

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    sys.stdout = Tee(sys.stdout, out_dir / f"{args.tag}_output.txt")

    print(f"reading {args.data}")
    started = time.time()
    X, y, class_names, paths = load_folder(args.data)
    print(f"{len(X)} images, {len(class_names)} classes, "
          f"read in {time.time() - started:.0f} s")
    for index, name in enumerate(class_names):
        print(f"  {name:24s} {int((y == index).sum())}")

    Xtr, ytr, ptr, Xte, yte, pte = split_train_test(
        X, y, paths, test_ratio=args.test_ratio, seed=args.seed
    )
    print(f"training on {len(Xtr)}, testing on {len(Xte)}")

    model = build_model(len(class_names), pretrained=not args.scratch,
                        freeze=args.freeze)
    n_parameters = sum(p.numel() for p in model.parameters())
    n_trainable = count_trainable(model)
    print(f"model has {n_parameters:,} parameters, "
          f"{n_trainable:,} of them will be trained")

    started = time.time()
    history = train(
        model, Xtr, ytr, Xte, yte,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )
    minutes = (time.time() - started) / 60

    train_accuracy = accuracy(model, Xtr, ytr)
    test_accuracy = accuracy(model, Xte, yte)
    print(f"\ntrained in {minutes:.1f} minutes")
    print(f"train accuracy {train_accuracy:.1%}")
    print(f"test  accuracy {test_accuracy:.1%}")

    plot_history(history, out_dir / f"{args.tag}_curves.png")
    plot_confusion(
        confusion_matrix(model, Xte, yte, len(class_names)),
        class_names, out_dir / f"{args.tag}_confusion.png",
    )
    try:
        mistakes = worst_examples(model, Xte, yte, pte, k=10)
    except NotImplementedError:
        print(f"worst_examples is not written yet (Problem 5), so there is no "
              f"{args.tag}_worst.png this time")
    else:
        plot_worst(mistakes, class_names, out_dir / f"{args.tag}_worst.png")

    save(model, out_dir / f"{args.tag}_model.pt")
    print(f"wrote {out_dir / (args.tag + '_model.pt')}")

    start = "random" if args.scratch else "ImageNet"
    trained = "last layer" if args.freeze else "all layers"
    settings = {
        "tag": args.tag,
        "start": start,
        "trained": trained,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "class_names": class_names,
        "n_images": int(len(X)),
        "n_parameters": int(n_parameters),
        "n_trainable": int(n_trainable),
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "minutes": round(minutes, 2),
        "test_paths": [str(Path(p).resolve()) for p in pte],
    }
    (out_dir / f"{args.tag}_settings.json").write_text(json.dumps(settings, indent=2))

    print("\nOne line for your experiment table:")
    print(
        f"| {args.tag} | {start} | {trained} | {args.epochs} | {args.lr:g} | "
        f"{n_trainable:,} | {train_accuracy:.1%} | {test_accuracy:.1%} |"
    )
    print(f"\nTo put this model on your web page: "
          f"python src/export_web.py --tag {args.tag}")


if __name__ == "__main__":
    main()
