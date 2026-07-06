import os
import matplotlib.pyplot as plt


def plot_comparison_metrics(history1, history2, model_name1, model_name2, save_dir=None):
    # If save_dir is given -> save the curves to file, otherwise show them on screen
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    metrics = [
        ("tr_iou", "val_iou", "Mean Intersection Over Union (mIoU) Learning Curve", "mIoU Score", "iou_curve.png"),
        ("tr_pa", "val_pa", "Pixel Accuracy (PA) Learning Curve", "PA Score", "pa_curve.png"),
        ("tr_loss", "val_loss", "Loss Learning Curve", "Loss Value", "loss_curve.png")
    ]

    for tr_metric, val_metric, title, ylabel, fname in metrics:
        plt.figure(figsize=(10, 5))
        plt.plot(history1[tr_metric], label=f"{model_name1} Train {ylabel.split(' ')[0]}")
        plt.plot(history1[val_metric], label=f"{model_name1} Validation {ylabel.split(' ')[0]}")
        plt.plot(history2[tr_metric], label=f"{model_name2} Train {ylabel.split(' ')[0]}", linestyle='--')
        plt.plot(history2[val_metric], label=f"{model_name2} Validation {ylabel.split(' ')[0]}", linestyle='--')
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)

        # If save_dir is set -> save to file, otherwise -> show on screen
        if save_dir:
            path = os.path.join(save_dir, fname)
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            print(f"Saqlandi -> {path}")
        else:
            plt.show()
