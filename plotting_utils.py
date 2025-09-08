import matplotlib.pyplot as plt
import numpy as np

class KFoldTrainingPlotter:
    """A class to handle plotting the results of K-Fold cross-validation."""
    def __init__(self, save_path="training_history_kfold.png"):
        """Initializes the plotter with a save path."""
        self.save_path = save_path

    def plot_and_save(self, histories):
        """
        Creates, saves, and closes the plot for k-fold histories.
        `histories` is a list of history dictionaries, one for each fold.
        """
        if not histories:
            print("⚠️ Warning: No history data to plot.")
            return

        # --- Prepare Data ---
        # Find the length of the longest training run for padding
        max_epochs = max(len(h['val_loss']) for h in histories)
        
        # Function to pad histories to the same length for averaging
        def get_padded_metric(metric_name):
            padded_metric = []
            for h in histories:
                metric = h[metric_name]
                padded = np.pad(metric, (0, max_epochs - len(metric)), 'edge') # Pad with the last value
                padded_metric.append(padded)
            return np.array(padded_metric)

        val_loss = get_padded_metric('val_loss')
        train_loss = get_padded_metric('train_loss')
        val_acc = get_padded_metric('val_acc')
        train_acc = get_padded_metric('train_acc')
        val_precision = get_padded_metric('val_precision')
        val_recall = get_padded_metric('val_recall')
        val_f1 = get_padded_metric('val_f1')

        # --- Create Plot ---
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        ax_loss, ax_acc, ax_metrics = axes[0], axes[1], axes[2]
        epochs = np.arange(max_epochs) + 1

        # --- Plot Helper ---
        def plot_metric(ax, data, label, color):
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            ax.plot(epochs, mean, color=color, label=f'Avg {label}')
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)

        # --- Plot Loss ---
        plot_metric(ax_loss, train_loss, 'Train Loss', 'C0')
        plot_metric(ax_loss, val_loss, 'Val Loss', 'C1')
        ax_loss.set_title("Loss vs. Epochs (Avg over K-Folds)")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid(True)

        # --- Plot Accuracy ---
        plot_metric(ax_acc, train_acc, 'Train Acc', 'C2')
        plot_metric(ax_acc, val_acc, 'Val Acc', 'C3')
        ax_acc.set_title("Accuracy vs. Epochs (Avg over K-Folds)")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        ax_acc.grid(True)

        # --- Plot Other Metrics ---
        plot_metric(ax_metrics, val_precision, 'Val Precision', 'C4')
        plot_metric(ax_metrics, val_recall, 'Val Recall', 'C5')
        plot_metric(ax_metrics, val_f1, 'Val F1', 'C6')
        ax_metrics.set_title("Validation Metrics vs. Epochs (Avg over K-Folds)")
        ax_metrics.set_xlabel("Epoch")
        ax_metrics.set_ylabel("Score")
        ax_metrics.legend()
        ax_metrics.grid(True)

        # --- Save and Finalize ---
        fig.tight_layout()
        try:
            fig.savefig(self.save_path)
            print(f"\n✅ K-Fold plot saved to {self.save_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save plot. Error: {e}")
        finally:
            plt.close(fig)
