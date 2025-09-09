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
        Produces two files:
         - self.save_path: average +/- std for key metrics
         - self.save_path with suffix _folds.png: individual fold curves
        """
        if not histories:
            print("⚠️ Warning: No history data to plot.")
            return

        # --- Prepare Data ---
        # Find the length of the longest training run for padding
        max_epochs = max(len(h.get('val_loss', [])) for h in histories)
        if max_epochs == 0:
            print("⚠️ Warning: Histories contain no epochs.")
            return

        # Function to pad histories to the same length for averaging
        def get_padded_metric(metric_name):
            padded_metric = []
            for h in histories:
                metric = h.get(metric_name, [])
                if len(metric) == 0:
                    padded = np.zeros(max_epochs)
                else:
                    padded = np.pad(metric, (0, max_epochs - len(metric)), 'edge')
                padded_metric.append(padded)
            return np.array(padded_metric)

        # Metrics to consider
        train_loss = get_padded_metric('train_loss')
        val_loss = get_padded_metric('val_loss')
        train_acc = get_padded_metric('train_acc')
        val_acc = get_padded_metric('val_acc')
        val_precision = get_padded_metric('val_precision')
        val_recall = get_padded_metric('val_recall')
        val_f1 = get_padded_metric('val_f1')

        epochs = np.arange(max_epochs) + 1

        # --- Create Average + Std Plot ---
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        ax_loss, ax_acc, ax_metrics = axes[0], axes[1], axes[2]

        def plot_metric(ax, data, label, color):
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            ax.plot(epochs, mean, color=color, label=f'Avg {label}', linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)

        # Loss
        plot_metric(ax_loss, train_loss, 'Train Loss', 'C0')
        plot_metric(ax_loss, val_loss, 'Val Loss', 'C1')
        ax_loss.set_title("Loss vs. Epochs (Avg over K-Folds)")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid(True)

        # Accuracy
        plot_metric(ax_acc, train_acc, 'Train Acc', 'C2')
        plot_metric(ax_acc, val_acc, 'Val Acc', 'C3')
        ax_acc.set_title("Accuracy vs. Epochs (Avg over K-Folds)")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        ax_acc.grid(True)

        # Precision/Recall/F1
        plot_metric(ax_metrics, val_precision, 'Val Precision', 'C4')
        plot_metric(ax_metrics, val_recall, 'Val Recall', 'C5')
        plot_metric(ax_metrics, val_f1, 'Val F1', 'C6')
        ax_metrics.set_title("Validation Metrics vs. Epochs (Avg over K-Folds)")
        ax_metrics.set_xlabel("Epoch")
        ax_metrics.set_ylabel("Score")
        ax_metrics.legend()
        ax_metrics.grid(True)

        fig.tight_layout()
        try:
            fig.savefig(self.save_path)
            print(f"✅ K-Fold average plot saved to {self.save_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save average plot. Error: {e}")
        finally:
            plt.close(fig)

        # --- Create Individual Fold Curves ---
        per_fold_path = self.save_path.rsplit('.', 1)[0] + '_folds.png'
        fig2, axes2 = plt.subplots(2, 2, figsize=(18, 10))
        ax_vloss = axes2[0, 0]
        ax_vacc = axes2[0, 1]
        ax_vprec = axes2[1, 0]
        ax_vf1 = axes2[1, 1]

        # Helper to plot each fold thin lines and average bold
        def plot_per_fold(ax, data, label, cmap='tab10'):
            n_folds = data.shape[0]
            colors = plt.get_cmap(cmap)(np.linspace(0, 1, n_folds))
            for i in range(n_folds):
                ax.plot(epochs, data[i], color=colors[i], alpha=0.4, linewidth=1, label=f'Fold {i+1}' if i==0 else None)
            mean = np.mean(data, axis=0)
            ax.plot(epochs, mean, color='k', linewidth=2.5, label='Avg')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(label)
            ax.legend()
            ax.grid(True)

        plot_per_fold(ax_vloss, val_loss, 'Val Loss')
        ax_vloss.set_title('Validation Loss per Fold')

        plot_per_fold(ax_vacc, val_acc, 'Val Accuracy')
        ax_vacc.set_title('Validation Accuracy per Fold')

        plot_per_fold(ax_vprec, val_precision, 'Val Precision')
        ax_vprec.set_title('Validation Precision per Fold')

        plot_per_fold(ax_vf1, val_f1, 'Val F1')
        ax_vf1.set_title('Validation F1 per Fold')

        fig2.tight_layout()
        try:
            fig2.savefig(per_fold_path)
            print(f"✅ Per-fold plots saved to {per_fold_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save per-fold plot. Error: {e}")
        finally:
            plt.close(fig2)

    def save_confusion_matrix(self, cm, class_names, save_path, normalize=True, cmap='Blues'):
        """Saves a confusion matrix heatmap to disk.
        cm: 2D array (num_classes x num_classes)
        class_names: list of class names in label-index order
        """
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            mat = cm.astype(float)
            if normalize:
                row_sums = mat.sum(axis=1, keepdims=True)
                # avoid division by zero
                row_sums[row_sums == 0] = 1
                mat = mat / row_sums

            im = ax.imshow(mat, interpolation='nearest', cmap=cmap)
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Normalized count' if normalize else 'Count', rotation=-90, va="bottom")

            # Show all ticks and label them with the respective list entries
            num_classes = len(class_names)
            ax.set(xticks=np.arange(num_classes), yticks=np.arange(num_classes),
                   xticklabels=class_names, yticklabels=class_names,
                   ylabel='True label', xlabel='Predicted label')

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            fmt = '.2f' if normalize else 'd'
            thresh = mat.max() / 2.
            for i in range(num_classes):
                for j in range(num_classes):
                    val = mat[i, j]
                    ax.text(j, i, format(val, fmt), ha="center", va="center",
                            color="white" if val > thresh else "black", fontsize=8)

            fig.tight_layout()
            fig.savefig(save_path)
            plt.close(fig)
            print(f"✅ Confusion matrix saved to {save_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save confusion matrix. Error: {e}")
