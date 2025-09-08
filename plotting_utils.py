import matplotlib.pyplot as plt

class TrainingPlotter:
    """A class to handle plotting training and validation metrics at the end."""
    def __init__(self, save_path="training_history.png"):
        """Initializes the plotter with a save path."""
        self.save_path = save_path

    def plot_and_save(self, history):
        """Creates, saves, and closes the plot."""
        if not history["train_loss"]:
            print("⚠️ Warning: No history data to plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_loss = axes[0]
        ax_acc = axes[1]

        # --- Plot Loss ---
        ax_loss.plot(history["train_loss"], label="Train Loss")
        ax_loss.plot(history["val_loss"], label="Val Loss")
        ax_loss.set_title("Loss vs. Epochs")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid(True)

        # --- Plot Accuracy ---
        ax_acc.plot(history["val_acc"], label="Val Acc", color="green")
        ax_acc.set_title("Accuracy vs. Epochs")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        ax_acc.grid(True)

        # --- Save and Finalize ---
        fig.tight_layout()
        try:
            fig.savefig(self.save_path)
            print(f"✅ Plot saved to {self.save_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save plot. Error: {e}")
        finally:
            plt.close(fig) # Ensure the figure is closed
