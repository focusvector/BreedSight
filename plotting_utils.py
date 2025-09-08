import matplotlib.pyplot as plt
import warnings

class LivePlot:
    """A class to handle live plotting of training and validation metrics."""
    def __init__(self, save_path="training_history.png"):
        """Initializes the plot."""
        # Suppress the UserWarning from Matplotlib about Tkinter not being in a main loop.
        warnings.filterwarnings("ignore", "Starting a Matplotlib GUI outside of the main thread")
        plt.ion() # Turn on interactive mode
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 5))
        self.ax_loss = self.axes[0]
        self.ax_acc = self.axes[1]
        self.save_path = save_path
        # Initial drawing of the empty plot
        self.fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def update(self, history):
        """Updates the plots with the latest history."""
        try:
            # Check if there is data to plot
            if not history["train_loss"]:
                return

            self.ax_loss.clear()
            self.ax_loss.plot(history["train_loss"], label="Train Loss")
            self.ax_loss.plot(history["val_loss"], label="Val Loss")
            self.ax_loss.set_title("Loss vs. Epochs")
            self.ax_loss.set_xlabel("Epoch")
            self.ax_loss.set_ylabel("Loss")
            self.ax_loss.legend()
            self.ax_loss.grid(True) # Add grid for better readability

            self.ax_acc.clear()
            self.ax_acc.plot(history["val_acc"], label="Val Acc", color="green")
            self.ax_acc.set_title("Accuracy vs. Epochs")
            self.ax_acc.set_xlabel("Epoch")
            self.ax_acc.set_ylabel("Accuracy")
            self.ax_acc.legend()
            self.ax_acc.grid(True) # Add grid for better readability

            self.fig.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.1) # A slightly longer pause can help with rendering
        except Exception as e:
            print(f"⚠️ Warning: Live plotting failed. Error: {e}")

    def save_and_close(self):
        """Saves the final plot to a file and closes the plot window."""
        try:
            self.fig.savefig(self.save_path)
            print(f"✅ Plot saved to {self.save_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save plot. Error: {e}")
        
        plt.ioff() # Turn off interactive mode
        plt.close(self.fig) # Close the figure window to allow the script to exit
