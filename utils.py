import matplotlib.pyplot as plt

def plot_curves(results):
    for metric in ['train_acc', 'test_acc', 'train_loss', 'test_loss']:
        plt.figure()
        for name, hist in results.items():
            plt.plot(hist[metric], label=name)
        plt.title(metric.replace("_", " ").title())
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{metric}.png")

def print_summary_table(results):
    print("\n=== Summary Table ===")
    print(f"{'Optimizer':<20} | {'Final Test Acc':<15} | {'Conv Epoch':<12} | {'Final Train Loss'}")
    print("-"*65)
    for name, hist in results.items():
        final_test_acc = hist['test_acc'][-1]
        conv_epoch = hist.get("convergence_epoch", "N/A")
        final_train_loss = hist['train_loss'][-1]
        print(f"{name:<20} | {final_test_acc:<15.4f} | {conv_epoch:<12} | {final_train_loss:.4f}")