import os
import pickle
import matplotlib.pyplot as plt

def load_histories(results_dir="results/records"):
    histories = {}
    for fname in os.listdir(results_dir):
        if fname.endswith(".pkl"):
            path = os.path.join(results_dir, fname)
            try:
                with open(path, "rb") as f:
                    history = pickle.load(f)
                key = fname.replace(".pkl", "")
                if isinstance(history, dict) and "train_acc" in history:
                    histories[key] = history
            except Exception as e:
                print(f"Failed to load {fname}: {e}")
    return histories

def plot_metric(histories, metric):
    plt.figure()
    has_data = False
    for name, hist in histories.items():
        if metric in hist and len(hist[metric]) > 0:
            plt.plot(hist[metric], label=name)
            has_data = True
    if has_data:
        plt.title(f"{metric.replace('_', ' ').title()} Comparison")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs("results/plots", exist_ok=True)
        plt.savefig(f"results/plots/{metric}_comparison.png")
        plt.show()
    else:
        print(f"No data found for metric: {metric}")

def print_summary_table(histories):
    print("\n=== 汇总表格 ===")
    print(f"{'配置':<35} | {'最终测试准确率':<18} | {'收敛轮数':<10} | {'最终训练损失'}")
    print("-" * 80)
    for name, hist in histories.items():
        acc = hist.get('test_acc', [None])[-1]
        loss = hist.get('train_loss', [None])[-1]
        conv_epoch = hist.get('convergence_epoch', 'N/A')
        if acc is not None and loss is not None:
            print(f"{name:<35} | {acc:<18.4f} | {conv_epoch:<10} | {loss:.4f}")
        else:
            print(f"{name:<35} | 数据不完整，跳过")

if __name__ == "__main__":
    histories = load_histories("results/records")
    if not histories:
        print("⚠️ 未加载到任何训练记录，请确认 'results/records/' 目录下有有效 .pkl 文件。")
    else:
        for metric in ['train_acc', 'test_acc', 'train_loss', 'test_loss']:
            plot_metric(histories, metric)
        print_summary_table(histories)