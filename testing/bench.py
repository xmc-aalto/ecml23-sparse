import csv
import matplotlib.pyplot as plt

tags = ['CPU', 'GPU_InnerBatchNaive', 'GPU_InnerBatchVectorized', 'GPU_Fast', 'GPU_OuterBatchNaive', 'GPU_OuterBatchVectorized']


def load_results(kind: str):
    sizes = []
    results = {key: [] for key in tags}
    normalized = {key: [] for key in tags}
    for key in tags:
        with open(f'forward-{kind}{key}.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=";")
            for row in reader:
                size = float(row['batch'])
                elapsed = float(row["elapsed"]) * 1000 * 1000   # convert to µs
                results[key].append(elapsed)
                normalized[key].append(elapsed / size)
                if len(sizes) < len(results[key]):
                    sizes.append(size)
                else:
                    assert sizes[len(results[key]) - 1] == size

    return sizes, results, normalized


def plot_result(kind: str, normalize: bool, *, ylabel: str, xlabel: str):
    sizes, results, norm_results = load_results(kind)
    plot_data = norm_results if normalize else results

    for key in tags:
        plt.plot(sizes, plot_data[key], label=key)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim(ymin=0)
    plt.legend()
    plt.show()


plot_result("batch", True, ylabel="duration per instance / µs", xlabel="batch size")
plot_result("hidden", False, ylabel="duration / µs", xlabel="inputs")
plot_result("out", True, ylabel="duration per output / µs", xlabel="outputs")
plot_result("weights", True, ylabel="duration per weight / µs", xlabel="weights")
