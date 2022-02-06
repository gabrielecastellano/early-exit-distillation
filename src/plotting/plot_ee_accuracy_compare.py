import json
import os
import glob

import matplotlib.pyplot as plt


# Name of the bottlenecked model
model = "cifar100-resnet50_mimic"
# Variants of the bottleneck to plot
bn_labels = ["ver7b-3ch", "ver10v-3ch"]
# Algorithms to consider
algorithms = [
    "gmm_layer",
    "linear",
    "faiss_kmeans",
    "faiss_knn"
]
# Joint/disjoint training, joint/disjoint evaluation
modes = {
    "solo_train-solo_eval": "solo train, solo eval",
    "joint_train-solo_eval": "joint train, solo eval",
    # "joint_train-joint_eval": "joint train, joint eval",
    # "solo_train-joint_eval": "solo_train, joint eval",
    }
# For each type of mode, a list of "key parameters" (e.g., number of components for kmeans) to plot
keys = {"faiss_kmeans": [8],
        "sdgm": [3],
        "gmm_layer": [8],
        "linear": [1],
        "faiss_knn": [20]}
instance_key_strings = {"sdgm": "components", "linear": "", "gmm_layer": "components", "faiss_kmeans": "components",
                        "faiss_knn": "k"}
# a:b plots data for a classes, b fraction of the total samples
subset_key = "5:1.0"

stats_files_s = list()

fig, ax = plt.subplots(1, 1)
ax.set_ylabel('Accuracy')
ax.set_xlabel('Coverage')

accuracy = dict()
coverage = dict()

# for instance in instances:  # datetime
instances = []
bn_labels_ = set()
algorithms_ = set()
modes_ = set()
for bn_label in bn_labels:  # bn version
    experiment_name = f"{model}-{bn_label}"
    for algorithm in algorithms:  # ee type
        for mode in modes:
            instance_key_string = instance_key_strings[algorithm]
            files = glob.glob(f"ee_stats/{algorithm}/{mode}/{experiment_name}-{algorithm}_*.json")
            if len(files) == 0:
                continue
            stats_file_s = sorted(files)[-1]
            instance = stats_file_s[-20:-5]
            ee_stats = None
            try:
                with open(stats_file_s, mode='r') as stats_file:
                    ee_stats = json.load(stats_file)
            except FileNotFoundError:
                continue
            if subset_key not in ee_stats:
                continue
            instances.append(instance)
            bn_labels_.add(bn_label)
            algorithms_.add(algorithm)
            modes_.add(mode)
            mode = modes[mode]
            for key_param in ee_stats[subset_key]:
                if int(float(key_param)) in keys[algorithm]:
                    model_label = f"{bn_label}, {algorithm}, {mode}" + (f", {instance_key_strings[algorithm]}={key_param}" if instance_key_strings[algorithm] else "")
                    if model_label not in accuracy:
                        accuracy[model_label] = list()
                        coverage[model_label] = list()
                    for t in ee_stats[subset_key][key_param]:
                        accuracy[model_label].append(ee_stats[subset_key][key_param][t]['confident_accuracy'])
                        coverage[model_label].append(ee_stats[subset_key][key_param][t]['coverage'])


for i, model_label in enumerate(accuracy.keys()):
    ax.grid(axis='y')
    # plt.ylim((0, 100))
    ax.plot(coverage[model_label], accuracy[model_label], label=model_label)
    # ax.legend(fontsize=8)

# ax.plot([0, 1], [76.4, 76.4], label="base model", color="gray", linestyle="dashed")
ax.legend(fontsize=8)
ax.grid(axis='y')

algorithms_ = sorted(algorithms_)
modes_ = sorted(modes)
bn_labels_ = sorted(bn_labels_)
if not os.path.exists(f"plots/{model}"):
    os.mkdir(f"plots/{model}")
if not os.path.exists(f"plots/{model}/{'-'.join(algorithms_)}"):
    os.mkdir(f"plots/{model}/{'-'.join(algorithms_)}")

fig_name = f"plots/{model}/{'-'.join(algorithms_)}/{'-'.join(bn_labels_)}_{'--'.join(modes_)}accuracy_coverage_compare_{subset_key}classes.png"
fig.savefig(fig_name)
print(f"Used instances: {instances}")
print(f"Saved '{fig_name}'")
plt.close()
