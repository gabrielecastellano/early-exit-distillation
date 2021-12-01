import json
import os

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import numpy as np

# '''
instances = ["20211201-201512", "20211126-160300", "20211202-194143", "20211202-180035"]
experiment_names = ["cifar100-resnet50_mimic-ver6b-6ch"]
algorithms = ["sdgm", "linear"]
modes = {"solo_train-solo_eval": "solo train, solo eval", "joint_train-joint_eval": "joint train, joint eval", "solo_train-joint_eval": "solo_train, joint eval", "joint_train-solo_eval": "joint_train, solo eval"}

instance_key_strings = {"sdgm": "components", "linear": ""}
bn_labels = {"cifar100-resnet50_mimic-ver6b-6ch": "ver6b-6ch"}

subset_key = "100:1.0"

stats_files_s = list()

fig, ax = plt.subplots(1, 1)
ax.set_ylabel('Accuracy')
ax.set_xlabel('Coverage')

accuracy = dict()
coverage = dict()

for instance in instances:  # datetime
    for experiment_name in experiment_names:  # bn version
        for algorithm in algorithms:  # ee type
            for mode in modes:
                instance_key_string = instance_key_strings[algorithm]
                stats_file_s = f"ee_stats/{algorithm}/{mode}/{experiment_name}-{algorithm}_{instance}.json"
                mode = modes[mode]
                ee_stats = None
                try:
                    with open(stats_file_s, mode='r') as stats_file:
                        ee_stats = json.load(stats_file)
                except FileNotFoundError:
                    continue
                for key_param in ee_stats[subset_key]:
                    model_label = f"{bn_labels[experiment_name]}, {algorithm}, {mode}" + (f", {instance_key_strings[algorithm]}={key_param}" if instance_key_strings[algorithm] else "")
                    if model_label not in accuracy:
                        accuracy[model_label] = list()
                        coverage[model_label] = list()
                    for t in ee_stats[subset_key][key_param]:
                        accuracy[model_label].append(ee_stats[subset_key][key_param][t]['confident_accuracy'])
                        coverage[model_label].append(ee_stats[subset_key][key_param][t]['coverage'])

for i, model_label in enumerate(accuracy.keys()):
    ax.scatter(coverage[model_label], accuracy[model_label], marker=MarkerStyle.filled_markers[i], label=model_label)
    ax.legend()

if not os.path.exists(f"plots/{'-'.join(algorithms)}"):
    os.mkdir(f"plots/{'-'.join(algorithms)}")
if not os.path.exists(f"plots/{'-'.join(algorithms)}/{'-'.join(instances)}"):
    os.mkdir(f"plots/{'-'.join(algorithms)}/{'-'.join(instances)}")

fig.savefig(f"plots/{'-'.join(algorithms)}/{'-'.join(instances)}/{'-'.join(bn_labels.keys())}_accuracy_coverage_scatter.png")
plt.close()
