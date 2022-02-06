import json
import os

import matplotlib.pyplot as plt
import numpy as np

# '''
instance = "20211210-023629"
experiment_name = "cifar100-resnet50_mimic-ver6b-6ch"
algorithm = "knn"
x_label = "threshold"
instance_key_str = "k"
stats_file_s = f"ee_stats/{algorithm}/solo_train-solo_eval/{experiment_name}-{algorithm}_{instance}.json"
'''
instance = "20211130-114937"
experiment_name = "cifar100-resnet50_mimic-ver6b-6ch"
algorithm = "faiss_kmeans"
x_label = "threshold"
instance_key_str = "k"
stats_file_s = f"ee_stats/{algorithm}/solo_train-solo_eval/{experiment_name}-{algorithm}_{instance}.json"
'''

with open(stats_file_s, mode='r') as stats_file:
    ee_stats = json.load(stats_file)
    fig = dict()
    ax1 = dict()
    ax2 = dict()

    for subset_key in ee_stats:
        classes, fraction_samples_per_class = int(subset_key.split(':')[0]), float(subset_key.split(':')[1])

        for key_param in ee_stats[subset_key]:
            accuracy = []
            o_accuracy = []
            predicted = []
            ks = []
            if key_param not in fig.keys():
                fig[key_param], (ax1[key_param], ax2[key_param]) = plt.subplots(2, 1)
                ax1[key_param].set_ylabel('Accuracy')
                ax2[key_param].set_ylabel('Coverage')
                ax2[key_param].set_xlabel(x_label)
            for x in ee_stats[subset_key][key_param]:
                o_accuracy.append(ee_stats[subset_key][key_param][x]['overall_accuracy'])
                accuracy.append(ee_stats[subset_key][key_param][x]['confident_accuracy'])
                predicted.append(ee_stats[subset_key][key_param][x]['coverage'])
                ks.append(float(x))
            ax1[key_param].plot(ks, accuracy, 'o-', label=f"{classes} classes")
            ax2[key_param].plot(ks, predicted, '.-', label=f"{classes} classes")
            ax1[key_param].legend()
            ax2[key_param].legend()

        #ax2.set_xticks(range(1, len(kmeans_stats[key].keys()) + 1), kmeans_stats[key].keys())
    if not os.path.exists(f"plots/{algorithm}"):
        os.mkdir(f"plots/{algorithm}")
    if not os.path.exists(f"plots/{algorithm}/{instance}"):
        os.mkdir(f"plots/{algorithm}/{instance}")
    for key_param, f in fig.items():
        f.savefig(f"plots/{algorithm}/{instance}/{experiment_name}_accuracy_{key_param}{instance_key_str}.png")
    plt.close()
