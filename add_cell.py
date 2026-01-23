import json

with open('fiber_processing_EPM.ipynb', 'r') as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "# Collect mean traces for open_entry\n",
        "mean_traces = []\n",
        "colors = plt.cm.tab10(np.linspace(0, 1, len(epm_psth_results)))\n",
        "for i, (animal, events) in enumerate(epm_psth_results.items()):\n",
        "    if 'open_entry' in events and events['open_entry']['n_trials'] > 0:\n",
        "        mean_trace = events['open_entry']['mean']\n",
        "        mean_traces.append(mean_trace)\n",
        "        plt.plot(events['open_entry']['tvec'], mean_trace, color=colors[i], label=animal)\n",
        "# Compute overall mean\n",
        "if mean_traces:\n",
        "    overall_mean = np.nanmean(np.array(mean_traces), axis=0)\n",
        "    tvec = list(epm_psth_results.values())[0]['open_entry']['tvec']  # assume same\n",
        "    plt.plot(tvec, overall_mean, color='black', linewidth=3, label='Mean')\n",
        "plt.xlabel('Time from event (s)')\n",
        "plt.ylabel('Z')\n",
        "plt.title('Mean signal and individual traces for open_entry')\n",
        "plt.legend()\n",
        "plt.show()\n"
    ]
}

nb['cells'].append(new_cell)

with open('fiber_processing_EPM.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
