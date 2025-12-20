import matplotlib.pyplot as plt

hashann_9_recall = {1.01: 89, 5.1:150, 9.96: 135, 15.02: 128, 22.93: 220}

hashann_95_recall = {1.01: 63, 5.1: 95, 9.96: 102, 15.02: 101, 22.93: 135}

hashann_99_recall = {1.01: 23, 5.1: 18, 9.96: 33, 15.02: 30, 22.93: 10}

navix_9_recall = {5.1:18, 9.96: 16, 15.02: 24, 22.93: 30}

sel_list = [1.01, 5.1, 9.96, 15.02, 22.93]


def get_qps(recall_dict, sel_list):
    target_sel = []
    target_qps = []
    for sel in sel_list:
        if sel in recall_dict:
            target_sel.append(sel)
            target_qps.append(recall_dict[sel])
    return target_sel, target_qps

# plot
plt.figure(figsize=(8,6))

# plot sel / qps
hashann_sel_list, hashann_9_recall_list = get_qps(hashann_9_recall, sel_list)
navix_sel_list, navix_9_recall_list = get_qps(navix_9_recall, sel_list)

plt.plot(hashann_sel_list, hashann_9_recall_list, marker='o', label='HashANN', color='blue')
plt.plot(navix_sel_list, navix_9_recall_list, marker='o', label='Navix', color='orange')
plt.xlabel('Selectivity (%)')
plt.ylabel('QPS for 90% Recall @K=10')
# plt.title('EF Search Comparison for 90% Recall @K=10')
plt.xticks(sel_list)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('wiki_navix_9_recall_comparison.png')
print("Saved wiki_navix_9_recall_comparison.png")
plt.show()