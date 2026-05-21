import matplotlib.pyplot as plt

# 数据
reward_diffs = [
    -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0,
    -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
    4.5, 5.0, 5.5
]
counts = [
    175, 483, 213, 625, 91, 613, 140, 770, 370, 1991,
    590, 13222, 701, 1771, 487, 727, 110, 1195, 234, 1115,
    914, 746, 630
]

plt.hist(reward_diffs, weights=counts, bins=22, edgecolor='black')
plt.title('Reward Difference Distribution')
plt.xlabel('Reward Difference (p_vote - n_vote)')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.75)
plt.savefig('reward_diff_distribution.png')
plt.show()