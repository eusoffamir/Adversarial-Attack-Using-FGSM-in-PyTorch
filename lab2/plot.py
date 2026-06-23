import matplotlib.pyplot as plt

# Data from your execution
categories = ['Baseline (Clean)', 'Poisoned (20% Attack)', 'After Defense (Isolation Forest)']
accuracies = [98.25, 94.74, 95.32]
colors = ['#2ca02c', '#d62728', '#1f77b4'] # Green, Red, Blue

plt.figure(figsize=(8, 5))
bars = plt.bar(categories, accuracies, color=colors, width=0.5)

# Formatting the chart
plt.ylim(90, 100) # Zoom in to see the distinct differences clearly
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Model Performance Under Data Poisoning Attack & Defense', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value tags on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval}%", ha='center', va='bottom', fontweight='bold')

# Save the plot to embed in your report
plt.tight_layout()
plt.savefig('poisoning_defense_chart.png', dpi=300)
print("Graph saved successfully as 'poisoning_defense_chart.png'!")
plt.show()