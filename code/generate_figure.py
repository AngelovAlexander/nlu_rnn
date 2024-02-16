import matplotlib.pyplot as plt

# Given arrays
#a = [74.1, 74.2, 74.3, 74.3, 74.3]
#b = [74.5, 73.6, 73.2, 72.6, 71.7]
#c = [78.6, 78.4, 78.9, 78.6, 78.6]
#d = [77.1, 77.4, 75.3, 75.2, 74.8]

a = [78.1, 78.4, 78.8, 78.4, 78.2]
b = [75.4, 74.3, 71.9, 70.7, 70.8]
c = [80.0, 79.6, 79.8, 79.4, 79.6]
d = [78.9, 79.5, 78.8, 80.9, 79.5]

# Indices for the x-axis
indices = range(1, len(a) + 1)  # Start from 1 to match your example

# Create the plot
plt.figure(figsize=(8, 5))
for i, (a_val, b_val) in enumerate(zip(a, b), start=1):
    # Plot the points for each array
    plt.plot([i, i], [a_val, b_val], marker='o', linestyle='--', color='blue', label='RNN accuracy gap' if i == 1 else "")
    # Plot points from array a
    plt.plot(i, a_val, marker='o', color='pink', label='RNN performance on sentences with distance <= D' if i == 1 else "")
    # Plot points from array b
    plt.plot(i, b_val, marker='o', color='brown', label='RNN performance on sentences with distance > D' if i == 1 else "")

for i, (c_val, d_val) in enumerate(zip(c, d), start=1):
    
    # Connect points with a line
    plt.plot([i, i], [c_val, d_val], linestyle='--', color='green', label='GRU accuracy gap' if i == 1 else "")
    # Plot points from array a
    plt.plot(i, c_val, marker='o', color='red', label='GRU performance on sentences with distance <= D' if i == 1 else "")
    # Plot points from array b
    plt.plot(i, d_val, marker='o', color='orange', label='GRU performance on sentences with distance > D' if i == 1 else "")

# Customize the plot
#plt.title('Visualization of the gap between the performance based on the distance between the subject and the verb with 0 BPTT steps')
plt.xlabel('Distance separation (D)')
plt.ylabel('Accuracy')
plt.legend(bbox_to_anchor=(1.04, 0.4))
plt.xticks(indices)  # Set x-axis ticks to match the indices
plt.grid(True, which='both', linestyle='--', linewidth=0.2)

plt.savefig("output.png", bbox_inches="tight")
# Show the plot
plt.show()