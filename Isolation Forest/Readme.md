# Isolation Forest from Scratch

We've all learnt and implemented Random Forest; it's one of the first algorithms we encounter in ML. Most of the time, we strive to find patterns using such algorithms. Another paradigm is anomaly detection — and that’s how I came across Isolation Forests.

In this project, I implemented the Isolation Forest algorithm from scratch using NumPy. The model builds binary trees by recursively splitting randomly sampled data to isolate outliers quickly. I used synthetic 2D data (normal + uniform noise), calculated anomaly scores based on average path lengths, and visualized the decision boundaries.

## Technical Details
- Isolation Trees (iTrees) are constructed by recursively partitioning the data with random splits.
- The path length of a point is the number of edges traversed until the point is isolated.
- Anomaly score is computed using the average path length across all trees, normalized by the expected path length in a Binary Search Tree.
- Outliers tend to have shorter path lengths, hence higher anomaly scores.

## Reference
Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation forest." *2008 Eighth IEEE International Conference on Data Mining*. IEEE, 2008.  
[Link to paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
