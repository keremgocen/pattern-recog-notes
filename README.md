# k-means

```
# STEPS:
# 0 - Choose k random centroid coordinates
# For each pixel x:
# 1 - Calculate rgb distance of x from each centroid
# 2 - Assign x to closest centroid's cluster
# For each centroid c:
# 3 - Calculate rgb means of assigned pixels for that c (move the centroid)
# repeat until all x are stable,
```

## output
> labels:4
> mean_diff_threshold:5.0

![example](/kmeans/label-4-output.png)
