---
layout: default
title:  "Making Graphs With Locality-Sensitive Hashing"
date:   2020-11-29 12:02:30 -0500
categories: algorithms
---

## Introduction
You're probably familiar with hashing as an algorithmic technique. It's used in some of the most common data structures like [hash tables](https://en.wikipedia.org/wiki/Hash_table) and in security for keeping passwords from prying eyes. The key attribute for our purposes is that two similar values should result in two random, unrelated hashes. [Locality-sensitive hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) (LSH) on the other hand tries to hash our points into the same buckets if the points themselves are similar. This is useful for a number of things like:

- **Dimensionality Reduction** - If we have many high-dimensional vectors and we want to group them based on their similarity, LSH allows us to bucket them as coarsely or finely as we like.
- **Nearest-Neighbor Search** - We can hash a value we want to compare quickly and then we only need to compare to all the points within that hash bucket.
- **Similarity Graph** - We can use the hash values to form a graph of all points weighted by their similarity and do some fun things with that. 

There are a few different ways to implement the algorithm, but I'm going to use the random hyperplane approach since that's what I first heard about and is the method that made me want to try my hand at implementing it from scratch. If you want an *excellent* introduction to this, I highly reccommend [this lecture](https://www.youtube.com/watch?v=Arni-zkqMBA).

## The Idea
The very high level summary of how to generate the hash code for each point is:
1. Generate a bunch of hyperplanes. 
2. If a given point is on the left of the \\(n^{th}\\) plane, its \\(n^{th}\\) hash bit is 1, else 0. 

This is very easy to do since determining which side of a hyperplane a point is on is as simple as evaluating

\\[sgn(a \cdot b)\\]

where \\(a\\) is our point and \\(b\\) is a normal vector of the hyperplane. So in code, that's

{% highlight python %}
def get_code_bit(data: numpy.ndarray) -> numpy.ndarray:
    norm_vec = 2 * numpy.random.random_sample((1, data.shape[1])).T - 1
    norm_vec = norm_vec / numpy.linalg.norm(norm_vec)
    return numpy.sign(data.dot(norm_vec)).reshape(data.shape[0])
{% endhighlight %}

Here we're normalizing the vector too, but since we don't care about anything related to the plane except for the normal vector, we can just generate a bunch of vectors and pretend they're normal vectors to planes Without Loss of Generality™. 

This simple procedure can reduce the dimensionality of our data a ton depending on how many bits we choose for the hash. But it also preserves a surprising amount of information about the original geometry. For example, consider the following paritioning:

![Initial Partitioning](/assets/img/hash1.png)

Here I've labeled the plane responsible for the nth bit in red. So here, the blue points will be hashed to 100, the brown to 111, and the green to 000. So the algorithm is claiming that all points with the same color are similar. But it also tells us that hashes differing by only 1 bit are in "touching buckets". You can verify this yourself, but blue and green are an example. That provides a decent amount of information on how dissimilar the sets of points in different buckets are. So we get a secondary measure out of this. Not only are all points in the same bucket, most similar, but other points can be directly compared by the [Hamming Distance](https://en.wikipedia.org/wiki/Hamming_distance) of their hashes. Neat! What else can we get out of this? Well let's repartition the points and see what happens.

![Second Partitioning](/assets/img/hash2.png)

Our buckets changed a bit and as a result, we're classifying different points as being similar. Hmmm... Perhaps we should do this a bunch of times and keep track of which points are in which buckets. We could then make the most frequent bucket for each point the "true" bucket which may give us better hashing results. But let's go one step further and create a similarity graph based on which points are in which bucket after each iteration. 

## Making the Graph
Our method will be:
1. Initiate an adjency matrix to 0 for every point in our data set
2. Get hashes for every point
3. If 2 points have the same hash, connect those points (add 1 to corresponding entry in matrix)

At the end, we'll get an adjacency matrix where each entry is weighted by how often those 2 points appeared in the same bucket together. The code again is pretty simple. Here's how I did it:

{% highlight python %}
#TODO: change string join to just tuple of row
def get_codes(data: numpy.ndarray, iterations: int) -> dict:
    codes = numpy.zeros((data.shape[0], iterations))
    iter_count = 0
    while iter_count < iterations:
        bit = get_code_bit(data)
        codes[:, iter_count] = bit
        iter_count += 1

    code_row_map = defaultdict(list)
    for index, row in enumerate(codes):
        code_row_map[''.join(map(str,row))].append(index)
    return code_row_map


def update_matrix(matrix: numpy.ndarray, code_map: dict) -> numpy.ndarray:
    for index_list in code_map.values():
        for index1 in index_list:
            for index2 in index_list:
                matrix[index1, index2] += 1
    return matrix


def weighted_hash_adj_mat(data: numpy.ndarray, iterations: int, prune_cutoff: int) -> numpy.ndarray:
    adj_matrix = numpy.zeros((data.shape[0], data.shape[0]))
    iteration = 0
    while iteration < iterations:
        code_map = get_codes(data, int(numpy.log2(data.shape[0])))
        adj_matrix = update_matrix(adj_matrix, code_map)
        iteration += 1
    
    return adj_matrix * (adj_matrix > prune_cutoff)
{% endhighlight %}

The efficiency-minded among you may have a concern here. Namely that this needs \\(O(n^2)\\) space and time. And you'd be right. It definitely defeats the purpose of a super efficient algorithm, but we're exploring ideas and generalizations here so we're not concerned with that right now. Let's see what this spits out on some fake data:

![Unweighted Graph](/assets/img/g1.png)

The similarity connections seem pretty reasonable to me. Let's see what happens if we weight the lines by the weight in the matrix and play with the number of hashes.

![Weighted Graph 1](/assets/img/g2.png)

Here we have unweighted on top and weghted on the bottom. If you look really hard you can see some points that are very close have much thicker lines. I also made the hashes smaller so that more points get clustered into each group

![Weighted Graph 2](/assets/img/g3.png)

Here is another but with different numbers of iterations. You can play with all the parameters to get the effect you want. Once we have this graph, there's a rich repository of techniques we can use to analyze this farther, depending on what we're looking for. 

## Conclusion
So now what? I'd be very interested to see how this clustering method works compared to others. Specifically, hierarchical clustering is very similar. Using LSH implicitly creates a cluster hierarchy if you simply recursively combine adjacent buckets. I'm also curious if this would be useful from a topological perspective. If we compute the persistent cohomology of this graph, is that a reasonable thing to do or does that require a different approach?

Anyway, I hope you at least found this interesting and a jumping off point for further exploration of LSH. 

Cheers,

Jeremy

P.S. Here's the full code listing if you want to play with this yourself:

{% highlight python %}
import numpy
import networkx
from collections import defaultdict
from typing import List

def generate_data(points_per_group: int, means: List[List[float]], vars: List[float]):
    fresh = True
    for i, mean in enumerate(means):
        if fresh:
            data = numpy.random.multivariate_normal(mean, numpy.identity(len(mean)) * vars[i], points_per_group)
            fresh = False
        else:
            cluster = numpy.random.multivariate_normal(mean, numpy.identity(len(mean)) * vars[i], points_per_group)
            data = numpy.vstack((data, cluster))
    return data

def get_code_bit(data: numpy.ndarray) -> numpy.ndarray:
    norm_vec = 2 * numpy.random.random_sample((1, data.shape[1])).T - 1
    norm_vec = norm_vec / numpy.linalg.norm(norm_vec)
    return numpy.sign(data.dot(norm_vec)).reshape(data.shape[0])


#TODO: change string join to just tuple of row
def get_codes(data: numpy.ndarray, iterations: int) -> dict:
    codes = numpy.zeros((data.shape[0], iterations))
    iter_count = 0
    while iter_count < iterations:
        bit = get_code_bit(data)
        codes[:, iter_count] = bit
        iter_count += 1

    code_row_map = defaultdict(list)
    for index, row in enumerate(codes):
        code_row_map[''.join(map(str,row))].append(index)
    return code_row_map


def update_matrix(matrix: numpy.ndarray, code_map: dict) -> numpy.ndarray:
    for index_list in code_map.values():
        for index1 in index_list:
            for index2 in index_list:
                matrix[index1, index2] += 1
    return matrix


def weighted_hash_adj_mat(data: numpy.ndarray, iterations: int, prune_cutoff: int) -> numpy.ndarray:
    adj_matrix = numpy.zeros((data.shape[0], data.shape[0]))
    iteration = 0
    while iteration < iterations:
        code_map = get_codes(data, int(numpy.log2(data.shape[0])))
        adj_matrix = update_matrix(adj_matrix, code_map)
        iteration += 1
    
    return adj_matrix * (adj_matrix > prune_cutoff)


def make_graph(weighted_adj_mat: numpy.ndarray, data: numpy.ndarray) -> networkx.Graph:
    weighted_adj_mat = weighted_adj_mat.astype([('weight', int)])
    graph = networkx.Graph(weighted_adj_mat)
    for node in range(weighted_adj_mat.shape[0]):
        graph.nodes[node]['pos'] = data[node, :]
    return graph

data = generate_data(10, [[1,0], [-1,0]], [1,2])
norm_data = data / numpy.linalg.norm(data, axis=1).reshape((data.shape[0],1))

iterations = 5
w_adj_mat = weighted_hash_adj_mat(data, iterations, 2)
graph = make_graph(w_adj_mat, data)

pos = networkx.get_node_attributes(graph, 'pos')
weights = networkx.get_edge_attributes(graph,'weight').values()
networkx.draw(graph, pos, width=list(weights))
{% endhighlight %}

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-MML-AM_CHTML" type="text/javascript"></script>