# Semeion-Handwritten-Digit-Dataset-Classification
Performed classification of several handwritten digits in the Semeion Handwritten Digit dataset  using Normalized Cut, Mean Shift, and Latent Class Analysis. Compared the performance of these  three approaches to identify the most effective one.

---
## **Explaination**
Perform classification of the Semeion Handwritten Digit data set using
```ruby
  -Latent class analysis;
  -Mean shift;
  -Normalized cut.
```
Mean shift and Normalized Cut assume that the images are vectors in a 256 dimensional Euclidean space. Provide the code and the extracted clusters as the number of clusters k varies from 5 to 15, for LCA and normalized-cut, while for Mean shift vary the kernel width. For each value of k (or kernel width) provide the value of the Rand index
```ruby
R=2(a+b)/(n(n-1))
```
where n is the number of images in the dataset. **a** is the number of pairs of images that represent the same digit and that are clustered together.
is the number of pairs of images that represent different digits and that are placed in different clusters.
Explain the differences between the three models.

**Tip**: Bernoulli models can be visualized as a greyscale image to inspect the learned model.

## **Overview**
In this assignment, we perform classification by using 3 algorithms LCA, Mean Shift, and Normalized Cut on the Semeion Handwritten
Digit dataset.
We modify the number of clusters in each cluster and the width band in case using mean shift algorithm.
For each cluster, we run the __rand score__, __adjusted rand score__ and compare them.
Also, Survey the averages by the __number of clusters__, and the __number of DR models__.
And in the final show, the best ARI achieved.
## **Clustering**
We have Supervised learning, Unsupervised learning and Semi supervised learning which each of them has different method to do. Clustering is one of the unsupervised learning method that trained a machine using information are not labeled. The task of clustering algorithms is divide the collection of data points or objects into number of groups with similar role or characterstics.
## **Code Explaination**
First we call data from dataset we downloaded before by the function get_data(). By the class NoDR() we make a fake class for no dimension reduction.
```ruby
def get_data():
 data = np.loadtxt('./semeion.data', dtype=np.int8)
 return data[:, :256], data[:, 256:]

def one_hot_decode(y: np.array):
 return np.argmax(y, axis=1)

def get_data_transformed():
 data = np.loadtxt('./semeion.data', dtype=np.int8)
 return data[:, :256], one_hot_decode(data[:, 256:])
```
```ruby
class NoDR:
 def __init__(self, n_components):
 pass

 def fit_transform(self, X):
 return X.copy()
```
**ks** is a vector containing the number of cluster we want for assignment. And, **ds** is a vector containing the dimension for dimensionality
reduction.
```ruby
simplefilter(action='ignore', category=FutureWarning)
ks = range(5, 16)
ds = (2, 64, 128, 256)
```
## **1- LCA**
**Latent Class Analysis** or **LCA** is a way to uncover hidden groupings in data. More specifically, it’s a way to to group subjects from multivariate data into “latent classes” — groups or subgroups with similar, unobservable, membership.
-Latent implies that the analysis is based on an error-free latent variable (Collins & Lanza, 2013).
-Classes are groups formed by uncovering hidden (latent) patterns in data.
LCA find the connections between the objects and group them like cluster analysis.

Now, A glance view to LCA code and briefly explanation the code, after that the explaining the performance and result.
The function **_log_no_underflow** computes log without underflow which has a parameter x which should be float.
```ruby
def _log_no_underflow(x):
 return np.log(np.clip(x, 1e-15, 1)
```
The function **estimate_bernoulli_parameters** estimate the bernouli distribution parameters. Parameter X is the input data array. resp array
has the responsibility for each data sample in X. nk is the numbers of data samples in the current components and means is the centers of
the current components.
```ruby
def _estimate_bernoulli_parameters(X, resp):
 nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
 means = np.dot(resp.T, X) / nk[:, np.newaxis]
 return nk, means
```
The function **estimate_log_bernouli_prob** has the duty of estimate the log bernoulli probability. It has X and means parameters.
```ruby
def _estimate_log_bernoulli_prob(X, means):
 return (X.dot(_log_no_underflow(means).T) +
 (1 - X).dot(_log_no_underflow(1 - means).T))
```
Since **Bernoulli_mixture** class has a lot of function, I decide to explain it inside the code.
```ruby
class Bernoulli_Mixture:
 def __init__(self, n_components, max_iter=500, tol=1e-3):
 self.n_components = n_components
 self.max_iter = max_iter
 self.tol = tol

 def __str__(self):
 return 'BMM'

 def fit(self, X, init_kmeans=True):
 #Estimate model parameters using X and predict the labels for X.
 #Parameter X is a list of n_features-dimentional data points. each row corresponds to a single data point.
 self.converged_ = False
 n_samples, _ = X.shape
 random_state = 42
 if init_kmeans:
 resp = np.zeros((n_samples, self.n_components))
 label = (
 KMeans(n_clusters=self.n_components, n_init=1, random_state=random_state)
 .fit(X)
 .labels_
 )
 resp[np.arange(n_samples), label] = 1
 else:
 resp = random_state.rand(n_samples, self.n_components)
 resp /= resp.sum(axis=1)[:, np.newaxis]
 self._initialize(X, resp)
 lower_bound = -np.inf
 for n_iter in range(1, self.max_iter + 1):
 prev_lower_bound = lower_bound
 log_prob_norm, log_resp = self._e_step(X)
 self._m_step(X, log_resp)
 lower_bound = log_prob_norm
 change = lower_bound - prev_lower_bound
 if abs(change) < self.tol:
 self.converged_ = True
 break
 if not self.converged_:
 raise ValueError('Not converged')
 _, log_resp = self._e_step(X)
 self.labels_ = log_resp.argmax(axis=1)

 def _e_step(self, X):
 # -log_prob_norm which is float value has duty to get Mean of the logarithms of the probabilities of each sample in X.
 #-log_responsibility get the Logarithm of the posterior probabilities (or responsibilities) of the point of each sample in X.
 log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
 return np.mean(log_prob_norm), log_resp
 def _estimate_weighted_log_prob(self, X):
 #Estimate the weighted log-probabilities, log P(X | Z) + log weights.
 return self._estimate_log_prob(X) + self._estimate_log_weights()

 def _estimate_log_prob_resp(self, X):
 #Estimate log probabilities and responsibilities for each sample.
 #Compute the log probabilities, weighted log probabilities for  component and responsibilities for each sample in X with respect to the current state of the model.
 weighted_log_prob = self._estimate_weighted_log_prob(X)
 log_prob_norm = logsumexp(weighted_log_prob, axis=1)
 with np.errstate(under="ignore"):
 # ignore underflow
 log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
 return log_prob_norm, log_resp

 def _initialize(self, X, resp):
 #Initialization of the Bernoulli mixture parameters.
 n_samples, n_dim = X.shape
 weights, means = _estimate_bernoulli_parameters(X, resp)
 weights /= n_samples
 self.weights_ = weights
 self.means_ = means

 def _m_step(self, X, log_resp):
 #get the Logarithm of the posterior probabilities (or responsibilities) of the point of each sample in X.
 n_samples, _ = X.shape
 self.weights_, self.means_ = (
 _estimate_bernoulli_parameters(X, np.exp(log_resp))
 )
 self.weights_ /= n_samples

 def _estimate_log_prob(self, X):
 return _estimate_log_bernoulli_prob(X, self.means_)

 def _estimate_log_weights(self):
 return np.log(self.weights_)
```
The **LCA** function read and split data and by using loop clustering model compute dimension, number of cluster, time, adjusted Rand Index
and Rand Index and record and save them in a LCA.csv file.
```ruby
def LCA(p=False):
 columns = ('clustering_model', 'dim_red_model', 'd', 'k', 'time', 'ARI', 'RI')
 rows = []
 # read data and split
 X, y = get_data_transformed()
 # loop clusteting model
 for k in ks:
 for d in ds:
 if d != 256:
 embedding_model = PCA
 else:
 embedding_model = NoDR
 start_time = time()
 embedding = embedding_model(n_components=d)
 X_transformed = embedding.fit_transform(X)
 model = Bernoulli_Mixture(k)
 model.fit(X_transformed)
 elapsed = time() - start_time
 title = f'Results for {model} on {d}-{embedding_model.__name__} - k={k} ({elapsed:.02f}s)'
 print(title)
 ari, ri = print_rand(y, model.labels_)
 if p: plot2D(X_transformed, model.labels_, title)
 result = (str(model), embedding_model.__name__, d, k, elapsed, ari, ri)
 rows.append(result)
 df = pd.DataFrame(data=rows, columns=columns)
 print(df)
 df.to_csv('LCA.csv', index=False)
```
## **Explaining the performance**
You can see all the results by using algorithm LCA in the left side and in the right side we compute the averages by the number of cluster,
number of DR models and Best Rand index with more than 90%.
## **2- Mean Shift**
Mean shift clustering aims to discover “blobs” in a smooth density of samples. It is a centroid-based algorithm, which works by updating
candidates for centroids to be the mean of the points within a given region. These candidates are then filtered in a post-processing stage
to eliminate near-duplicates to form the final set of centroids.
Seeding is performed using a binning technique for scalability.
*Class **Mean_Shift** using mean shift segmentation algorithm which has (int)Bandwindth parameter which is the number of neighbors
considered.
we use n_jobs=1 because this will be used in nested calls under parallel calls to mean_shift_single_seed. so we don't need for future
parallelism.
**all_res** execute iterations on all seeds in parallel and copy results in a dictionary.
post processing remove near duplicate points. If the distance between two kernels is less than the bandwidth, then we have to remove one
because it is a duplicate. Remove the one with fewer points.
```ruby
class Mean_Shift:
 def __init__(self, bandwidth, max_iter=200, n_jobs=4):
 self.bandwidth = bandwidth
 self.max_iter = max_iter
 self.n_jobs = n_jobs
 def __str__(self):
 return 'MS'
 def fit(self, X):
 seeds = X
 n_samples, n_features = X.shape
 center_intensity_dict = {}
 nbrs = NearestNeighbors(radius=self.bandwidth, n_jobs=1).fit(X)
 all_res = Parallel(n_jobs=self.n_jobs)(
 delayed(_mean_shift_single_seed)
 (seed, X, nbrs, self.max_iter) for seed in seeds)
 for i in range(len(seeds)):
 if all_res[i][1]: # i.e. len(points_within) > 0
 center_intensity_dict[all_res[i][0]] = all_res[i][1]
 self.n_iter_ = max([x[2] for x in all_res])
 if not center_intensity_dict:
 # nothing near seeds
 raise ValueError("No point was within bandwidth=%f of any seed."
 " Try a different seeding strategy \
 or increase the bandwidth."
 % self.bandwidth)
 sorted_by_intensity = sorted(center_intensity_dict.items(),
 key=lambda tup: (tup[1], tup[0]),
 reverse=True)
 sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
 unique = np.ones(len(sorted_centers), dtype=bool)
 nbrs = NearestNeighbors(radius=self.bandwidth,
 n_jobs=self.n_jobs).fit(sorted_centers)
 for i, center in enumerate(sorted_centers):
 if unique[i]:
 neighbor_idxs = nbrs.radius_neighbors([center],
 return_distance=False)[0]
 unique[neighbor_idxs] = 0
 unique[i] = 1 # leave the current point as unique
 cluster_centers = sorted_centers[unique]
 # ASSIGN LABELS: a point belongs to the cluster that it is closest to
 nbrs = NearestNeighbors(n_neighbors=1,
 n_jobs=self.n_jobs).fit(cluster_centers)
 labels = np.zeros(n_samples, dtype=int)
 idxs = nbrs.kneighbors(X, return_distance=False)
 labels = idxs.flatten()
 self.cluster_centers_, self.labels_, self.nlabels_ = cluster_centers, labels, len(np.unique(labels))
 return self
```
the **mean_shift_single_seed** function is a separate function for each seed iterative loop. For each seed, climb gradient until convergence. if
converged or at max_iter, it add the cluster.
```ruby
def _mean_shift_single_seed(my_mean, X, nbrs, max_iter):
 bandwidth = nbrs.get_params()['radius']
 stop_thresh = 1e-3 * bandwidth
 completed_iterations = 0
 while True:
 i_nbrs = nbrs.radius_neighbors([my_mean], bandwidth,
 return_distance=False)[0]
 points_within = X[i_nbrs]
 if len(points_within) == 0:
 break
 my_old_mean = my_mean
 my_mean = np.mean(points_within, axis=0)
 if (np.linalg.norm(my_mean - my_old_mean) < stop_thresh or
 completed_iterations == max_iter):
 break
 completed_iterations += 1
 return tuple(my_mean), len(points_within), completed_iterations
```
The **MS** function read and split data and by using loop clustering model compute dimension, number of cluster, time, adjusted Rand Index
and Rand Index and record and save them in a MS.csv file.
```ruby
def MS(p=False):
 columns = ('clustering_model','dim_red_model','d','k','time','ARI','RI')
 rows = []
 params = ((1, 2), (6.3, 64), (7.5, 128), (7, 256))
 X, y = get_data_transformed()
 for w,d in params:
 model = Mean_Shift(w)
 start_time = time()
 if d != 256:
 embedding_model = PCA
 else:
 embedding_model = NoDR
 X_transformed = embedding_model(n_components=d).fit_transform(X)
 ename = embedding_model.__name__
 model.fit(X_transformed)
 elapsed = time() - start_time
 title = f'Results for {model} on dim={d} ({elapsed:.02f}s)\n' +\
 f'number of centers: {model.nlabels_} - window size: {w}'
 labels = model.labels_
 print(title)
 ari, ri = print_rand(y, model.labels_)
 if p: plot2D(X_transformed, model.labels_, title)
 result = (str(model),ename,d, w, elapsed, ari, ri)
 rows.append(result)
 df = pd.DataFrame(data=rows, columns=columns)
 print(df)
 df.to_csv('MS.csv', index=False)
```
## **Explaining the performance**
You can see all the results at the bottom by using algorithm Mean Shift. We compute the averages by the number of cluster, number of DR
models and Best Rand index which it is less than LCA algoritm.
## **3- Normalized Cut**
The basic idea of this problem is that we want to find the weights in that graph that minimize the cut. In doing so we found a big big
problem because to find the 2 clusters and the cut we do not take into consideration what happens in between the clusters and so we will
find two unbalanced clusters. Normalized cut is the CUT problem that takes into consideration what happens in between the clusters. It
does the CUT as before but takes into consideration the volume of A and the volume of B in such a way that now it creates the balanced
clusters.
Class **N_Cut** and specially fit function inside has duty to find largest and smallest eigenvalues and nonzero eigenvalues.
```ruby
class N_Cut():
 def __init__(self, n_clusters):
 self.n_clusters = n_clusters
 def __str__(self):
 return 'NCUT'
 def fit(self, X, verbose=False):
 self.n_samples, self.n_features = X.shape
 A = rbf_kernel(X)
 np.fill_diagonal(A, 0.0)
 G = nx.from_numpy_matrix(A)
 L = nx.normalized_laplacian_matrix(G)
 w = np.linalg.eigvals(L.A)
 if verbose:
 print("Largest eigenvalue:", max(w))
 print("Smallest eigenvalue:", min(w))
 print(f"first k={self.n_clusters} nonzero eigenvalues:", w[1:self.n_clusters+1])
 w, v = np.linalg.eig(L.A)
 if not abs(w[0]) < 1e-10:
 raise ValueError(f'First eigenvalue is {w[0]}.\nMust be close to zero.')
 U = v[:, 1:self.n_clusters+1]
 kmeans = KMeans(n_clusters=self.n_clusters).fit(U)
 self.labels_ = kmeans.labels_
```
The **NCut_Result** function read and split data and by using loop clustering model compute dimension, number of cluster, time, adjusted
Rand Index and Rand Index and record and save them in a NCut.csv file.
```ruby
def NCut_Result(p=False):
 columns = ('clustering_model', 'dim_red_model', 'd', 'k', 'time', 'ARI', 'RI')
 rows = []
 X, y = get_data_transformed()
 for k in ks:
 for d in ds:
 if d != 256:
 embedding_model = PCA
 else:
 embedding_model = NoDR
 start_time = time()
 embedding = embedding_model(n_components=d)
 X_transformed = embedding.fit_transform(X)
 model = N_Cut(k) # clustering model
 model.fit(X_transformed)
 elapsed = time() - start_time
 title = f'Results for {model} on {d}-{embedding_model.__name__} - k={k} ({elapsed:.02f}s)'
 print(title)
 ari, ri = print_rand(y, model.labels_)
 if p: plot2D(X_transformed, model.labels_, title)
 result = (str(model), embedding_model.__name__, d, k, elapsed, ari, ri)
 rows.append(result)
 df = pd.DataFrame(data=rows, columns=columns)
 print(df)
 df.to_csv('NCut.csv', index=False)
```
## **Explaining the performance**
You can see all the results by using algorithm **NCut** in the left side and in the right side we compute the averages by the number of cluster,
number of DR models and Best Rand index with more than 86% which is almost similar to LCA.
## **Compare Overall**
As you can see by the results the best classifier used in this assignment is the **LCA** by looking the data. After that, second place by the
results we export is **Normalized Cut** which almost near and similar to LCA. and the third place by the results belongs to **MeanShift**. After
discussions, we agreed that we need more input data to make clustering better. Also, if we have to chance to choose one of the these
algorithms, clearly we choose Normalized cut since it is really easy to understand how it works and also where are the problem.
