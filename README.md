# graph_optimization
Example code for graph optimization
- Example of `sklearn.cluster.SpectralClustering`
- Example of denseCRF

## Run examples
```bash
# install pydensecrf
pip install -U cython
pip install git+https://github.com/lucasb-eyer/pydensecrf.git

# install sklearn
pip install scikit-learn pyamg

# run example
git clone https://github.com/DIYer22/graph_optimization.git
cd graph_optimization/graph_optimization
python main.py

# install DIYer22/graph_optimization
pip install ..
```

## Refrence
- [lucasb-eyer/pydensecrf](https://github.com/lucasb-eyer/pydensecrf/blob/master/examples/Non%20RGB%20Example.ipynb)
- [iseg.affinity.normalized_cut](https://github.com/DIYer22/iseg)
- [ Overview of clustering methods - sklearn doc](https://scikit-learn.org/stable/modules/clustering.html)  
![](https://scikit-learn.org/stable/_images/sphx_glr_plot_cluster_comparison_001.png)
