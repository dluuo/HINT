# HINT: Hierarchical Mixture Networks For Coherent Probabilistic Forecasting
We present the Hierarchical Mixture Networks (HINT), a model family for efficient and accurate coherent forecasting. We specialize the networks on the task via a multivariate mixture optimized with composite likelihood and made coherent via bootstrap reconciliation. 
<div style="text-align:center">
<img src="./images/HINTLikelihood.jpg" width="700">
</div>

Additionally, we robustify the networks to stark time series scale variations, incorporating normalized feature extraction and recomposition of output scales within their architecture.
<div style="text-align:center">
<img src="./images/HINTArchitecture.jpg" width="700">
</div>

We demonstrate improved accuracy on several datasets compared to the existing state-of-the-art. 
<div style="text-align:center">
<img src="./images/HINTResults.jpg" width="700">
</div>

## Getting Started
To start using the HINT repository, run the following:
1. `bash setup.sh`
2. `conda activate HINT`

Now, you can run run_hint.py. For the jupyter notebooks, upload them to Colab to run on GPU.
