# Reference

## Paper

ATL: Autonomous Knowledge Transfer from Many Streaming Processes

[ArXiv](https://arxiv.org/abs/1910.03434)

[ResearchGate](https://www.researchgate.net/publication/336361712_ATL_Autonomous_Knowledge_Transfer_from_Many_Streaming_Processes)

[ACM Digital Library](https://dl.acm.org/action/doSearch?AllField=ATL&expand=all&ConceptID=119445)

## Bibtex

```
@inproceedings{10.1145/3357384.3357948,
author = {Pratama, Mahardhika and de Carvalho, Marcus and Xie, Renchunzi and Lughofer, Edwin and Lu, Jie},
title = {ATL: Autonomous Knowledge Transfer from Many Streaming Processes},
year = {2019},
isbn = {9781450369763},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3357384.3357948},
doi = {10.1145/3357384.3357948},
booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
pages = {269–278},
numpages = {10},
keywords = {concept drif, transfer learning, deep learning, multistream learning},
location = {Beijing, China},
series = {CIKM ’19}
}
```

# Notes

If you want to see the original code used for this paper, access [ATL_Matlab](https://github.com/Ivsucram/ATL_Matlab)

`ATL_Python` is a reconstruction of `ATL_Matlab` made by the same author, but using Python 3.6 and PyTorch (with autograd enabled and GPU support).

# ATL_Python

ATL: Autonomous Knowledge Transfer From Many Streaming Processes
ACM CIKM 2019

1. Clone `ATL_Python` git to your computer, or just download the files.

2. Install [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html).

3. Open Anaconda prompt and travel until ATL folder.

4. Run the following command `conda env create -f environment.yml`. This will create an environment called `atl` with every python packaged/library needed to run ATL.

5. Enable ATL environment by running the command `activate atl` or `conda activate atl`. 

6. Provide a dataset by replacing the file `data.csv`
The current `data.csv` holds [SEA](https://www.researchgate.net/publication/221653408_A_Streaming_Ensemble_Algorithm_SEA_for_Large-Scale_Classification) dataset.
`data.csv` must be prepared as following:

```
- Each row presents a new data sample
- Each column presents a data feature
- The last column presents the label for that sample. Don't use one-hot encoding. Use a format from 1 onwards
```

7. Run `python ATL.py`

ATL will automatically normalize your data and split your data into 2 streams (Source and Target data streams) with a bias between them, as described in the paper.

ATL statues are printed at the end of every minibatch, where you will be able to follow useful information as:

```
- Training time (maximum, mean, minimum, current and accumulated)
- Testing time (maximum, mean, minimum, current and accumulated)
- Classification Rate for the Source (maximum, mean, minimum and current)
- Classification Rate for the Target (maximum, mean, minimum and current)
- Classification Loss for the Source (maximum, mean, minimum and current)
- Classification Loss for the Target (maximum, mean, minimum and current)
- Reconstruction Loss for the Source (maximum, mean, minimum and current)
- Reconstruction Loss for the Target (maximum, mean, minimum and current)
- Kullback-Leibler Loss (maximum, mean, minimum and current)
- Number of nodes (maximum, mean, minimum and current)
- And a quick review of ATL structure (both discriminative and generative phases), where you can see how many automatically generated nodes were created.
```

At the end of the process, ATL will plot 6 graphs:

```
- The processing time per mini-batch and the total processing time as well, both for training and testing
- The evolution of nodes over time
- The target and source classification rate evolution, as well as the final mean accuracy of the network 
- The number of GMMs on Source AGMM and Target AGMM
- Losess for the source and target classification as well as source and target reconstruction
- Bias and Variance of the discriminative phase
- Bias and Variance of the generative phase
```

Thank you.

# Download all datasets used on the paper

As some datasets are too big, we can't upload them to GitHub. GitHub has a size limit of 35MB per file. Because of that, you can find all the datasets in a csv format on the anonymous link below. To test it, copy the desired dataset to the same folder as ATL and rename it to `data.csv`.

- [https://drive.google.com/drive/folders/1Te0KMqJ5DUVuJK3tVt1l3AbHuKR4CxP9](https://drive.google.com/drive/folders/1Te0KMqJ5DUVuJK3tVt1l3AbHuKR4CxP9)



