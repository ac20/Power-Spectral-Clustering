# Power Spectral Clustering

This repository contains the guide/code for generating the results from article "Power Spectral Clustering" [paper](https://hal.archives-ouvertes.fr/hal-01516649/document).

## Example 1 : Illustration of PRcut vs Rcut on Two Circles

`Example1_TwoCircles.py` is used to generate the Figs. 2 and 3 in the paper. It compares Power Ratio Cut with Ratio cut on time/accuracy and robustness to noise on the toy example of two concentric circles.

run `python Example1_TwoCircles.py` and results are saved in (./img/Example1/)

## Example 2 : Illustration of PRcut vs Ncut on Blobs

`Example2_twoBlobs.py` contains the code to illustrate the use of Power Ratio cut on another toy example - Blobs data, and compare with the Normalized Cut.

run `python Example2_twoBlobs.py` and results are saved in (./img/Example2/)

## Example 3 : PRcut behavior on flat zones

`Example3a_flatzone.py` and `Example3b_flatzone.py` are used to illustrate how PRcut behaves on flat zones. This is used to generate Figs. 4 and 5. in the article.

run `python Example3a_flatzone.py` and results are saved in (./img/Example3/). Generates Fig 4 in the paper - typical case of clustering flat zones.

run `python Example3b_flatzone.py` and results are saved in (./img/Example3/). Generates Fig 5 in the paper - typical case of clustering flat zones.

## Example 4 : Timing analysis of PRcut

`Example4_timing.py` is used to analyze the timing of PRcut vs Rcut on the blobs dataset. This code generates the Fig 6. in the article.

## Example 5 : Application of PRcut on connect-4 dataset

(Yet to be uploaded)

## Example 6 : Application of PRcut and Rcut on MNIST dataset

Download the dataset from [here](http://yann.lecun.com/exdb/mnist/).

`Example6_MNIST.py` is used to calculate the time/accuracy of PRcut vs Rcut and generate some typical results.

`Example6_plotResults.py` is used to output the results calculated using  `Example6_MNIST.py` using Macbook Pro (2015) 15" model, 16 GB RAM and i7 processor. These results are taken from the file `results_6_i7.csv`. Another set of results are stored in `results_6_Xeon.csv` generated on 2010 intel xeon processor with 8 GB RAM.

Results from the above files are stored in (./img/)

## Example 7: Results on high-dimensional blobs

(Yet to be Uploaded)

## Example 8 : Results on hyperspectral data

Files `Example8a_hyperspectral.py`, `Example8b_hyperspectral.py` and `Example8c_hyperspectral.py` calculates the result of PRcut and Rcut on hyperspectral images of "University Pavia", "Pavia City" and "Salinas" datasets, which can be obtained from [here](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).

Download the datasets and save them in (./Hyperspectral_data/) and then run the codes.

`Example8_genetateGraphs.py` uses the saved results generated on 2010 intel xeon processor with 8 GB RAM, in `results_8a.csv`, `results_8b.csv` and `results_8c.csv`.
