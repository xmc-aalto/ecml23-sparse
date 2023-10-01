# Memory-Efficient Training for Extremely Large Output Spaces

The content of this repository provides supplementary material for the ECMLPKDD 2023
submission "Towards Memory-Efficient Training for Extremely Large Output Spaces – Learning with 500k Labels on a Single Commodity GPU"

## Improved Code
If you are not interested in the exact code for that paper, we recommend to use
* https://version.aalto.fi/gitlab/xmc/xmc-sparse-tensorflow (TF)
* https://version.aalto.fi/gitlab/AaltoRSE/xmc-sparse-pytorch (PyTorch)

instead, which provide improved tf/pytorch bindings for the CUDA kernels described in this work.
The kernels themselves are shared between the two implementations, and can be found at
  https://version.aalto.fi/gitlab/xmc/xmc-kernels.


This repository contains the implementation for several variations of a sparse
layer to be used with large output spaces.

## Additional Results

### Results for AmazonCat-670k with CascadeXML features.

| setup        |   connectivity | intermediate   |   train-p@1 |   train-p@3 |   train-p@5 |   test-p@1 |   test-p@3 |   test-p@5 |    memory |   epochs |   time-per-epoch |
|--------------|----------------|----------------|-------------|-------------|-------------|------------|------------|------------|-----------|----------|------------------|
| Dense        |            768 | --             |     99.7911 |     94.5024 |     89.0127 |    47.522  |    42.2592 |    38.2813 | 13.43     |     28.4 |          624.011 |
| Unstructured |             32 | --             |     88.8298 |     71.4445 |     55.6207 |    30.4205 |    23.7528 |    18.9574 |  6.26742  |     95   |         1369.11  |
| FFI          |             32 | --             |     92.4215 |     83.957  |     74.7516 |    37.0917 |    31.6191 |    27.5646 |  0.967195 |     76.2 |          233.85  |
| Unstructured |             32 | 32k            |     99.6758 |     94.3217 |     88.6631 |    42.5016 |    37.0632 |    33.0475 |  6.4952   |     36   |         1512.44  |
| FFI          |             32 | 16k            |     99.4587 |     93.9987 |     88.024  |    41.324  |    35.912  |    31.9143 |  1.24696  |     34   |          269.559 |
| FFI          |             32 | 32k            |     99.6657 |     94.3189 |     88.6623 |    42.5886 |    37.1176 |    33.1259 |  1.44841  |     36.4 |          270.986 |
| FFI          |             32 | 65k            |     99.7005 |     94.3741 |     88.7742 |    43.704  |    38.4273 |    34.4225 |  1.74489  |     39   |          305.333 |
| FFI          |             32 | 100k           |     99.6681 |     94.3223 |     88.6739 |    44.7123 |    39.3021 |    35.2915 |  2.37732  |     34   |          334.324 |
| FFI          |             64 | 16k            |     99.7429 |     94.4275 |     88.8914 |    43.3635 |    38.0539 |    34.1793 |  2.13541  |     27   |          290.407 |
| FFI          |             64 | 32k            |     99.7498 |     94.4322 |     88.8973 |    44.264  |    38.9257 |    35.0395 |  2.40721  |     31   |          305.677 |
| FFI          |             64 | 65k            |     99.7388 |     94.4095 |     88.8565 |    45.2638 |    39.8118 |    35.8794 |  2.54922  |     33   |          391.121 |
| FFI          |             64 | 100k           |     99.7233 |     94.3898 |     88.8111 |    45.6474 |    40.3326 |    36.3935 |  2.90615  |     31   |          435.258 |
| FFI          |             72 | 65k            |     99.7521 |     94.4363 |     88.9021 |    45.2501 |    39.9026 |    35.9882 |  2.69849  |     31   |          440.355 |
| Bottleneck   |             64 | 64             |     99.0821 |     93.28   |     86.391  |    37.9956 |    33.7436 |    30.3927 |  1.12913  |     31.6 |          231.95  |

### Results for AmazonCat-670k with Slice features.
| setup        |   connectivity | intermediate   |   train-p@1 |   train-p@3 |   train-p@5 |   test-p@1 |   test-p@3 |   test-p@5 |   memory |   epochs |   time-per-epoch |
|--------------|----------------|----------------|-------------|-------------|-------------|------------|------------|------------|----------|----------|------------------|
| Dense        |            512 | --             |     99.183  |     93.9117 |     88.3889 |   33.7581  |   29.6174  |   26.5754  | 8.95668  |     27.2 |          472.119 |
| Unstructured |             32 | --             |     64.7745 |     49.3734 |     38.9805 |   14.4525  |   11.5382  |    9.50668 | 6.35896  |     73   |         1357.4   |
| FFI          |             32 | --             |     16.1666 |     13.8996 |     12.4246 |    7.11897 |    6.30394 |    5.64143 | 0.967265 |     24.8 |          223.226 |
| Unstructured |             32 | 32k            |     98.8492 |     93.411  |     87.534  |   32.6522  |   28.6781  |   25.7911  | 6.42263  |     45   |         1618.49  |
| FFI          |             32 | 16k            |     98.2186 |     92.4527 |     85.9577 |   31.7301  |   27.8595  |   25.0244  | 1.10681  |     42   |          258.69  |
| FFI          |             32 | 32k            |     98.7012 |     93.2419 |     87.3461 |   32.8017  |   28.7496  |   25.9066  | 1.23167  |     38   |          244.41  |
| FFI          |             32 | 65k            |     98.9585 |     93.6096 |     87.9811 |   33.6958  |   29.6914  |   26.8382  | 1.34702  |     36   |          308.917 |
| FFI          |             32 | 100k           |     99.0403 |     93.7231 |     88.1518 |   34.246   |   30.1975  |   27.3442  | 1.77284  |     35   |          301.943 |
| FFI          |             64 | 16k            |     99.1016 |     93.796  |     88.1919 |   33.1554  |   29.1721  |   26.4054  | 1.94632  |     33   |          301     |
| FFI          |             64 | 32k            |     99.0978 |     93.7926 |     88.2467 |   33.9284  |   29.8703  |   27.0679  | 2.22399  |     32   |          314.375 |
| FFI          |             64 | 65k            |     99.0949 |     93.7879 |     88.2557 |   34.5571  |   30.5033  |   27.6944  | 2.54016  |     30   |          395.633 |
| FFI          |             64 | 100k           |     99.0678 |     93.7597 |     88.2203 |   35.0211  |   30.9764  |   28.0937  | 2.58486  |     29   |          410.931 |
| Bottleneck   |             64 | 64             |     96.3551 |     88.8836 |     80.0844 |   30.6909  |   27.332   |   24.5586  | 1.12888  |     33.6 |          218.986 |

### Results for Wiki500k with CascadeXML features
| setup        |   connectivity | intermediate   |   train-p@1 |   train-p@3 |   train-p@5 |   test-p@1 |   test-p@3 |   test-p@5 |    memory |   epochs |   time-per-epoch |
|--------------|----------------|----------------|-------------|-------------|-------------|------------|------------|------------|-----------|----------|------------------|
| Dense        |            768 | --             |     96.7006 |     79.7239 |     64.2443 |    77.1725 |    58.5511 |    45.109  | 10.0425   |     25.6 |         1743.5   |
| Unstructured |             32 | --             |     78.272  |     54.6852 |     39.8174 |    65.2482 |    43.7313 |    31.397  |  4.78792  |    100   |         3869.91  |
| FFI          |             32 | --             |     69.0913 |     51.786  |     40.4382 |    58.6727 |    41.9824 |    32.2036 |  0.723222 |     59.4 |          714.551 |
| Unstructured |             32 | 32k            |     92.4128 |     73.5267 |     58.1329 |    73.6962 |    54.7483 |    42.0102 |  4.91421  |     58   |         4422.74  |
| FFI          |             32 | 16k            |     90.715  |     71.4406 |     56.3196 |    73.1226 |    54.1514 |    41.5197 |  0.918788 |     68   |          746.029 |
| FFI          |             32 | 32k            |     92.9627 |     74.3053 |     58.8948 |    73.6475 |    54.7795 |    42.0569 |  1.02335  |     67.4 |          841.638 |
| FFI          |             32 | 65k            |     94.3522 |     76.3125 |     60.8353 |    74.0494 |    55.4152 |    42.6315 |  1.56608  |     56   |          928.446 |
| FFI          |             32 | 100k           |     94.4506 |     76.5713 |     61.1356 |    74.3155 |    55.7567 |    42.9536 |  2.35529  |     49   |         1261.92  |
| FFI          |             64 | 16k            |     94.0797 |     76.0055 |     60.4043 |    74.3868 |    55.6304 |    42.7501 |  1.66523  |     56   |          878     |
| FFI          |             64 | 32k            |     94.7653 |     77.0454 |     61.4781 |    74.3879 |    55.81   |    42.9298 |  1.89684  |     48   |          929.083 |
| FFI          |             64 | 65k            |     95.2841 |     77.8303 |     62.3177 |    74.511  |    56.0454 |    43.1884 |  2.03168  |     43   |         1167.09  |
| FFI          |             64 | 100k           |     95.844  |     78.6381 |     63.1405 |    74.6282 |    56.2354 |    43.3671 |  2.65133  |     45   |         1530.38  |
| Bottleneck   |             64 | 64             |     86.4291 |     64.8097 |     49.5022 |    71.8954 |    50.6959 |    37.9261 |  0.961938 |     47.6 |          678.258 |

### Results for Wiki500k with Slice features
| setup        |   connectivity | intermediate   |   train-p@1 |   train-p@3 |   train-p@5 |   test-p@1 |   test-p@3 |   test-p@5 |   memory |   epochs |   time-per-epoch |
|--------------|----------------|----------------|-------------|-------------|-------------|------------|------------|------------|----------|----------|------------------|
| Dense        |            512 | --             |     97.3323 |     77.4726 |     60.4051 |    58.2486 |    37.9121 |    28.0296 | 6.69749  |     39.4 |         1248.76  |
| Unstructured |             32 | --             |     58.3302 |     37.7008 |     28.0696 |    45.4878 |    27.2922 |    19.8603 | 4.78605  |     78   |         3612.26  |
| FFI          |             32 | --             |     42.6392 |     28.0085 |     21.9497 |    37.5337 |    23.2219 |    17.6438 | 0.723264 |     54.8 |          659.136 |
| Unstructured |             32 | 32k            |     83.736  |     61.4217 |     47.6871 |    59.0107 |    38.4807 |    28.9021 | 4.81408  |     40   |         3977.18  |
| FFI          |             32 | 16k            |     80.2585 |     58.1899 |     45.1227 |    58.0007 |    37.7422 |    28.3634 | 0.915993 |     59   |          946.475 |
| FFI          |             32 | 32k            |     84.239  |     62.1931 |     48.3855 |    58.86   |    38.4366 |    28.8706 | 1.04134  |     45.8 |          722.607 |
| FFI          |             32 | 65k            |     88.4613 |     66.8306 |     52.229  |    59.828  |    39.2378 |    29.4813 | 1.18353  |     37   |          820.946 |
| FFI          |             32 | 100k           |     89.8285 |     68.5116 |     53.6939 |    60.5609 |    39.7886 |    29.9324 | 1.59235  |     34   |         1105.5   |
| FFI          |             64 | 16k            |     86.4969 |     64.5537 |     50.1941 |    59.3778 |    38.6824 |    29.0105 | 1.539    |     52   |          807.365 |
| FFI          |             64 | 32k            |     89.062  |     67.444  |     52.6204 |    59.9681 |    39.2525 |    29.4548 | 1.78299  |     43   |          842.814 |
| FFI          |             64 | 65k            |     92.195  |     71.209  |     55.8075 |    60.5108 |    39.7495 |    29.8347 | 1.92379  |     38   |         1035.21  |
| FFI          |             64 | 100k           |     93.2908 |     72.7596 |     57.2106 |    61.0234 |    40.1683 |    30.1757 | 2.05862  |     38   |         1335.05  |
| Bottleneck   |             64 | 64             |     71.7854 |     50.0146 |     38.4837 |    56.5045 |    36.5249 |    27.5013 | 0.967719 |     41.8 |          638.877 |

### Preliminary results for Amazon3M with CascadeXML features
| setup      |   connectivity | intermediate   |   train-p@1 |   train-p@3 |   train-p@5 |   test-p@1 |   test-p@3 |   test-p@5 |   memory |   epochs |   time-per-epoch |
|------------|----------------|----------------|-------------|-------------|-------------|------------|------------|------------|----------|----------|------------------|
| Dense      |            768 | --             |   89.8897   |   83.936    |    79.4433  |  53.3569   |  50.6458   |  48.381    | 56.3     |       37 |          4227.95 |
| FFI        |             32 | 65k            |   68.0821   |   62.0092   |    58.0273  |  48.0666   |  44.1058   |  41.4422   | 4.47     |      100 |          1900.88 |
| FFI        |             32 | 131k           |   71.1833   |   64.8428   |    60.6424  |  49.1193   |  44.9967   |  42.3026   | 5.01     |      100 |          1901.56 |
| FFI        |             64 | 131k           |   78.1226   |   71.9641   |    67.6447  |  50.3997   |  46.7399   |  44.2132   | 8.67     |      100 |          2336.67 |

## Building the library
First, create a conda environment as provided by `environment.yml`, e.g. through
```bash
conda conda env create -f environment.yml
```

Activate the environment.
Then, configure `CMake` for a build directory (e.g. `build`) and run the build
```bash
cmake -S . -B build
cmake --build build --target sparseops
```

After this, there should be a file `build/libsparseops.so` which contains the compiled
parts of the library.

## Running the python code
The `sparse` subdirectory contains the glue code that makes the custom kernels usable
in tensorflow (the `ops` subdirectory), and several implementations of sparse multiplication layers
and corresponding utilities (`layers` subdirectory). In order to be able to use the fast sparse
layer, the `libsparseops.so` file needs to be placed alongside `ops/fixed_fan_in_ops.py`.

An example script is given in `run.py`, which runs a (sparse) training experiment specified in
a `json` file. The tasks uses for the paper are given in the `tasks` subdirectory.
