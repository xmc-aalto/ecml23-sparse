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
| setup        |   connectivity | intermediate   |   train-p@1 |   train-p@3 |   train-p@5 |   test-p@1 |   test-p@3 |   test-p@5 |   memory |   epochs |   time-per-epoch |
|--------------|----------------|----------------|-------------|-------------|-------------|------------|------------|------------|----------|----------|------------------|
| Dense        |            768 | --             |       99.79 |       94.5  |       89.01 |      47.52 |      42.26 |      38.28 |    13.43 |     28.4 |              624 |
| Unstructured |             32 | --             |       88.83 |       71.44 |       55.62 |      30.42 |      23.75 |      18.96 |     6.27 |     95   |             1369 |
| FFI          |             32 | --             |       92.42 |       83.96 |       74.75 |      37.09 |      31.62 |      27.56 |     0.97 |     76.2 |              234 |
| Unstructured |             32 | 32k            |       99.68 |       94.32 |       88.66 |      42.5  |      37.06 |      33.05 |     6.5  |     36   |             1512 |
| FFI          |             32 | 16k            |       99.46 |       94    |       88.02 |      41.32 |      35.91 |      31.91 |     1.25 |     34   |              270 |
| FFI          |             32 | 32k            |       99.67 |       94.32 |       88.66 |      42.59 |      37.12 |      33.13 |     1.45 |     36.4 |              271 |
| FFI          |             32 | 65k            |       99.7  |       94.37 |       88.77 |      43.7  |      38.43 |      34.42 |     1.74 |     39   |              305 |
| FFI          |             32 | 100k           |       99.67 |       94.32 |       88.67 |      44.71 |      39.3  |      35.29 |     2.38 |     34   |              334 |
| FFI          |             64 | 16k            |       99.74 |       94.43 |       88.89 |      43.36 |      38.05 |      34.18 |     2.14 |     27   |              290 |
| FFI          |             64 | 32k            |       99.75 |       94.43 |       88.9  |      44.26 |      38.93 |      35.04 |     2.41 |     31   |              306 |
| FFI          |             64 | 65k            |       99.74 |       94.41 |       88.86 |      45.26 |      39.81 |      35.88 |     2.55 |     33   |              391 |
| FFI          |             64 | 100k           |       99.72 |       94.39 |       88.81 |      45.65 |      40.33 |      36.39 |     2.91 |     31   |              435 |
| FFI          |             72 | 65k            |       99.75 |       94.44 |       88.9  |      45.25 |      39.9  |      35.99 |     2.7  |     31   |              440 |
| Bottleneck   |             64 | 64             |       99.08 |       93.28 |       86.39 |      38    |      33.74 |      30.39 |     1.13 |     31.6 |              232 |

### Results for AmazonCat-670k with Slice features.
| setup        |   connectivity | intermediate   |   train-p@1 |   train-p@3 |   train-p@5 |   test-p@1 |   test-p@3 |   test-p@5 |   memory |   epochs |   time-per-epoch |
|--------------|----------------|----------------|-------------|-------------|-------------|------------|------------|------------|----------|----------|------------------|
| Dense        |            512 | --             |       99.18 |       93.91 |       88.39 |      33.76 |      29.62 |      26.58 |     8.96 |     27.2 |              472 |
| Unstructured |             32 | --             |       64.77 |       49.37 |       38.98 |      14.45 |      11.54 |       9.51 |     6.36 |     73   |             1357 |
| FFI          |             32 | --             |       16.17 |       13.9  |       12.42 |       7.12 |       6.3  |       5.64 |     0.97 |     24.8 |              223 |
| Unstructured |             32 | 32k            |       98.85 |       93.41 |       87.53 |      32.65 |      28.68 |      25.79 |     6.42 |     45   |             1618 |
| FFI          |             32 | 16k            |       98.22 |       92.45 |       85.96 |      31.73 |      27.86 |      25.02 |     1.11 |     42   |              259 |
| FFI          |             32 | 32k            |       98.7  |       93.24 |       87.35 |      32.8  |      28.75 |      25.91 |     1.23 |     38   |              244 |
| FFI          |             32 | 65k            |       98.96 |       93.61 |       87.98 |      33.7  |      29.69 |      26.84 |     1.35 |     36   |              309 |
| FFI          |             32 | 100k           |       99.04 |       93.72 |       88.15 |      34.25 |      30.2  |      27.34 |     1.77 |     35   |              302 |
| FFI          |             64 | 16k            |       99.1  |       93.8  |       88.19 |      33.16 |      29.17 |      26.41 |     1.95 |     33   |              301 |
| FFI          |             64 | 32k            |       99.1  |       93.79 |       88.25 |      33.93 |      29.87 |      27.07 |     2.22 |     32   |              314 |
| FFI          |             64 | 65k            |       99.09 |       93.79 |       88.26 |      34.56 |      30.5  |      27.69 |     2.54 |     30   |              396 |
| FFI          |             64 | 100k           |       99.07 |       93.76 |       88.22 |      35.02 |      30.98 |      28.09 |     2.58 |     29   |              411 |
| Bottleneck   |             64 | 64             |       96.36 |       88.88 |       80.08 |      30.69 |      27.33 |      24.56 |     1.13 |     33.6 |              219 |

### Results for Wiki500k with CascadeXML features
| setup        |   connectivity | intermediate   |   train-p@1 |   train-p@3 |   train-p@5 |   test-p@1 |   test-p@3 |   test-p@5 |   memory |   epochs |   time-per-epoch |
|--------------|----------------|----------------|-------------|-------------|-------------|------------|------------|------------|----------|----------|------------------|
| Dense        |            768 | --             |       96.7  |       79.72 |       64.24 |      77.17 |      58.55 |      45.11 |    10.04 |     25.6 |             1744 |
| Unstructured |             32 | --             |       78.27 |       54.69 |       39.82 |      65.25 |      43.73 |      31.4  |     4.79 |    100   |             3870 |
| FFI          |             32 | --             |       69.09 |       51.79 |       40.44 |      58.67 |      41.98 |      32.2  |     0.72 |     59.4 |              715 |
| Unstructured |             32 | 32k            |       92.41 |       73.53 |       58.13 |      73.7  |      54.75 |      42.01 |     4.91 |     58   |             4423 |
| FFI          |             32 | 16k            |       90.71 |       71.44 |       56.32 |      73.12 |      54.15 |      41.52 |     0.92 |     68   |              746 |
| FFI          |             32 | 32k            |       92.96 |       74.31 |       58.89 |      73.65 |      54.78 |      42.06 |     1.02 |     67.4 |              842 |
| FFI          |             32 | 65k            |       94.35 |       76.31 |       60.84 |      74.05 |      55.42 |      42.63 |     1.57 |     56   |              928 |
| FFI          |             32 | 100k           |       94.45 |       76.57 |       61.14 |      74.32 |      55.76 |      42.95 |     2.36 |     49   |             1262 |
| FFI          |             64 | 16k            |       94.08 |       76.01 |       60.4  |      74.39 |      55.63 |      42.75 |     1.67 |     56   |              878 |
| FFI          |             64 | 32k            |       94.77 |       77.05 |       61.48 |      74.39 |      55.81 |      42.93 |     1.9  |     48   |              929 |
| FFI          |             64 | 65k            |       95.28 |       77.83 |       62.32 |      74.51 |      56.05 |      43.19 |     2.03 |     43   |             1167 |
| FFI          |             64 | 100k           |       95.84 |       78.64 |       63.14 |      74.63 |      56.24 |      43.37 |     2.65 |     45   |             1530 |
| Bottleneck   |             64 | 64             |       86.43 |       64.81 |       49.5  |      71.9  |      50.7  |      37.93 |     0.96 |     47.6 |              678 |

### Results for Wiki500k with Slice features
| setup        |   connectivity | intermediate   |   train-p@1 |   train-p@3 |   train-p@5 |   test-p@1 |   test-p@3 |   test-p@5 |   memory |   epochs |   time-per-epoch |
|--------------|----------------|----------------|-------------|-------------|-------------|------------|------------|------------|----------|----------|------------------|
| Dense        |            512 | --             |       97.33 |       77.47 |       60.41 |      58.25 |      37.91 |      28.03 |     6.7  |     39.4 |             1249 |
| Unstructured |             32 | --             |       58.33 |       37.7  |       28.07 |      45.49 |      27.29 |      19.86 |     4.79 |     78   |             3612 |
| FFI          |             32 | --             |       42.64 |       28.01 |       21.95 |      37.53 |      23.22 |      17.64 |     0.72 |     54.8 |              659 |
| Unstructured |             32 | 32k            |       83.74 |       61.42 |       47.69 |      59.01 |      38.48 |      28.9  |     4.81 |     40   |             3977 |
| FFI          |             32 | 16k            |       80.26 |       58.19 |       45.12 |      58    |      37.74 |      28.36 |     0.92 |     59   |              946 |
| FFI          |             32 | 32k            |       84.24 |       62.19 |       48.39 |      58.86 |      38.44 |      28.87 |     1.04 |     45.8 |              723 |
| FFI          |             32 | 65k            |       88.46 |       66.83 |       52.23 |      59.83 |      39.24 |      29.48 |     1.18 |     37   |              821 |
| FFI          |             32 | 100k           |       89.83 |       68.51 |       53.69 |      60.56 |      39.79 |      29.93 |     1.59 |     34   |             1106 |
| FFI          |             64 | 16k            |       86.5  |       64.55 |       50.19 |      59.38 |      38.68 |      29.01 |     1.54 |     52   |              807 |
| FFI          |             64 | 32k            |       89.06 |       67.44 |       52.62 |      59.97 |      39.25 |      29.45 |     1.78 |     43   |              843 |
| FFI          |             64 | 65k            |       92.19 |       71.21 |       55.81 |      60.51 |      39.75 |      29.83 |     1.92 |     38   |             1035 |
| FFI          |             64 | 100k           |       93.29 |       72.76 |       57.21 |      61.02 |      40.17 |      30.18 |     2.06 |     38   |             1335 |
| Bottleneck   |             64 | 64             |       71.79 |       50.01 |       38.48 |      56.5  |      36.52 |      27.5  |     0.97 |     41.8 |              639 |


### Preliminary results for Amazon3M with CascadeXML features
| setup      |   connectivity | intermediate   |   train-p@1 |   train-p@3 |   train-p@5 |   test-p@1 |   test-p@3 |   test-p@5 |   memory |   epochs |   time-per-epoch |
|------------|----------------|----------------|-------------|-------------|-------------|------------|------------|------------|----------|----------|------------------|
| Dense      |            768 | --             |       89.89 |       83.94 |       79.44 |      53.36 |      50.65 |      48.38 |    56.36 |       37 |             4228 |
| FFI        |             32 | 65k            |       68.08 |       62.01 |       58.03 |      48.07 |      44.11 |      41.44 |     4.47 |      100 |             1901 |
| FFI        |             32 | 131k           |       71.18 |       64.84 |       60.64 |      49.12 |      45    |      42.3  |     5.01 |      100 |             1902 |
| FFI        |             64 | 131k           |       78.12 |       71.96 |       67.64 |      50.4  |      46.74 |      44.21 |     8.67 |      100 |             2337 |
| FFI        |             96 | 131k           |       82.99 |       76.8  |       72.3  |      51.02 |      47.64 |      45.22 |    12.61 |      100 |             2776 |


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
