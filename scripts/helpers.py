import numpy as np
from io import StringIO

# Read a text file, convert it to a numpy array, reducing the number of columns
def reduce_cols(data_file, full_dtype, reduced_dtype, blockSize=100000):
    reduced_catalog = np.zeros(blockSize, dtype=reduced_dtype)
    with open(data_file) as f:
        count = 0
        for line in f:
            if line.startswith("#"):
                continue
            row = np.loadtxt(StringIO(line), dtype=full_dtype)
            reduced_catalog[count] = row[list(reduced_catalog.dtype.names)].copy()
            count += 1
            if count % blockSize == 0:
                reduced_catalog.resize(len(reduced_catalog) + blockSize)
                print(count)
    return reduced_catalog[:count]
