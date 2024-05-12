import numpy as np
import math

# calculate coeficient of cosine similarity between two vectors
def cosim(vector1, vector2) -> float:
    
    # nominator as a dot product
    nominator = sum([i*j for (i, j) in zip(vector1, vector2)])
    
    # denominator as a product of two magnitudes 
    # call the second function below
    mag1 = magnitude(vector1)    
    mag2 = magnitude(vector2)
    denominator = mag1 * mag2
    
    # divide
    if not denominator:  # we cannot divide if it is null
         sim = 0.0
    else:
         sim = float(nominator)/denominator
    print('Cosine similarity: ', sim)
    return sim

# calculate one magnitude
def magnitude(v) -> float:
    # square() returns the element-wise square of the input
    # math.sqrt() returns sqrt of a number
    mag = math.sqrt(sum(np.square(v))) 
    return mag
