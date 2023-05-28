import numpy as np
from numpy import random


Random_array= np.random.randint(0, 101, size=100)

# Central tendancy 
mean =  np.sum(Random_array)/np.size(Random_array)
median = Random_array[49]

# Spreads of the array 
standard_dev = np.std(Random_array)

print("Central tendancy ",mean)
print("Central tendancy ",median)
print("Spreads of the array", standard_dev)
print(Random_array)