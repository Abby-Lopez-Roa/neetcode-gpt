import numpy as np
from numpy.typing import NDArray
import math

class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4) 

        numStability = math.e ** (z-max(z))

        sumJ = 0
        for j in range(len(z)): sumJ += math.e ** ((z[j]) - max(z))
        
        return np.round(np.divide(numStability, sumJ), 4)