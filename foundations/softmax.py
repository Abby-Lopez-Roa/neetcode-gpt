import numpy as np
from numpy.typing import NDArray
import math

class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4) 

        # numStability = math.e ** (z-max(z))

        # sumJ = sum(math.e ** (z[j] - max(z)) for j in range(len(z)))
        
        # return np.round(np.divide(numStability, sumJ), 4)

        return np.round(np.divide(math.e ** (z-max(z)), sum(math.e ** (z[j] - max(z)) for j in range(len(z)))), 4)

