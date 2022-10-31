import numpy as np
from numba import njit
from .FitnessFunction import FitnessFunction, Bounds
import benchmark_functions as bf


## Bowl-Shaped 

#TODO create adjustment fitness function 
#sphere(0,1) + akley(0,1) +



@njit(nogil=True)
def njitSphere(x, opposite):
    evl = sum(x ** 2)
    return - evl if opposite else evl

class Sphere(FitnessFunction):
    def __init__(self, numDimensions:int=2, opposite:bool=False):
        super().__init__(numDimensions=numDimensions, opposite=opposite)

    def evaluate(self, X: np.ndarray) -> float: 
        return njitSphere(X, self.opposite)

    def bounds(self) -> Bounds:
        return Bounds(min=-5.12, max=5.12)
    
#TODO Hypersphere, Hyperellipsoid

# #Many Local minima

@njit(nogil=True)
def njitAckley(x, opposite, a=20, b=0.2, c=2*np.pi):
    n = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum( np.cos( c * x) )
    term1 = - a * np.exp( -b * np.sqrt( sum1/n ) )
    term2 =  -np.exp(sum2/n)
    evl = term1 + term2 + a + np.exp(1)
    return - evl if opposite else evl

class Ackley(FitnessFunction):
    def __init__(self, numDimensions:int=2, opposite:bool=False):
        super().__init__(numDimensions=numDimensions, opposite=opposite)

    def evaluate(self, X: np.ndarray) -> float: 
        return njitAckley(X, self.opposite)

    def bounds(self) -> Bounds:
        return Bounds(min=-32.768, max=32.768)
    
    def minimum(self) -> float: 
        return 0

@njit(nogil=True)
def njitGriewank(x, opposite, fr=4000):
    n = len(x)
    ii = np.arange( 1., n+1 )
    evl = np.sum( x**2 )/fr - np.prod( np.cos( x / np.sqrt(ii) )) + 1
    return - evl if opposite else evl

class Griewank(FitnessFunction):
    def __init__(self, numDimensions:int=2, opposite:bool=False):
        super().__init__(numDimensions=numDimensions, opposite=opposite)

    def evaluate(self, X:np.ndarray) -> float: 
        return njitGriewank(X, self.opposite)

    def bounds(self) -> Bounds:
        return Bounds(min=-600.0, max=600.0)
    
@njit(nogil=True)
def njitRastrigin(x, opposite):
    n = len(x)
    evl = 10 * n + np.sum( x**2 - 10 * np.cos( 2 * np.pi * x ))
    return - evl if opposite else evl

class Rastrigin(FitnessFunction):
    def __init__(self, numDimensions:int=2, opposite:bool=False):
        super().__init__(numDimensions=numDimensions, opposite=opposite)

    def evaluate(self, X: np.ndarray) -> float: 
        return njitRastrigin(X, self.opposite)

    def bounds(self):
        return Bounds(min=-5.12, max=5.12)

@njit(nogil=True)
def njitSchwefel(x, opposite):
    n = len(x)
    evl = 418.9829 * n - np.sum( x * np.sin( np.sqrt( np.abs( x ))))
    return - evl if opposite else evl

class Schwefel(FitnessFunction):
    def __init__(self, numDimensions:int=2, opposite:bool=False):
        super().__init__(numDimensions=numDimensions, opposite=opposite)

    def evaluate(self, X: np.ndarray) -> float: 
        return njitSchwefel(X, self.opposite)

    def bounds(self) -> Bounds:
        return Bounds(min=-500, max=500)
    
#TODO EggHolder, Schaffer2

# Plate-Shaped

#TODO McCormick

#Valley-Shaped

@njit(nogil=True)
def njitRosenbrock(x, opposite):
    n = len(x)
    xi = x[:-1]
    xnext = x[1:]
    evl = np.sum(100*(xnext-xi**2)**2 + (xi-1)**2)
    return - evl if opposite else evl

class Rosenbrock(FitnessFunction):
    def __init__(self, numDimensions:int=2, opposite:bool=False):
        super().__init__(numDimensions=numDimensions, opposite=opposite)

    def evaluate(self, X: np.ndarray) -> float: 
        return njitRosenbrock(X, self.opposite)

    def bounds(self) -> Bounds:
        return Bounds(min=-2.048, max=2.048)

# #Steep Ridges/Drops

@njit(nogil=True)
def njitMichalewicz(x, opposite, m=10):
    ii = np.arange(1,len(x)+1)
    evl = - sum(np.sin(x) * (np.sin(ii*x**2/np.pi))**(2*m))
    return - evl if opposite else evl

class Michalewicz(FitnessFunction):
    def __init__(self, numDimensions:int=2, opposite:bool=False):
        super().__init__(numDimensions=numDimensions, opposite=opposite)

    def evaluate(self, X: np.ndarray) -> float: 
        return njitMichalewicz(X, self.opposite)

    def bounds(self) -> Bounds:
        return Bounds(min=0, max=np.pi)
    
@njit(nogil=True)
def njitDeJong3(x, opposite):
    evl = np.sum(np.floor(x))
    return - evl if opposite else evl

class DeJong3(FitnessFunction):
    def __init__(self,  numDimensions:int=2, opposite:bool=False):
        super().__init__(numDimensions=numDimensions, opposite=opposite)

    def evaluate(self, X: np.ndarray) -> float: 
        return njitDeJong3(X, self.opposite)

    def bounds(self) -> Bounds:
        return Bounds(min=-5.12, max=5.12)

@njit(nogil=True)
def njitDeJong5(x, opposite):
    x1 = x[0]
    x2 = x[1]
    a1 = np.array([-32, -16,  0,  16,  32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32])
    a2 = np.array([-32, -32, -32,-32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32])  
    ii = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    evl = (0.002  + np.sum (1 / (ii + (x1 - a1) ** 6 + (x2 - a2) ** 6 ))) ** -1
    return - evl if opposite else evl

class DeJong5(FitnessFunction):
    def __init__(self, opposite:bool=False):
        super().__init__(numDimensions=2, opposite=opposite)

    def evaluate(self, X: np.ndarray) -> float: 
        return njitDeJong5(X, self.opposite)

    def bounds(self) -> Bounds:
        return Bounds(min=-65.536, max=65.536)

@njit(nogil=True)
def njitEasom(x, opposite):
    x1 = x[0]
    x2 = x[1]
    evl = -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)
    return - evl if opposite else evl

class Easom(FitnessFunction):
    def __init__(self, opposite:bool=False):
        super().__init__(numDimensions=2, opposite=opposite)

    def evaluate(self, X: np.ndarray) -> float: 
        return njitEasom(X, self.opposite)

    def bounds(self) -> Bounds:
        return Bounds(min=-100, max=100)

# #Other
##TODO GoldsteinAndPrice,PichenyGoldsteinAndPrice,StyblinskiTang