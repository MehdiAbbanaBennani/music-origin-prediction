import autograd.numpy as np
from autograd import grad

def f(x):
  a = x[0]
  b = x[1]
  return np.sum(a)**2 - b


grad_f = grad(f)
a = np.array([float(i) for i in range(5)])
x = (a, 5.3)
grad_f(x)