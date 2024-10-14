# ساخت بردار

```python
from numpy import array
# توصیف بردار
v = array([1, 2, 3]) #کاربرد؟
# جمع
c = a + b
# تفریق برداری
c = a - b
# ضرب برداری 
c = a * b
# تقسیم برداری
c = a / b
# ضرب داخلی
c = a.dot(b)
# ضرب بردار اسکالر
c = s * 0.5

# نرم برداری 
from numpy.linalg import norm
c = norm(a, 1) # 1
c = norm(a) # 2
# نرم برداری ماکزیمم
from math import inf
maxnorm = norm(a, inf)

# ساخت ماتریس
A = array([[1, 2, 3], [4, 5, 6]])
# جمع
C = A + B 
# ضرب  داخلی
C = A.dot(B) # = A @ B
# ضرب
C = A * 0.5

# ماتریس همانی - identity matrix
from numpy import identity
I = identity(3)

# معکوس ماتریس
from numpy.linalg import inv
V = inv(Q)
# ماتریس متعامد
I = Q.dot(Q.T)

#Transpose
C = A.T

# Trace
from numpy import trace
B = trace(A)

# دترمینان - Determinant
from numpy.linalg import det
B = det(A)

import numpy as np
np.random.seed(0) # seed -> تولید مجدد دیتا
x1 = np.random.randint(10, size=6) # یک بعدی array
x2 = np.random.randint(10, size=(3, 4)) # دو بعدی array
x3 = np.random.randint(10, size=(3, 4, 5)) # سه بعدی array

print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)
print("dtype:", x3.dtype)
print("itemsize:", x3.itemsize, "bytes")
print("nbytes:", x3.nbytes, "bytes")


x[::-1]# تمام مقادیر برعکس می شوند
x2[:2, :3]# دو سطر، سه ستون
x2[:3, ::2]# تمامی سطرها، ستون ها یکی در میان
x2[::-1, ::-1] #  همه با هم برعکس
print(x2[:, 0]) # اولین ستون x2 - =x2[0]

np.arange(1, 10).reshape((3, 3))
np.concatenate([x, y])
np.vstack([x, grid])
np.hstack([grid, y])
np.split(x, [3, 6])
left, right = np.hsplit(grid, [2])


x = np.arange(4)
print("x =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2)
print("-x = ", -x)
print("x ** 2 = ", x ** 2)
print("x % 2 = ", x % 2)
print(-(0.5*x + 1) ** 2)
np.add(x, 2)

x = [1, 2, 3]
print("x =", x)
print("e^x =", np.exp(x))
print("2^x =", np.exp2(x))
print("3^x =", np.power(3, x))

x = [1, 2, 4, 10]
print("x =", x)
print("ln(x) =", np.log(x))
print("log2(x) =", np.log2(x))
print("log10(x) =", np.log10(x))

x = [0, 0.001, 0.01, 0.1]
print("exp(x) - 1 =", np.expm1(x))
print("log(1 + x) =", np.log1p(x))

shahin= np.random.random(100)

np.min(shahin), np.max(shahin)

M = np.random.random((3, 4))

M.sum()

M.min(axis=0)

M.max(axis=1)

np.sort(x)
# sort each row of X
np.sort(X, axis=1)

```
