import numpy as np
from np import linalg as LA

O = []
# 輸入方陣A的維度、元，及欲求的次方
axis = int(input('axis: '))
power = int(input('power: '))
print('Please input A (comma-divided)')
for i in range(axis):
  a = list(map(float, input('row ' + str(i + 1) + ': ').split(',')))
  O.append(a)
A = np.array(O)
print('-' * 30)

# 輸出方陣A
print('✦ Square Matrix A\n')
print(A, '\n')

# 計算Eigenvalues及對應的Eigenvectors
eigvals, P = LA.eig(A)
for i in range(axis):
  eigv = P[:, i] / P[:, i][0]
  for j in range(axis):
    P[j][i] = eigv[j]

# 輸出Eigenvalues及對應的Eigenvectors
print('✦ Eigenvalue and Eigenvector corresponding to the Eigenvalue\n')
for i in range(axis):
  print('eigenvalue ' + str(i + 1) + ' :', eigvals[i])
  print('eigenvector ' + str(i + 1) + ':', P[:, i], '\n')

# 輸出方陣P
print('✦ Square Matrix P\n')
print(P, '\n')

# 設定方陣Bn，表示B的n次方
Bn = np.zeros((axis, axis))
for i in range(axis):
  Bn[i][i] = eigvals[i]**power

# 計算A的n次方
print('✦ A to the Power of ' + str(power), '\n')
Ans = np.matmul(np.matmul(P, Bn), LA.inv(P))
print(np.fix(Ans), '\n')

# 以NumPy內建函式驗算
print('✦ Checking: Result Done by np.linalg.matrix_power()\n')
print(LA.matrix_power(A, power))
