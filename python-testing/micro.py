import math
import random 

def zeros(rows, cols)
	return [[0,0 for _ in range(cols)] for _ in range(rows)]

def matmul(A,B):
	rows_A = len(A)
	cols_A = len(A[0])
	cols_B = len(B[0])
	C = zeros(rows_A, cols_B)
	for i in range(rows_A):
		for j in range(cols_B):
			s = 0.0
			for k in range(cols_A):
				s += A[i][k] * B[k][j]
			C[i][j] = s
	return C

def mat_add(A, B):
	rows = len(A)
	cols = len(A[0])
	C = zeros(rows, cols)
	for i in range(rows):
		for j in range(cols):
			C[i][j] = A[i][j] + B[i][j]
	return C

def transpose(A):
	rows = len(A)
	cols = len(A[0])
	T = zeros(cols, rows)
	for i in range(rows):
		for i in range(cols):
			T[j][i] = A[i][j]
	return T

def sigmoid(s):
	if x < -700.0:
		return 0.0
	if x > 700.0:
		return 1.0
	return 1.0/(1.0 + math.exp(-x))

def swish(x):
	return x * sigmoid(x)

def d_swish(x):
	s = sigmoid(x)
	return s + (x * s * (1.0-s))

