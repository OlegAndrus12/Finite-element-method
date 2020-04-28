import numpy as np
import matplotlib.pyplot as plt

def U(x):
    return np.sin(20*x)

def U_diff(x):
  return 20*np.cos(20*x)

def T(x):
    return 1

def b(x):
    return 0

def sigma(x):
    return 0

def F(x):
    return 400*np.sin(20*x) + 20*np.cos(20*x)

def fillMatrixA(func_t, func_b, func_f, x, q):
    n = len(x)
    h = x[1]-x[0]
    matr = list(np.zeros((n,n+1)))
    for i in range(n-1):
        matr[i][i]=1./h*func_t(x[i]-h/2.)+1./h*func_t(x[i]+h/2.)+1./2*func_b(x[i]-h/2.)-1./2*func_b(x[i]+h/2.)
        matr[i][i+1]=-1./h*func_t(x[i]+h/2.)+1./2*func_b(x[i]+h/2.)
        matr[i+1][i]=-1./h*func_t(x[i]+h/2.)-1./2*func_b(x[i]+h/2.)
    for i in range(1,n-1):
        matr[i][n]=h/2.*func_f(x[i]-h/2.)+h/2.*func_f(x[i]+h/2.)
    matr[0][0]= 0
    matr[0][n]= 0
    matr[n-1][n-1] = 1/h * func_t(1-h/2.) + 1/2. * func_b(1-h/2)
    matr[n-1][n] = h/2.*func_f(1-h/2.)-q
    return matr[1:]



def toNormal(a,b,x):
	return (b-a)/2.*x+(a+b)/2.

def gauss(func,k):
	ai = [128./225, (322. + 13 * np.sqrt(70))/900.,(322. + 13 * np.sqrt(70))/900,(322. - 13 * np.sqrt(70))/900.,(322. - 13 * np.sqrt(70))/900.]
	dots = [0,1/3.*np.sqrt(5-2*np.sqrt(10/7.)),-1/3.*np.sqrt(5-2*np.sqrt(10/7.)),1/3.*np.sqrt(5+2*np.sqrt(10/7.)),-1/3.*np.sqrt(5+2*np.sqrt(10/7.))]
	res = 0
	h = 1./k
	prom = [0]
	for i in range(k):
		prom.append(prom[-1]+h)
	for j in range(len(prom[:-1])):
		subres = 0
		for l in range(len(dots)):
			subres += func(toNormal(prom[j],prom[j+1],dots[l]))*ai[l]
		res += subres/k
	return res/2.

def tdm_solver(matr):
	n = len(matr)
	a = np.array([matr[i][i] for i in range(N-1)],dtype=np.float64)
	b = np.array([matr[i][i+1] for i in range(N-1)],dtype=np.float64)
	c = np.array([matr[i][i+2] for i in range(N-2)],dtype=np.float64)
	d = np.array([matr[i][N] for i in range(N-1)],dtype=np.float64)
	x = np.zeros(n)
	for i in range(1, n):
		m = a[i-1]/b[i-1]
		b[i] -= m*c[i-1] 
		d[i] -= m*d[i-1]    
	x[-1] = d[-1]/b[-1]
	for i in range(n-2, -1, -1):
		x[i] = (d[i]-c[i]*x[i+1])/b[i]
	return x

def normUh_sec(Uh):
	return np.sqrt(
		sum([(Uh.qi[i+1] - Uh.qi[i])**2 / Uh.h for i in range(Uh.length-1)])
		+ 
		sum(Uh.qi)**2 * Uh.h / 3)

def normE(func, diff, Uh):
	return np.sqrt(
	abs(gauss(lambda x : (func(x)-Uh.at(x))**2,4)
	+
	sum([((diff(Uh.h*i) - (Uh.qi[i+1] - Uh.qi[i])/Uh.h)**3 - (diff(Uh.h*(i+1)) - (Uh.qi[i+1] - Uh.qi[i])/Uh.h)**3)/3 for i in range(1,Uh.length-1)])))

def norm(func,diff):
	return np.sqrt(gauss(lambda x : func(x)**2 + diff(x)**2,4))

class Uh:
	def __init__(self,qi):
		self.qi = qi
		self.h = 1/(len(qi)-1)
		self.length = len(qi)

	def at(self,x):
		i0 = int(np.floor(round(x/self.h,4)))
		i1 = int(np.ceil(round(x/self.h,4)))
		if i0 == i1:
			return self.qi[i0]
		else:
			return ((self.qi[i0]-self.qi[i1]) * x)/self.h - self.qi[i0]*i0 + self.qi[i1] * i1



N = 40
q = -7.27
plt.subplots_adjust(1,2,3,4)

print("Norm = {}".format(norm(U,U_diff)))
for i in [511,512,513,514,515]:
	ax = plt.subplot(i)
	print("N = {}".format(N))
	lin = np.linspace(0,1,N)
	qi = list(tdm_solver(fillMatrixA(T, b, F, list(np.linspace(0, 1, N)), q)))
	qi.insert(0,0)
	uh1 = Uh(qi)
	print("Max = {}".format(max([(x-y)**2 for x,y in zip([U(x) for x in lin],qi)])))
	res = [U(x) for x in np.linspace(0,1,100)]
	uh = [uh1.at(x) for x in np.linspace(0,1,100)]
	ax.plot(np.linspace(0,1,100),res)
	ax.plot(np.linspace(0,1,100),uh)
	ax.set_title("N = {0}".format(N))
	print("Norm Uh = {}".format(normUh_sec(uh1)))
	print("Norm E = {}".format(normE(U,U_diff,uh1)))
	plt.clf()
	plt.plot(np.linspace(0,1,1000),[uh1.at(x) for x in np.linspace(0,1,1000)])
	N*=2



