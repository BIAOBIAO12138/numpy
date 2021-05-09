import numpy as np

print(np.__version__)
np.show_config()

z=np.zeros(10)
print(z)

z=np.zeros((10,10))
print(z.size*z.itemsize)

np.info(np.add)

z=np.zeros(10)
z[4]=1
print(z)

z=np.arange(10,50)
print(z)

z=np.arange(50)
z=z[::-1]
print(z)

z=np.arange(9).reshape(3,3)
print(z)

nz=np.nonzero([1,2,0,0,4,0])
print(nz)

z=np.eye(3)
print(z)

z=np.random.random((3,3,3))
print(z)

z=np.random.random((10,10)) #random.random生成一个0到1的随机数
zmax,zmin=z.max(),z.min()
print(z.min,z.max)

z=np.random.random(30)
mean=z.mean()
print(mean) #mean函数生成平均值

z=np.ones((10,10))
z[1:-1,1:-1]=0 #行和列从1到-1用0填充
print(z)

z=np.ones((10,10))
z=np.pad(z,pad_width=1,mode='constant',constant_values=0) #np.pad()函数用处很大，用法很多
print(z)

z=np.diag([1,2,3,4],k=-1)#k代表偏移值
print(z)

z=np.zeros((8,8),dtype=int)
z[1::2,::2]=1 #从1开始，数两步，从0开始，数两步
z[::2,1::2]=1
print(z)

print(np.unravel_index(100,(6,7,8)))#unravel_index函数的作用是获取一个的索引值，分为”C"和“F”

z=np.tile(np.array([[1,0],[0,1]]),(4,4))#np.tile函数为沿x轴或y轴扩大相应倍数，比较适合构造有一定规律的矩阵
print(z)

z=np.random.random((5,5))
zmax,zmin=z.max(),z.min()
z=(z-zmin)/(zmax-zmin)#归一化是一种简化计算的方式，即将有量纲的表达式，经过变换，化为无量纲的表达式，成为标量。 在多种计算中都经常用到这种方法。
print(z)

23

z=np.zeros((5,3))@np.zeros((3,2))#矩阵相乘有两种方法，一种为np.dot,另一种为@
print(z)

z=np.arange(11)
z[(3<=z)&(z<8)]*=-1
print(z)

print(sum(range(5),-1))#-1代表列压缩

27

print(np.array(0))

Z = np.random.uniform(-10,+10,10)
print (np.copysign(np.ceil(np.abs(Z)), Z))

Z1 = np.random.randint(0, 10, 10)#np.random.randiant函数产生离散均匀分布的整数
Z2 = np.random.randint(0, 10, 10)
print (np.intersect1d(Z1, Z2))


# np.sqrt(-1) == np.emath.sqrt(-1) 在实数域无法对负数开平方

yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today = np.datetime64('today', 'D')
print(np.datetime64('today', 'D') + np.timedelta64(1, 'D'))

Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print (Z)

A = np.ones(3) * 1
B = np.ones(3) * 1
C = np.ones(3) * 1
np.add(A, B, out=B)
np.divide(A, 2, out=A)
np.negative(A, out=A)
np.multiply(A, B, out=A)
print(A)

36

Z = np.zeros((5, 5))
Z += np.arange(5)#在矩阵上加入[0,5)的随机数
print (Z)

def generate():
    for x in range(10):
      yield x #yield就是 return 返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后开始。
Z = np.fromiter(generate(), dtype=float, count=-1)
print (Z)

Z = np.linspace(0, 1, 12, endpoint=True)[1:-1]#分成12个因为要掐头去尾
print (Z)

Z = np.random.random(10)
Z.sort()
print (Z)

Z = np.arange(10)
print(np.add.reduce(Z))#适用于矩阵

A = np.random.randint(0, 2, 5)
B = np.random.randint(0, 2, 5)
print(A)
equal = np.allclose(A, B)
print(equal)
equal = np.array_equal(A, B)
print(equal)

"""Z = np.zeros(5)
Z.flags.writeable = False
Z[0] = 1"""

Z = np.random.random((10, 2))
X, Y = Z[:, 0], Z[:, 1]
R = np.sqrt(X**2 + Y**2)
T = np.arctan2(Y, X) #np.arctan2是tan反函数
print (R)
print (T)

Z = np.random.random(10)
Z[Z.argmax()] = 0 #返回axis维度的最大值的索引
print (Z)

Z = np.zeros((5, 5), [('x', float), ('y', float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))#生成网格点坐标矩阵
print (Z)

47

for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)

np.set_printoptions(threshold=np.nan)
Z = np.zeros((16,16))
print(Z)

Z = np.arange(100)
v = np.random.uniform(0, 100)
index = (np.abs(Z-v)).argmin()
print(Z[index])

Z = np.zeros(10, [('position', [('x', float, 1),
                                ('y', float, 1)]),
                  ('color',    [('r', float, 1),
                                ('g', float, 1),
                                ('b', float, 1)])])
print (Z)

Z = np.random.random((100, 2))
X, Y = np.atleast_2d(Z[:, 0], Z[:, 1])
D = np.sqrt((X - X.T) ** 2 + (Y - Y.T) ** 2)
print(D)

import scipy.spatial

Z = np.random.random((100, 2))
D = scipy.spatial.distance.cdist(Z, Z)
print(D)






















