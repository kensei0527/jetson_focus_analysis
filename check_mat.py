import scipy.io as sio
import numpy as np

data = sio.loadmat('reflect_66.mat')
# data は dict 形式になる

# もし 'reflect_66' というキーで配列が格納されているなら:
reflect_66 = data['reflect_66']

print("reflect_66 shape:", reflect_66.shape)
print("reflect_66 contents:")
print(reflect_66)