from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bone_list = [[1,2], [2,21], [21,3], [3,4], [21,9], [9, 10], [10,11], [11,12], [12,24], [12,25], [21,5], [5,6], [6,7], [7,8], [8,22], [8,23], [1,17], [17,18], [18,19], [19,20], [1,13], [13,14], [14,15], [15,16]]
movement = np.loadtxt("./Dataset/test.txt")
bone_list = np.array(bone_list)-1
number_of_postures = int(len(movement)/25)


for i in range(number_of_postures):
    fig, axis = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    plt.title('Skeleton')
    plt.xlim(-30,30)
    plt.ylim(-110, 60)
    skeleton = movement[i*25:(i+1)*25]

    x = skeleton[:, 0]
    y = skeleton[:, 1]
    z = skeleton[:, 2]

    sc = axis.scatter(x, y, z, c='r', marker='o')

    for bone in bone_list:
        print(y[bone[0]], y[bone[1]])
        plt.plot([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]], [z[bone[0]], z[bone[1]]])

plt.show()