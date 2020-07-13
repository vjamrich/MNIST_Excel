import numpy as np

data = np.load('NPY_Weights_MNIST_Excel.npy')
labels = ['ReLU weights', 'ReLU biases', 'Softmax weights', 'Softmax biases']

newlist = ""

for i in range(len(data)):
    newlist = newlist + "\n" + labels[i] + "\n"
    try:
        for j in range(len(data[i])):
            newlist = newlist + ', '.join(map(str, data[i][j]))
            newlist = newlist + "\n"
    except:
        newlist = newlist + ', '.join(map(str, data[i]))
        newlist = newlist + "\n"

print(newlist)

text_file = open('CSV_Weights_MNIST_Excel.csv', "w")
n = text_file.write(newlist)
text_file.close()
