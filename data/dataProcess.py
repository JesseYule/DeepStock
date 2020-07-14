import pandas as pd
import numpy as np

output = []

for i in range(900):
    if i % 5 == 0:
        filename = "sz" + str(i) + ".csv"

        data = pd.read_csv(filename)

        tmp = data['Close'].values

        output.append(tmp)

output = np.mat(output)
output = output.T
print(output.shape)

np.savetxt('../data.csv', output, delimiter=',', fmt='%.4f')

