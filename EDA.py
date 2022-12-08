import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from sklearn.datasets.samples_generator import make_blobs
#from pandas.tools.plotting import parallel_coordinates


npf = pd.read_csv("../npf_train.csv")
npf["class2"] = 0
npf["class2"] = npf["class2"].where(npf["class4"]=="nonevent",1)
#npf = npf.set_index("date")
npf = npf.drop("id", axis=1)

y = npf["class2"]
X = npf.drop(["class4","class2","partlybad","date"], axis=1)

X_norm = (X - X.min())/(X.max() - X.min())

pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm))

#y = y.reset_index()

plt.scatter(transformed[y==0][0], transformed[y==0][1], label='nonevent', c='red')
plt.scatter(transformed[y==1][0], transformed[y==1][1], label='event', c='blue')

plt.legend()
plt.show()
"""
print(transformed, y)
"""