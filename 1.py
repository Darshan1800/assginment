from sklearn import datasets
import pandas as pd
import numpy as np

# reading diabets datasets and splittinf into A and B
df = pd.read_csv("diabetes.csv")

df1 = pd.DataFrame(df)
A = df[:500]

B = df[500:200]


Atypes = A.dtypes.to_dict()

Btypes = B.dtypes.to_dict()

print('A EQUALS B',Atypes==Btypes)
from sklearn import datasets
import pandas as pd
import numpy as np

# load iris dataset
iris = datasets.load_iris()


# Since this is a bunch, create a dataframe
iris_df=pd.DataFrame(iris.data)
iris_df['class']=iris.target
iris_df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
iris_df.dropna(how="all", inplace=True) # remove any empty lines


dftypes = df.dtypes.to_dict()

iristypes = iris_df.dtypes.to_dict()

print('two different datasets',dftypes==iristypes)
