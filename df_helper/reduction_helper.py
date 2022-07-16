import pandas as pd
from sklearn.decomposition import PCA

def PCA_reduce(features,components):
    pca_n = PCA(n_components=components)
    principalComponents_N = pca_n.fit_transform(features)
    principalDf_N = pd.DataFrame(data = principalComponents_N)
    return principalDf_N

# df = pd.read_csv("../../Training and Testing sets/train_tfidf_features.csv")
# label = df.loc[:,["label"]]
# features_names = [str(i) for i in range(0,5000)]
# features = df.loc[:,features_names]
# print(PCA_reduce(features,100))