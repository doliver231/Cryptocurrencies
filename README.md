# Cryptocurrencies

## Overview

Accountability Accounting, a prominent investment bank, is interested in offering a new cryptocurrency investment portfolio for its customers. The company, however, is lost in the vast universe of cryptocurrencies. So they’ve requested a report to be created that includes the cryptocurrencies currently on the trading market and how they could be grouped to create a classification system for the new investment. It will need to be processed to fit the machine learning models, and since there is no known output for what we are looking for, it is decided to use unsupervised learning. To group the cryptocurrencies, we decided on a clustering algorithm. We’ll use data visualizations to share relevant findings with the board of Accountability Accounting.

## Resources

* Data Source: [CryptoCompare](https://min-api.cryptocompare.com/data/all/coinlist), [crypto_data.csv](https://github.com/doliver231/Cryptocurrencies/blob/main/crypto_data.csv)
* Unsupervised Machine Learning code: [crypto_clustering.ipynb](https://github.com/doliver231/Cryptocurrencies/blob/main/crypto_clustering.ipynb)
* Software/Languages: Python, Pandas Library, HvPlot Library, Plotly Express, Scikit-Learn, Jupyter Notebook

## Analysis

### Part 1: Preprocessing the Data for PCA

This is the original dataset that was loaded in into a Dataframe:

![Original crypto_df](https://github.com/doliver231/Cryptocurrencies/blob/main/Images/crypto_df.png)

After preprocessing the data (removing rows with null values, filtering columns, and removing unwanted columns):

![Cleaned crypto_df](https://github.com/doliver231/Cryptocurrencies/blob/main/Images/cleaned_crypto_df.png)

We finished preprocessing by using `pd.get_dummies()` function to convert the string-type columns to binary encoded columns for further analysis.

-------------------------------------------------------------------

### Part 2: Reducing Data Dimensions Using PCA

We reduced the data dimensions using Principle Component Analysis (PCA):

```py
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
X_pca = pca.fit_transform(X_scaled)
pcs_df = pd.DataFrame(data=X_pca, columns=["PC 1", "PC 2", "PC 3"], index = df_clean.index)
```

![PCA](https://github.com/doliver231/Cryptocurrencies/blob/main/Images/PCA.png)

-------------------------------------------------------------------

### Part 3: Clustering Cryptocurrencies Using K-means

```py
inertia = []
k = list(range(1, 11))
for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(pcs_df)
    inertia.append(km.inertia_)

elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
plot = df_elbow.hvplot.line(x="k", y="inertia", title="Elbow Curve", xticks=k)
plot
labels = hv.Labels(data=df_elbow, kdims=['k', 'inertia'],vdims='inertia')
plot * labels
```

![Elbow Data](https://github.com/doliver231/Cryptocurrencies/blob/main/Images/ElbowCurve.png)

We can clearly see the elbow in the Elbow Graph is at k = 4. So we will use this value for the KMeans Model:

```py
model = KMeans(n_clusters=4, random_state=0)
model.fit(pcs_df)
predictions = model.predict(pcs_df)
clustered_df["Class"] = model.labels_
```

![Clustered_df](https://github.com/doliver231/Cryptocurrencies/blob/main/Images/ClusteredDF.png)

-------------------------------------------------------------------

### Part 4: Visualizing Cryptocurrencies Results

We created a 3D Scatter plot using Plotly Express using the 3 PCA components:

![3d Plot](https://github.com/doliver231/Cryptocurrencies/blob/main/Images/3DPLOT.png)

We created a table using `hvplot.table()` to visualize the necessary data to be analyzed:

![HV Table](https://github.com/doliver231/Cryptocurrencies/blob/main/Images/hvtable.png)

Finally, using `hvplot.scatter()`, we creating a 2D Scatter plot that displays two features from the dataset, color coded by the four cluster classes:

![Scatter](https://github.com/doliver231/Cryptocurrencies/blob/main/Images/hvScatter.png)



