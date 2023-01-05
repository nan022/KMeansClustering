import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pandas as pd

st.title(""" Selamat Datang Di Web Clastering Dengan K-Means \n""")

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)



# @st.cache
def data():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv('Social_Network_Ads.csv')
    
    dataset = df
    X = dataset.iloc[:,[3,4]].values
    return X
    
X = data()

# Sidebar
with st.sidebar:
    st.header("Clustering K-Means - Social Network Ads")
    
    klaster_slider = st.slider(
        min_value=1, max_value=6, value=2, label="Jumlah Klaster"
    )
    kmeans = KMeans(n_clusters=klaster_slider, random_state=2023).fit(X)
    labels = kmeans.labels_

    seleksi1 = st.selectbox("Visualisasi Batas Confidence", [False, True])
    seleksi2 = st.selectbox("Jumlah Standar Devsiasi : ", [1,2,3])

    warna = ["red", "seagreen", "orange", "blue", "yellow", "purple"]

    jumlah_label = len(set(labels))

    individu = st.selectbox("Subplot Individu", [False, True])

if individu:
    fig, ax = plt.subplots(ncols=jumlah_label)
else:
    fig, ax = plt.subplots()

for i, yi in enumerate(set(labels)):
    if not individu:
        a = ax
    else:
        a = ax[i]

    xi = X[labels == yi]
    x_pts = xi[:, 0]
    y_pts = xi[:, 1]
    a.scatter(x_pts, y_pts, c=warna[yi])

    if seleksi1:
        confidence_ellipse(
            x=x_pts,
            y=y_pts,
            ax=a,
            edgecolor="black",
            facecolor=warna[yi],
            alpha=0.2,
            n_std=seleksi2,
        )
plt.tight_layout()
st.write(fig)

# Exploratory Dataset
st.subheader ("Exploratory Dataset")
uploaded_file = st.file_uploader("Silahkan Pilih File")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

# Add some matplotlib code !
    fig, ax = plt.subplots()
    df.hist(
        bins=6,
        column="Age",
        grid=False,
        figsize=(2, 2),
        color="#86bf91",
        zorder=1,
        rwidth=0.9,
        ax=ax,
    )
    st.write(fig)