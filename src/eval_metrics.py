import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

def lat_long_spread(reconstr_df,X_test):
    fig, ax = plt.subplots(figsize=(10, 7))

    X_test.plot(
        kind="scatter",
        x="Longitude",
        y="Latitude",
        alpha=0.4,
        s=X_test["Population"] / 100,
        color="red",
        label="Actual",
        ax=ax
    )

    reconstr_df.plot(
        kind="scatter",
        x="Longitude",
        y="Latitude",
        alpha=0.3,
        s=reconstr_df["Population"] / 100,
        color="blue",
        label="Reconstructed",
        ax=ax
    )

    plt.title("Geographic Distribution: Actual vs Reconstructed")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()

    plt.show()
    
def plot_histograms(df1, df2):
    fig, axes = plt.subplots(1, len(df1.columns), figsize=(20, 5))
    for i, col in enumerate(df1.columns):
        df1[col].hist(ax=axes[i], alpha=0.5, color='blue', label='Generated')
        df2[col].hist(ax=axes[i], alpha=0.5, color='red', label='Real')
        axes[i].set_title(col)
        axes[i].legend()
    plt.tight_layout()
    plt.show()

def plot_tsne_comparison(X_test, reconstr_df):
    real_data = X_test.values
    synth_data = reconstr_df.values
    combined = np.vstack([real_data, synth_data])
    
    print("Computing t-SNE")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(combined)
    
    real_tsne = tsne_results[:len(real_data), :]
    synth_tsne = tsne_results[len(real_data):, :]
    
    plt.figure(figsize=(10, 7))
    plt.scatter(real_tsne[:, 0], real_tsne[:, 1], c='blue', alpha=0.3, s=10, label='Real Data')
    plt.scatter(synth_tsne[:, 0], synth_tsne[:, 1], c='red', alpha=0.3, s=10, label='Synthetic Data')
    plt.title("t-SNE Overlap: Real vs Synthetic Housing Data")
    plt.legend()
    plt.show()
