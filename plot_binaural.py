import analyze_binaural
import seaborn as sns
import matplotlib.pyplot as plt

def plot_ILD(subject):
    df_binaural_data = analyze_binaural.get_binaural_data(subject)
    g = sns.FacetGrid(data=df_binaural_data, row="freq", sharey=False, palette="crest")
    g.map(sns.lineplot, "azimuth", "rms_right_norm", marker="o")
    g.add_legend()
    plt.show()


def plot_ITD(subject, freqs="all"):
    df_binaural_data = analyze_binaural.get_binaural_data(subject)
    g = sns.FacetGrid(data=df_binaural_data, row="freq", palette="crest")
    g.map(sns.lineplot, "azimuth", "itd_norm", marker="o")
    g.add_legend()
    plt.show()
