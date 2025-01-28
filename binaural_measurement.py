import record_binaural
import analyze_binaural

subject = "kemar_inner_mic"

record_binaural.initialize()

record_binaural.record_sounds(subject)









df_binaural_measurements[df_binaural_measurements.azimuth == 0].groupby(["subject", "freq"])["ild", "itd"].mean()
df_binaural_measurements.groupby(["subject", "freq", "azimuth"])["ild", "itd"].mean()

g = sns.FacetGrid(data=df_binaural_measurements, row="freq", col="subject", sharey=False)
g.map(sns.lineplot, "azimuth", "rms_left", marker="o", color="g")
g.map(sns.lineplot, "azimuth", "rms_right", marker="o", color="r")
# g.map(sns.lineplot, "azimuth", "ild", marker="o", color="b")
plt.show()

g = sns.FacetGrid(data=df_binaural_measurements, hue="freq", col="subject", palette="copper")
g.map(sns.lineplot, "azimuth", "ild", marker="o")
plt.show()

g = sns.FacetGrid(data=df_binaural_measurements, hue="freq", col="subject", palette="copper")
g.map(sns.lineplot, "azimuth", "itd", marker="o")
g.add_legend()
plt.show()

g = sns.FacetGrid(data=df_binaural_measurements[df_binaural_measurements.freq < 700], hue="subject")
g.map(sns.lineplot, "azimuth", "itd", marker="o")
g.add_legend()
plt.show()

df_sub = df_binaural_measurements[df_binaural_measurements.subject == subject]

fig, ax = plt.subplots(5, 2)
sns.lineplot(data=df_sub, x="azimuth", y="rms_left", hue="freq", ax=ax[0], marker="o")
sns.lineplot(data=df_sub, x="azimuth", y="rms_right", hue="freq", ax=ax[1], marker="o")
plt.show()

sns.lineplot(data=df_sub, x="azimuth", y="ild", hue="freq", marker="o")
plt.show()