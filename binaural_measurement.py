import record_binaural
import plot_binaural

subject = "kirke5"

record_binaural.initialize()

record_binaural.record_sounds(subject, n_reps=3)

plot_binaural.plot_ILD(subject)
plot_binaural.plot_ITD(subject)

