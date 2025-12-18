# Latent-Constrained Autoencoder for Unsupervised Clustering of 4D-STEM Data

Autoencoders are a standard approach for dimensionality reduction and representation learning. In this project, we explore 
whether 128 × 128 diffraction patterns from 4D-STEM data can be encoded into a
compact latent representation. This serves both as a standalone analysis tool
and as a first step toward more advanced models, such as attention-based
architectures for large-scale 4D-STEM analysis.

The learned latent features are analyzed using unsupervised clustering to
identify distinct structural regions in the sample. Initial results showed
fragmented clusters, as the latent space captured not only structural
information but also rotational variations of the diffraction patterns.

To mitigate this effect, we introduce a constraint on the latent space that
reduces sensitivity to rotation, reflecting the underlying physical symmetries
of the data. This leads to more physically meaningful clustering, including
continuous grain-boundary regions with non-zero spatial extent.



