This repository contains a series of notebooks developed for 
workshop **Spatial Modeling with Stan**  at [GeoMed 2024](https://www.uhasselt.be/en/events-en/2023-2024/geomed2024).

>The goals of this workshop are twofold:

> * To introduce new users to existing implementations of spatial models in Stan and the corresponding tools and workflow for model validation and comparison.

> * To provide researchers with the necessary understanding of Stan language syntax and computation so that they can develop custom models and extend existing ones.


This repository contains the R and Python Jupyter notebooks, stan models, and datasets.
Files are named "h1" through "h6" (h for "handout").
The HTML versions in this directory are generated from the notebooks in the `r-notebooks` directory.
The directory `python-notebooks` contains their Python counterparts.

* Notebook "h2\_spatial\_data" shows how to work with GIS data using Python packages
`libpysal` and `plotnine` and R packages `sf`, `spdep`, and `ggplot2`.

* Notebook "h3\_stan\_workflow" introduces the basic concept of stepwise model development.
It demonstrates key data transforms and general best practices.

* Notebook "h4\_icar" introduces the Intrinsic Conditional Auto-Regressive model,
the computationally tractible component for spatial smoothing.

* Notebook "h5\_bym2" introduces the BYM2 model, a widely-used model for spatial epidemiology.

* Notebook "h6\_bym2\_multicomp" extends the BYM2 model to larger, more complex maps with
  disconnected regions and islands.

I hope you find these useful, whether learning Stan, spatial modeling, or learning Python from R and vice versa.



