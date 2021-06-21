# LDCU

LDCU allows one to compute limb-darkening coefficients and their corresponding uncertainties.
The code is a modified version of the Python code `limb-darkening` implemented by Néstor Espinoza ([Espinoza & Jordán 2015](https://doi.org/10.1093/mnras/stv744)) and available at the following link: https://github.com/nespinoza/limb-darkening.

### Description
LDCU uses two libraries of stellar atmosphere models [ATLAS9](http://kurucz.harvard.edu/grids.html) ([Kurucz 1979](https://doi.org/10.1086/190589)) and [PHOENIX](https://phoenix.astro.physik.uni-goettingen.de) ([Husser et al. 2013](https://doi.org/10.1051/0004-6361/201219058)) to compute stellar intensity profiles for any given instrumental passband. The atmosphere models are selected based on the provided stellar parameters: effective temperature (Teff), surface gravity (logg), metallicity ([M/H]) and microturbulent velocity (vturb). The uncertainties on the stellar parameters are propagated by selecting several models within the uncertainty range and weighting them accordingly. The obtained intensity profiles are then fitted with the following laws:
linear, square-root, quadratic, 3-parameter, non-linear, logarithmic, exponential and power-2.

Inherited from `limb-darkening` code, LDCU performs 3 different fits for each intensity profile and each law. The first one uses all data points computed from the atmosphere models. The second one discards the data points near the limb (mu < 0.05) as done by [Sing 2010](https://doi.org/10.1051/0004-6361/200913675). The third one interpolates the intensity profile with a cubic spline and fits 100 interpolated data points (evenly spaced in mu) as done by [Claret & Bloemen 2011](https://doi.org/10.1051/0004-6361/20111645).

Finally, LDCU provides for each law a series of additional coefficient values by *merging* the outcomes from the previous fits. The *merging* is done assuming normal distributions from the estimated uncertainties and merging them together before recomputing the global uncertainties from quantiles. This approach allows to compute a precise median value while allowing the uncertainties to encompass the whole merged distribution and be more conservative. **Among the several outcomes, the *Merged/ALL* result is the most conservative one as it accounts for all fits and should be used by default**.

### Dependencies
The code has the following known dependencies:
`numpy`, `scipy`, `astropy`, `uncertainties`, `tqdm`

### How to run LDCU
The current version of the code is still under development and poorly documented. The coefficients can nevertheless be computed and displayed using the following commands.
```
import os
from uncertainties import ufloat
import get_lds_with_errors_v3 as glds

# create a dict with your input stellar parameters
star = {"Teff": ufloat(5261, 60),       # K
        "logg": ufloat(4.47, 0.05),     # cm/s2 (= log g)
        "M_H": ufloat(0.04, 0.05),      # dex (= M/H)
        "vturb": None}                  # km/s

# list of response functions (pass bands) to be used
RF_list = ["cheops_response_function.dat", ]

# name of the file in which the LD coefficients are written
savefile = "results/my_results.txt"
if os.path.exists(savefile):
    raise FileExistsError("file '{}' already exists !".format(savefile))

# query the ATLAS and PHOENIX database and build up a grid of available models
#   (to be run only once each)
glds.update_atlas_grid()
glds.update_phoenix_grid()

# compute the limb-darkening coefficients
ldc = glds.get_lds_with_errors(**star, RF=RF_list)

# print and/or store the results
header = glds.get_header(**star)
summary = glds.get_summary(ldc)
print(summary)
if savefile:
    with open(savefile, "w") as f:
        f.write(header + summary)

```

### Future developments
- code documentation !!
- MCMC fit of the intensity profiles
- partial rewriting: wrap in a Python Class, store intermediate products, provide log_likelihood option
