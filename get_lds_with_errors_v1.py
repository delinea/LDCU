import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from uncertainties import ufloat
from scipy.special import erf
from scipy.optimize import root_scalar
import get_lds as lds


def cdf_minus_quantile(x, q, distribs, weights):
    cdf = 0
    for d, w in zip(distribs, weights):
        cdf += .5*(1+erf((x-d.n)/d.s/np.sqrt(2)))*w
    cdf /= np.sum(weights)
    return cdf-q


# Merging several distributions
def merge_distrib(distribs, weights=None, nsig=3):
    if weights is None:
        weights = np.ones(len(distribs))
    quantiles = [.5*(1+erf(i/np.sqrt(2)))
                 for i in np.array([-1, 0, 1])*nsig]

    xmin = np.min([d.n-d.s*5 for d in distribs])
    xmax = np.max([d.n+d.s*5 for d in distribs])
    xq = [root_scalar(cdf_minus_quantile, (qt, distribs, weights),
                      bracket=[xmin, xmax], xtol=1e-16).root
          for qt in quantiles]
    mu = xq[1]
    sig_m, sig_p = np.diff(xq)/nsig
    return mu, sig_m, sig_p


def get_lds_with_errors(Teff=None, logg=None, M_H=None, vturb=None, n=10000,
                        RF="cheops_response_function.dat"):
    star = {"Teff": Teff, "grav": logg, "metal": M_H, "vturb": vturb}

    if star["vturb"] is None:
        star["vturb"] = ufloat(2, .5)

    # Drawing stellar parameters from normal distributions
    vals = {pn: np.random.normal(star[pn].n, star[pn].s, n) for pn in star}
    for pn in ["Teff", "vturb"]:
        # remove negative values
        while np.any(vals[pn] < 0):
            idx = np.where(vals[pn] < 0, True, False)
            vals[pn][idx] = np.random.normal(star[pn].n, star[pn].s,
                                             np.sum(idx))

    # Counting the number of model occurences in the ATLAS/PHOENIX grids
    fns_atlas = {}
    fns_phoenix = {}
    for i in tqdm.trange(n, desc="Sampling parameter space",
                         dynamic_ncols=True):
        args = [vals[pn][i] for pn in ["metal", "grav", "Teff", "vturb"]]
        kwargs = dict(verbose=False, verbose_download=True)
        lds_atlas = lds.ATLAS_model_search(*args, **kwargs)
        lds_phoenix = lds.PHOENIX_model_search(*args, **kwargs)
        if lds_atlas[0] in fns_atlas:
            assert lds_atlas == fns_atlas[lds_atlas[0]]["lds"]
            fns_atlas[lds_atlas[0]]["counts"] += 1
        else:
            fns_atlas[lds_atlas[0]] = {"counts": 1, "lds": lds_atlas}
        if lds_phoenix[0] in fns_phoenix:
            assert lds_phoenix == fns_phoenix[lds_phoenix[0]]["lds"]
            fns_phoenix[lds_phoenix[0]]["counts"] += 1
        else:
            fns_phoenix[lds_phoenix[0]] = {"counts": 1, "lds": lds_phoenix}

    # Computing LD coefficients for each model occurence
    kwargs = dict(RF=RF, min_w=None, max_w=None,
                  name="My star", interpolation_order=1,
                  atlas_correction=True, photon_correction=True, verbose=False)

    ldc_atlas_all = {}
    for fn in tqdm.tqdm(fns_atlas, desc="ATLAS LD fit", dynamic_ncols=True):
        path, Teff, grav, metal, vturb = fns_atlas[fn]["lds"]
        ldc_atlas_i = lds.lds(Teff=Teff, grav=grav, metal=metal, vturb=vturb,
                              FT="A100", **kwargs)[0]
        for ldm in ldc_atlas_i:
            if ldm not in ldc_atlas_all:
                ldc_atlas_all[ldm] = [(ldc_atlas_i[ldm],
                                       fns_atlas[fn]["counts"]), ]
            else:
                ldc_atlas_all[ldm].append((ldc_atlas_i[ldm],
                                           fns_atlas[fn]["counts"]))
    ldc_atlas = {}
    for ldm in tqdm.tqdm(ldc_atlas_all, desc="ATLAS LD coeff",
                         dynamic_ncols=True):
        ldc_atlas[ldm] = []
        n_ld = len(ldc_atlas_all[ldm][0][0])
        for i in range(n_ld):
            distribs = [ldc[0][i] for ldc in ldc_atlas_all[ldm]]
            weights = [ldc[1] for ldc in ldc_atlas_all[ldm]]
            ldc_atlas[ldm].append(merge_distrib(distribs, weights))
            continue
            plt.figure()
            plt.title("ATLAS: {} coeff {}".format(ldm, i+1))
            for d, w in zip(distribs, weights):
                plt.errorbar(w, d.n, d.s, marker='.')
            plt.axhline(ldc_atlas[ldm][i][0], color="k", ls="--")
            plt.axis(plt.axis())
            plt.fill_between([plt.axis()[0], plt.axis()[1]],
                             ldc_atlas[ldm][i][0]-ldc_atlas[ldm][i][1],
                             ldc_atlas[ldm][i][0]+ldc_atlas[ldm][i][2],
                             fc="k", alpha=.2)

    ldc_phoenix_all = {}
    for fn in tqdm.tqdm(fns_phoenix, desc="PHOENIX LD fit",
                        dynamic_ncols=True):
        path, Teff, grav, metal, vturb = fns_phoenix[fn]["lds"]
        ldc_phoenix_i = lds.lds(Teff=Teff, grav=grav, metal=metal, vturb=vturb,
                                FT="P100", **kwargs)[0]
        for ldm in ldc_phoenix_i:
            if ldm not in ldc_phoenix_all:
                ldc_phoenix_all[ldm] = [(ldc_phoenix_i[ldm],
                                         fns_phoenix[fn]["counts"]), ]
            else:
                ldc_phoenix_all[ldm].append((ldc_phoenix_i[ldm],
                                             fns_phoenix[fn]["counts"]))
    ldc_phoenix = {}
    for ldm in tqdm.tqdm(ldc_phoenix_all, desc="PHOENIX LD coeff",
                         dynamic_ncols=True):
        ldc_phoenix[ldm] = []
        n_ld = len(ldc_phoenix_all[ldm][0][0])
        for i in range(n_ld):
            distribs = [ldc[0][i] for ldc in ldc_phoenix_all[ldm]]
            weights = [ldc[1] for ldc in ldc_phoenix_all[ldm]]
            ldc_phoenix[ldm].append(merge_distrib(distribs, weights))
            continue
            plt.figure()
            plt.title("PHOENIX: {} coeff {}".format(ldm, i+1))
            for d, w in zip(distribs, weights):
                plt.errorbar(w, d.n, d.s, marker='.')
            plt.axhline(ldc_phoenix[ldm][i][0], color="k", ls="--")
            plt.axis(plt.axis())
            plt.fill_between([plt.axis()[0], plt.axis()[1]],
                             ldc_phoenix[ldm][i][0]-ldc_phoenix[ldm][i][1],
                             ldc_phoenix[ldm][i][0]+ldc_phoenix[ldm][i][2],
                             fc="k", alpha=.2)

    ldc_merged = {}
    for ldm in tqdm.tqdm(ldc_atlas, desc="Merging LD coeff",
                         dynamic_ncols=True):
        assert ldm in ldc_phoenix
        ldc_merged[ldm] = []
        n_ld = len(ldc_atlas[ldm])
        for i in range(n_ld):
            distribs = [ufloat(ldc_atlas[ldm][i][0],
                               np.max(ldc_atlas[ldm][i][1:])),
                        ufloat(ldc_phoenix[ldm][i][0],
                               np.max(ldc_phoenix[ldm][i][1:]))]
            ldc_merged[ldm].append(merge_distrib(distribs))
            continue
            plt.figure()
            plt.title("Merged: {} coeff {}".format(ldm, i+1))
            for d, w in zip(distribs, weights):
                plt.errorbar(w, d.n, d.s, marker='.')
            plt.axhline(ldc_merged[ldm][0], color="k", ls="--")
            plt.axis(plt.axis())
            plt.fill_between([plt.axis()[0], plt.axis()[1]],
                             ldc_merged[ldm][0]-ldc_merged[ldm][1],
                             ldc_merged[ldm][0]+ldc_merged[ldm][2],
                             fc="k", alpha=.2)

    ldc = {}
    for ldm in ldc_merged:
        ldc[ldm] = {"ATLAS": ldc_atlas[ldm], "PHOENIX": ldc_phoenix[ldm],
                    "Merged": ldc_merged[ldm]}

    return ldc


if __name__ == "__main__":
    print_result = True
    savefile = None

    # input stellar parameter
    star = {"Teff": ufloat(5261, 60),       # K
            "logg": ufloat(4.47, 0.05),     # cm/s2 (= log g)
            "M_H": ufloat(0.04, 0.05),      # dex (= M/H)
            "vturb": None}                  # km/s
    star = {"Teff": ufloat(6431, 48),       # K
            "logg": ufloat(4.47, 0.13),     # cm/s2 (= log g)
            "M_H": ufloat(0.11, 0.08),      # dex (= M/H)
            "vturb": None}                  # km/s
    star = {"Teff": ufloat(8000, 80),       # K
            "logg": ufloat(3.9, 0.20),      # cm/s2 (= log g)
            "M_H": ufloat(0.29, 0.13),      # dex (= M/H)
            "vturb": ufloat(2.7, 0.3)}      # km/s

    RFs = ["cheops_response_function.dat", ]

    """# TWI computation
    savefile = "results/LD_coefficients_TWI.txt"
    RFs = ["TWI/Bessel_B_bandpass_transmission.dat",
           "cheops_response_function.dat",
           "TWI/PanSTARRS_zs_bandpass_transmission.dat",
           "TESS_response_function_v2.0.dat"]
    star = {"Teff": ufloat(4734, 40),       # K
            "logg": ufloat(4.60, 0.06),     # cm/s2 (= log g)
            "M_H": ufloat(0.05, 0.08),      # dex (= M/H)
            "vturb": None}                  # km/s"""
    # ALE computation
    savefile = "results/LD_coefficients_ALE.txt"
    RFs = ["TESS_response_function_v2.0.dat", ]
    star = {"Teff": ufloat(5767, 127.443),       # K
            "logg": ufloat(4.47047, 0.076629),     # cm/s2 (= log g)
            "M_H": ufloat(-.14, .29),      # dex (= M/H)
            "vturb": None}                  # km/s"""

    if savefile and os.path.exists(savefile):
        raise FileExistsError("file '{}' already exists !".format(savefile))

    ldc = {}
    for rf in RFs:
        print("\n>> RF file: '{}' <<".format(rf))
        # number of sample should be large (typically n ~ 10'000)
        ldc[rf] = get_lds_with_errors(**star, n=20000, RF=rf)

    if print_result or savefile:
        text = ""
        for rf in RFs:
            text += "\n" + 60*"#" + "\n"
            text += "\n>> RF file: '{}' <<".format(rf)
            for ldm in ldc[rf]:
                text += "\n\nLD {} law:".format(ldm)
                for m in ldc[rf][ldm]:
                    coeffs = []
                    for c in ldc[rf][ldm][m]:
                        c_i = "{0:8.6f} +{2:8.6f}/-{1:8.6f}".format(*c)
                        coeffs.append(c_i)
                    txt = ", ".join(c for c in coeffs)
                    text += "\n  - {:8}: {}".format(m, txt)
            text += "\n\n" + 60*"#"
        if print_result:
            print(text)
        if savefile:
            if os.path.exists(savefile):
                raise FileExistsError("file '{}' already exists !"
                                      .format(savefile))
            with open(savefile, "w") as f:
                f.write(text)
