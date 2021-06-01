import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import uncertainties
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


def get_intensity(Teff=None, grav=None, metal=None, vturb=-1,
                  RF=None, FT=None, interpolation_order=1,
                  atlas_correction=True, photon_correction=True, verbose=True):
    """
    Get the I(mu) curve for a given set of (M/H, Teff, logg, vturb).
    This function is a partial copy from the original 'calc_lds' function.
    """
    if verbose:
        print('\n\t Reading response functions\n\t --------------------------')

    # Get the response file minimum and maximum wavelengths and all the
    # wavelengths and values:
    min_w, max_w, S_wav, S_res = lds.get_response(None, None, RF,
                                                  verbose=verbose)

    ######################################################################
    # IF USING ATLAS MODELS....
    ######################################################################
    if 'A' in FT:
        # Search for best-match ATLAS9 model for the input stellar parameters:
        if verbose:
            print('\n\t ATLAS modelling\n\t ---------------\n'
                  '\t > Searching for best-match Kurucz model ...')
        ATLAS_ms = lds.ATLAS_model_search(metal, grav, Teff, vturb,
                                          verbose=verbose)
        chosen_filename = ATLAS_ms[0]
        chosen_teff, chosen_grav, chosen_met, chosen_vturb = ATLAS_ms[1:]

        # Read wavelengths and intensities (I) from ATLAS models.
        # If model is "A100", it also returns the interpolated
        # intensities (I100) and the associated mu values (mu100).
        # If not, those arrays are empty:
        wavelengths, I, mu = lds.read_ATLAS(chosen_filename, FT)

        # Now use these intensities to obtain the (normalized) integrated
        # intensities with the response function:
        I0 = lds.integrate_response_ATLAS(wavelengths, I, mu, S_res, S_wav,
                                          atlas_correction, photon_correction,
                                          interpolation_order)

        # Finally, obtain the limb-darkening coefficients:
        if FT == "AS":
            idx = mu >= 0.05    # Select indices as in Sing (2010)
        else:
            idx = mu >= 0.0     # Select all

    ######################################################################
    # IF USING PHOENIX MODELS....
    ######################################################################
    elif 'P' in FT:
        # Search for best-match PHOENIX model for the input stellar parameters:
        if verbose:
            print('\n\t PHOENIX modelling\n\t -----------------\n'
                  '\t > Searching for best-match PHOENIX model ...')
        PHOENIX_ms = lds.PHOENIX_model_search(metal, grav, Teff, vturb,
                                              verbose=verbose)
        chosen_path = PHOENIX_ms[0]
        chosen_teff, chosen_grav, chosen_met, chosen_vturb = PHOENIX_ms[1:]

        # Read PHOENIX model wavelenghts, intensities and mus:
        wavelengths, I, mu = lds.read_PHOENIX(chosen_path)

        # Now use these intensities to obtain the (normalized) integrated
        # intensities with the response function:
        I0 = lds.integrate_response_PHOENIX(wavelengths, I, mu, S_res, S_wav,
                                            photon_correction,
                                            interpolation_order)

        # Obtain correction due to spherical extension. First, get r_max:
        r, fine_r_max = lds.get_rmax(mu, I0)

        # Now get r for each intensity point and leave out those that have r>1:
        new_r = r/fine_r_max
        idx_new = new_r <= 1.0
        new_r = new_r[idx_new]
        # Reuse variable names:
        mu = np.sqrt(1.0-(new_r**2))
        I0 = I0[idx_new]

        # Now, if the model requires it, obtain 100-mu points interpolated
        # in this final range of "usable" intensities:
        if FT == 'P100':
            mu, I100 = lds.get100_PHOENIX(wavelengths, I, mu, idx_new)
            I0 = lds.integrate_response_PHOENIX(wavelengths, I100, mu,
                                                S_res, S_wav,
                                                photon_correction,
                                                interpolation_order)

        # Now define each possible model and fit LDs:
        if FT == 'PQS':      # Quasi-spherical model (Claret et al. 2012)
            idx = mu >= 0.1
        elif FT == 'PS':     # Sing method
            idx = mu >= 0.05
        else:
            idx = mu >= 0.0

    return mu[idx], I0[idx]


def get_all_ldc(mu, I, Ierr=None, mu_min=None):
    params = (mu, I, Ierr)
    ld_coeffs = {"linear": lds.fit_linear(*params),
                 "square-root": lds.fit_square_root(*params),
                 "quadratic": lds.fit_quadratic(*params),
                 "three-parameter": lds.fit_three_parameter(*params),
                 "non-linear": lds.fit_non_linear(*params),
                 "logarithmic": lds.fit_logarithmic(*params),
                 "exponential": lds.fit_exponential(*params, mu_min=mu_min),
                 "power-2": lds.fit_power2(*params)}
    return ld_coeffs


def get_header(name=None, Teff=None, logg=None, M_H=None, vturb=None):
    text = (79*"#" + "\n"
            "#\n# Limb Darkening Calculations {}\n"
            "#\n# Limb-darkening coefficients for linear (a), "
            "square-root (s1, s2),\n"
            "# quadratic (u1, u2), three-parameter (b1, b2, b3),\n"
            "# non-linear (c1, c2, c3, c4), logarithmic (l1, l2), "
            "exponential (e1, e2)\n"
            "# and power-2 laws (p1, p2).\n"
            "#\n# Author:       Nestor Espinoza   (nespino@astro.puc.cl) \n"
            "#\n# Contributors: Benjamin Rackham  "
            "(brackham@email.arizona.com) \n"
            "#               Andres Jordan     (ajordan@astro.puc.cl) \n"
            "#               Ashley Villar     (vvillar@cfa.harvard.edu) \n"
            "#               Patricio Cubillos "
            "(patricio.cubillos@oeaw.ac.at) \n"
            "#\n# DISCLAIMER: If you make use of this code for your "
            "research,\n"
            "#             please consider citing Espinoza & Jordan (2015).\n"
            "#\n# MODIFIED BY:  Adrien Deline  (adrien.deline@unige.ch):\n"
            "#   1) replaced 'wget' URL donwloader by 'curl'\n"
            "#   2) included power-2 limb-darkening law"
            " (Morello et al. 2017)\n"
            "#   3) corrected bug in the computation of closest vturb"
            " for ATLAS models\n"
            "#   4) setting vturb to 2 km/s for [M/H]!=0 in ATLAS"
            " models\n"
            "#   5) added missing -0.5 possible metallicity for PHOENIX"
            " models\n"
            "#   6) changed the LD law fit to return coefficient"
            " uncertainties\n"
            "#   7) added bounds to the possible LD coefficient values to "
            "avoid negative stellar\n"
            "#          intensity or non monotically increasing intensity "
            "toward center\n"
            "#\n" + 79*"#" + "\n")
    text = text.format(lds.VERSION)

    text += ("#\n# Limb-darkening laws\n"
             "#\n# Linear law:\n"
             "#   I(mu)/I(1) = 1 - a * (1 - mu)\n"
             "#\n# Square-root law:\n"
             "#   I(mu)/I(1) = 1 - s1 * (1 - mu) - s2 * (1 - mu^(1/2))\n"
             "#\n# Quadratic law:\n"
             "#   I(mu)/I(1) = 1 - u1 * (1 - mu) - u2 * (1 - mu)^2\n"
             "#\n# Three-parameter law:\n"
             "#   I(mu)/I(1) = 1 - b1 * (1 - mu) - b2 * (1 - mu^(3/2)) "
             "- b3 * (1 - mu^2)\n"
             "#\n# Non-linear law:\n"
             "#   I(mu)/I(1) = 1 - c1 * (1 - mu^(1/2)) - c2 * (1 - mu) "
             "- c3 * (1 - mu^(3/2) - c4 * (1 - mu^2)\n"
             "#\n# Logarithmic law:\n"
             "#   I(mu)/I(1) = 1 - l1 * (1 - mu) - l2 * mu * log(mu)\n"
             "#\n# Exponential law:\n"
             "#   I(mu)/I(1) = 1 - e1 * (1 - mu) - e2 / (1 - exp(mu))\n"
             "#\n# Power-2 law:\n"
             "#   I(mu)/I(1) = 1 - p1 * (1 - mu^p2)\n"
             "#\n" + 79*"#" + "\n")

    n1, n2 = 10, 40
    txt = "# {{:{}}} #".format(n2-4)
    Teff_txt = txt.format("  Teff   = {:6f}  K".format(Teff))
    logg_txt = txt.format("  logg   = {:6f}  cm/s2".format(logg))
    M_H_text = txt.format("  [M/H]  = {:6f}  dex".format(M_H))
    if vturb is None:
        vturb_txt = txt.format("  v_turb = {:15}  km/s".format("      ---"))
    else:
        vturb_txt = txt.format("  v_turb = {:6f}  km/s".format(vturb))
    text += ("\n\n" + n1*" " + n2*"#"
             + "\n" + n1*" " + txt.format("")
             + "\n" + n1*" " + txt.format("Stellar properties")
             + "\n" + n1*" " + txt.format("")
             + "\n" + n1*" " + Teff_txt
             + "\n" + n1*" " + logg_txt
             + "\n" + n1*" " + M_H_text
             + "\n" + n1*" " + vturb_txt
             + "\n" + n1*" " + txt.format("")
             + "\n" + n1*" " + n2*"#" + "\n")

    return text


def get_summary(ldc, fmt="8.6f"):
    text = ""
    for RF in ldc:
        ldc_rf = ldc[RF]
        text += "\n" + 79*"#" + "\n"
        text += "\n>> RF file: '{}' <<".format(RF)
        for ldm in ldc_rf:
            ldc_ldm = ldc_rf[ldm]
            text += "\n\nLD {} law:".format(ldm)
            for FT in ldc_ldm:
                text += "\n >> {:8}:".format(FT)
                ldc_ft = ldc_ldm[FT]
                for m in ldc_ft:
                    coeffs = []
                    for c in ldc_ft[m]:
                        c_txt = "{{:{}}}".format(fmt)
                        if hasattr(c, "__iter__") and len(c) == 3:
                            c_i = c
                            if c_txt.format(c[1]) == c_txt.format(c[2]):
                                c_i = c[:2]
                        elif type(c) is uncertainties.core.Variable:
                            c_i = (c.n, c.s)
                        else:
                            raise
                        if len(c_i) == 2:
                            c_txt = "{{0:{0}}} +/- {{1:{0}}}".format(fmt)
                        elif len(c_i) == 3:
                            c_txt = ("{{0:{0}}} +{{2:{0}}}/-{{1:{0}}}"
                                     .format(fmt))
                        else:
                            raise
                        coeffs.append(c_txt.format(*c_i))
                    txt = ", ".join(c for c in coeffs)
                    text += "\n     - {:8}: {}".format(m, txt)
        text += "\n\n" + 79*"#"
    return text


def get_lds_with_errors(Teff=None, logg=None, M_H=None, vturb=None, n=10000,
                        RF="cheops_response_function.dat"):
    if type(RF) is str:
        RF_list = [RF, ]
    else:
        RF_list = list(RF)

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

    ldc_all = {RF: {} for RF in RF_list}
    for RF in RF_list:
        print("\n>> RF file: '{}' <<".format(RF))
        kwargs = dict(RF=RF, interpolation_order=1, verbose=False,
                      atlas_correction=True, photon_correction=True)

        FT_atlas = ["A17", "A100", "AS"]
        mu_atlas = {FT: np.array([]) for FT in FT_atlas}
        I_atlas = {FT: np.array([]) for FT in FT_atlas}
        Ierr_atlas = {FT: np.array([]) for FT in FT_atlas}
        for fn in tqdm.tqdm(fns_atlas, desc="Computing ATLAS LD curves",
                            dynamic_ncols=True):
            assert fns_atlas[fn]["counts"] > 0
            path, Teff, grav, metal, vturb = fns_atlas[fn]["lds"]
            for FT in FT_atlas:
                mu, I = get_intensity(Teff=Teff, grav=grav, metal=metal,
                                      vturb=vturb, FT=FT, **kwargs)
                Ierr = np.ones_like(I)/np.sqrt(fns_atlas[fn]["counts"])
                mu_atlas[FT] = np.concatenate((mu_atlas[FT], mu))
                I_atlas[FT] = np.concatenate((I_atlas[FT], I))
                Ierr_atlas[FT] = np.concatenate((Ierr_atlas[FT], Ierr))

        ldc_atlas = {}
        for FT in tqdm.tqdm(FT_atlas, desc="Fitting ATLAS LD curves",
                            dynamic_ncols=True):
            ind_sort = np.argsort(mu_atlas[FT])
            mu_atlas[FT] = mu_atlas[FT][ind_sort]
            I_atlas[FT] = I_atlas[FT][ind_sort]
            Ierr_atlas[FT] = Ierr_atlas[FT][ind_sort]
            params = mu_atlas[FT], I_atlas[FT], Ierr_atlas[FT]
            ldc_FT = get_all_ldc(*params)
            for ldm in ldc_FT:
                if ldm in ldc_atlas:
                    if FT in ldc_atlas[ldm]:
                        raise
                    ldc_atlas[ldm][FT] = ldc_FT[ldm]
                else:
                    ldc_atlas[ldm] = {FT: ldc_FT[ldm]}
            """plt.figure()
            plt.title(FT)
            plt.plot(mu_atlas[FT], I_atlas[FT], ".", alpha=.05)
            m = np.linspace(0, 1, 1000)
            plt.plot(m, 1-a.n*(1-m), label="linear")
            plt.plot(m, 1-s1.n*(1-m)-s2.n*(1-np.sqrt(m)), label="sqrt")
            plt.plot(m, 1-u1.n*(1-m)-u2.n*(1-m)**2, label="quad")
            plt.plot(m, 1-b1.n*(1-m)-b2.n*(1-np.sqrt(m)**3)-b3.n*(1-m**2),
                     label="3p")
            plt.plot(m, 1-c1.n*(1-np.sqrt(m))-c2.n*(1-m)-c3.n*(1-np.sqrt(m)**3)
                     - c4.n*(1-m**2), label="nl")
            plt.plot(m, 1-l1.n*(1-m)-l2.n*m*np.log(m), label="log")
            plt.plot(m, 1-e1.n*(1-m)-e2.n/(1-np.exp(m)), label="exp")
            plt.plot(m, 1-p1.n*(1-m**p2.n), label="power2")
            plt.ylim(max(-0.05, plt.ylim()[0]), min(1.05, plt.ylim()[1]))
            plt.grid(True)
            plt.legend()"""

        FT_phoenix = ["P", "PS", "PQS", "P100"]
        mu_phoenix = {FT: np.array([]) for FT in FT_phoenix}
        I_phoenix = {FT: np.array([]) for FT in FT_phoenix}
        Ierr_phoenix = {FT: np.array([]) for FT in FT_phoenix}
        for fn in tqdm.tqdm(fns_phoenix, desc="Computing PHOENIX LD curves",
                            dynamic_ncols=True):
            assert fns_phoenix[fn]["counts"] > 0
            path, Teff, grav, metal, vturb = fns_phoenix[fn]["lds"]
            for FT in FT_phoenix:
                mu, I = get_intensity(Teff=Teff, grav=grav, metal=metal,
                                      vturb=vturb, FT=FT, **kwargs)
                Ierr = np.ones_like(I)/np.sqrt(fns_phoenix[fn]["counts"])
                mu_phoenix[FT] = np.concatenate((mu_phoenix[FT], mu))
                I_phoenix[FT] = np.concatenate((I_phoenix[FT], I))
                Ierr_phoenix[FT] = np.concatenate((Ierr_phoenix[FT], Ierr))

        ldc_phoenix = {FT: {} for FT in FT_phoenix}
        for FT in tqdm.tqdm(FT_phoenix, desc="Fitting PHOENIX LD curves",
                            dynamic_ncols=True):
            ind_sort = np.argsort(mu_phoenix[FT])
            mu_phoenix[FT] = mu_phoenix[FT][ind_sort]
            I_phoenix[FT] = I_phoenix[FT][ind_sort]
            Ierr_phoenix[FT] = Ierr_phoenix[FT][ind_sort]
            params = mu_phoenix[FT], I_phoenix[FT], Ierr_phoenix[FT]
            ldc_FT = get_all_ldc(*params)
            for ldm in ldc_FT:
                if ldm in ldc_phoenix:
                    if FT in ldc_phoenix[ldm]:
                        raise
                    ldc_phoenix[ldm][FT] = ldc_FT[ldm]
                else:
                    ldc_phoenix[ldm] = {FT: ldc_FT[ldm]}
            """plt.figure()
            plt.title(FT)
            plt.plot(mu_phoenix[FT], I_phoenix[FT], ".", alpha=.5, zorder=100)
            m = np.linspace(0, 1, 1000)
            plt.plot(m, 1-a.n*(1-m), label="linear")
            plt.plot(m, 1-s1.n*(1-m)-s2.n*(1-np.sqrt(m)), label="sqrt")
            plt.plot(m, 1-u1.n*(1-m)-u2.n*(1-m)**2, label="quad")
            plt.plot(m, 1-b1.n*(1-m)-b2.n*(1-np.sqrt(m)**3)-b3.n*(1-m**2),
                     label="3p")
            plt.plot(m, 1-c1.n*(1-np.sqrt(m))-c2.n*(1-m)-c3.n*(1-np.sqrt(m)**3)
                     - c4.n*(1-m**2), label="nl")
            plt.plot(m, 1-l1.n*(1-m)-l2.n*m*np.log(m), label="log")
            plt.plot(m, 1-e1.n*(1-m)-e2.n/(1-np.exp(m)), label="exp")
            plt.plot(m, 1-p1.n*(1-m**p2.n), label="power2")
            plt.ylim(max(-0.05, plt.ylim()[0]), min(1.05, plt.ylim()[1]))
            plt.grid(True)
            plt.legend()"""

        """ldms = {"linear": ("$a$",), "square-root": ("$s_1$", "$s_2$"),
                "quadratic": ("$u_1$", "$u_2$"),
                "three-parameter": ("$b_1$", "$b_2$", "$b_3$"),
                "non-linear": ("$c_1$", "$c_2$", "$c_3$", "$c_4$"),
                "logarithmic": ("$l_1$", "$l_2$"),
                "exponential": ("$e_1$", "$e_2$"),
                "power-2": ("$p_1$", "$p_2$")}
        for ldm in ldms:
            n = len(ldms[ldm])
            ncols = min(n, 2)
            nrows = int(np.ceil(n/2))
            fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
            fig.suptitle(ldm)
            axes = np.ravel(axes)
            for FT in FT_atlas:
                for ldc, ax in zip(ldc_atlas[FT][ldm], axes):
                    x = np.linspace(ldc.n-ldc.s*6, ldc.n+ldc.s*6, 200)
                    ax.plot(x, np.exp(-.5*((x-ldc.n)/ldc.s)**2), label=FT)
            for FT in FT_phoenix:
                for ldc, ax in zip(ldc_phoenix[FT][ldm], axes):
                    x = np.linspace(ldc.n-ldc.s*6, ldc.n+ldc.s*6, 200)
                    ax.plot(x, np.exp(-.5*((x-ldc.n)/ldc.s)**2), label=FT)
            for ldc, ax in zip(ldms[ldm], axes):
                ax.legend()
                ax.set_title(ldc)"""

        FT_merged = {"AP": ["A17", "P"], "APS": ["AS", "PS"],
                     "AP100": ["A100", "P100"], "ATLAS": ["A17", "A100", "AS"],
                     "PHOENIX": ["P", "PS", "PQS", "P100"],
                     "ALL": ["A17", "A100", "AS", "P", "PS", "PQS", "P100"]}
        ldc_merged = {ldm: {} for ldm in ldc_atlas}
        for ldm in tqdm.tqdm(ldc_merged, desc="Merging LD coeff",
                             dynamic_ncols=True):
            assert ldm in ldc_phoenix
            ldc_ldm = dict(**ldc_atlas[ldm], **ldc_phoenix[ldm])
            n_ld = [len(ldc) for ldc in ldc_ldm.values()]
            assert np.all(np.diff(n_ld) == 0)
            n_ld = n_ld[0]
            for FT in FT_merged:
                ldc_FT = []
                for i in range(n_ld):
                    distribs = [ldc_ldm[FT_][i] for FT_ in FT_merged[FT]]
                    ldc_FT.append(merge_distrib(distribs))
                    continue
                    plt.figure()
                    plt.title("Merged {}: {} coeff {}".format(FT, ldm, i+1))
                    for i, d in enumerate(distribs):
                        plt.errorbar(i, d.n, d.s, marker='.')
                    plt.axhline(ldc_ldm[-1][0], color="k", ls="--")
                    plt.axis(plt.axis())
                    plt.fill_between([plt.axis()[0], plt.axis()[1]],
                                     ldc_ldm[-1][0]-ldc_ldm[-1][1],
                                     ldc_ldm[-1][0]+ldc_ldm[-1][2],
                                     fc="k", alpha=.2)
                ldc_merged[ldm][FT] = tuple(ldc_FT)

        ldc = {}
        for ldm in ldc_merged:
            ldc[ldm] = {"ATLAS": ldc_atlas[ldm], "PHOENIX": ldc_phoenix[ldm],
                        "Merged": ldc_merged[ldm]}

        ldc_all[RF] = ldc

    return ldc_all
