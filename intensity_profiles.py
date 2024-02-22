import os
import pickle
import numpy as np
from uncertainties import ufloat
import tqdm
import matplotlib
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy import table

import get_lds_with_errors_v3 as glds


LINECOLOR = matplotlib.rcParams["ytick.color"]


# Compute and store intensity profile interpolators
def intensity_profile_interpolators(star, RF_list, main_dir="",
                                    overwrite=False, nsig=4):
    Teff = star["Teff"]
    logg = star["logg"]
    M_H = star["M_H"]
    vturb = ufloat(2, 0.5) if star["vturb"] is None else star["vturb"]

    name = star["Name"].replace(" ", "_")
    fn = os.path.join(main_dir,
                      "{}_intensity_profile_interpolators.pkl".format(name))
    if not os.path.isfile(fn) or overwrite:
        bounds = {"Teff": (max(0, Teff.n-nsig*Teff.s), Teff.n+nsig*Teff.s),
                  "logg": (logg.n-nsig*logg.s, logg.n+nsig*logg.s),
                  "M_H": (M_H.n-nsig*M_H.s, M_H.n+nsig*M_H.s),
                  "vturb": (max(0, vturb.n-nsig*vturb.s),
                            vturb.n+nsig*vturb.s)}
        glds.download_files(**bounds, force_download=False)
        subgrids = glds.get_subgrids(**bounds)
        ip_interp = glds.get_profile_interpolators(subgrids, RF_list,
                                                   interpolation_order=1,
                                                   atlas_correction=True,
                                                   photon_correction=True,
                                                   max_bad_RF=0.0,
                                                   overwrite_pck=False,
                                                   atlas_hdu=1)
        with open(fn, "wb") as f:
            pickle.dump(ip_interp, f)
    else:
        with open(fn, "rb") as f:
            ip_interp = pickle.load(f)

    return(ip_interp)


# Drawing stellar parameters from normal distributions
def get_samples(star, RF_list, nsamples=10000):
    Teff = star["Teff"]
    logg = star["logg"]
    M_H = star["M_H"]
    vturb = ufloat(2, 0.5) if star["vturb"] is None else star["vturb"]

    vals = np.full((nsamples, 4), np.nan)
    for i, p in enumerate([Teff, logg, M_H, vturb]):
        vals[:, i] = np.random.normal(p.n, p.s, nsamples)
        if (i == 0) or (i == 3):    # removing negative values
            n_max = 100
            j = 0
            while np.any(vals[:, i] < 0):
                idx = np.where(vals[:, i] < 0, True, False)
                vals[:, i][idx] = np.random.normal(p.n, p.s, sum(idx))
                if j >= n_max:
                    raise RuntimeError("failed to draw stellar parameters")
                j += 1
    return(vals)


# Interpolating ATLAS and PHOENIX LD curves
def intensity_profiles(star, ip_interp, samples, main_dir="", save=False):
    intensity_profiles_dict = {}
    RF_list = list(ip_interp.keys())
    pbar = tqdm.tqdm(total=len(RF_list)*(len(ip_interp[RF_list[0]])),
                     desc="Computing intensity profiles", dynamic_ncols=True)
    for RF in RF_list:
        intensity_profiles_RF = {}
        for FT in ip_interp[RF]:
            lndi_mu, lndi_I = ip_interp[RF][FT]
            if lndi_I is None:
                continue
            if FT.startswith("A"):
                mu = lndi_mu
                I = lndi_I(samples)
                mu = np.tile(mu, (len(I), 1))
            elif FT.startswith("P"):
                mu = lndi_mu(samples[:, :-1])
                I = lndi_I(samples[:, :-1])
            idx = np.isfinite(mu) & np.isfinite(I)
            mu = mu[idx]
            I = I[idx]
            idx = np.argsort(mu)
            mu = mu[idx]
            I = I[idx]
            intensity_profiles_RF[FT] = (mu, I)
            pbar.update()
        intensity_profiles_dict[RF] = intensity_profiles_RF
        
        if save:
            # saving intensity profiles
            rf_name = os.path.splitext(os.path.basename(RF))[0]
            fn = os.path.join(main_dir,
                              "{}_intensity_profiles__{}.fits"
                              .format(star["Name"].replace(" ", "_"), rf_name))
            hdu = fits.PrimaryHDU()
            for k, v in star.items():
                hdu.header[k] = str(v)
            hdu.header["RF"] = RF
            hdulist = fits.HDUList([hdu])
            for mn, (mu, I) in intensity_profiles_RF.items():
                tbl = table.Table(data=[mu, I], names=["mu", "Intensity"])
                hdu = fits.table_to_hdu(tbl)
                hdu.name = mn
                hdulist.append(hdu)
            hdulist.writeto(fn, overwrite=True)
            os.system("gzip '{}'".format(fn))
    pbar.close()

    return(intensity_profiles_dict)


# plotting and computing log-likelihood
def ld_linear_model(mu, a):
    return(1.0-a*(1.0-mu))


def ld_square_root_model(mu, s1, s2):
    return(1.0-s1*(1.0-mu)-s2*(1.0-np.sqrt(mu)))


def ld_quadratic_model(mu, u1, u2):
    return(1.0-u1*(1.0-mu)-u2*(1.0-mu)**2)


def ld_kipping2013_model(mu, q1, q2):
    u1 = 2*np.sqrt(q1)*q2
    u2 = np.sqrt(q1)*(1-2*q2)
    return(1.0-u1*(1.0-mu)-u2*(1.0-mu)**2)


def ld_three_parameter_model(mu, b1, b2, b3):
    return(1.0-b1*(1.0-mu)-b2*(1.0-np.sqrt(mu)**3)-b3*(1-mu**2))


def ld_non_linear_model(mu, c1, c2, c3, c4):
    return(1.0-c1*(1.0-np.sqrt(mu))-c2*(1.0-mu)-c3*(1.0-np.sqrt(mu)**3)
           - c4*(1-mu**2))


def ld_logarithmic_model(mu, l1, l2):
    return(1.0-l1*(1.0-mu)-l2*mu*np.log(mu))


def ld_exponential_model(mu, e1, e2):
    return(1.0-e1*(1.0-mu)-e2/(1.0-np.exp(mu)))


def ld_power2_model(mu, p1, p2):
    return(1.0-p1*(1.0-mu**p2))


def chi2(mu, I, u1, u2, ld_model):
    return(np.sum((I-ld_model(mu, u1, u2))**2))


def lnlike(mu, I, u1, u2, fit_func):
    u1_best, u2_best = fit_func(mu, I)
    chi2_min = chi2(mu, I, u1_best.n, u2_best.n)
    chi2_val = chi2(mu, I, u1, u2)
    ll = -0.5*(chi2_val/chi2_min-1)
    return(ll)


ld_laws = {"linear": (ld_linear_model, glds.lds.fit_linear),
           "square-root": (ld_square_root_model, glds.lds.fit_square_root),
           "quadratic": (ld_quadratic_model, glds.lds.fit_quadratic),
           "kipping2013": (ld_kipping2013_model, glds.lds.fit_kipping2013),
           "three-parameter": (ld_three_parameter_model,
                               glds.lds.fit_three_parameter),
           "non-linear": (ld_non_linear_model, glds.lds.fit_non_linear),
           "logarithmic": (ld_logarithmic_model, glds.lds.fit_logarithmic),
           "exponential": (ld_exponential_model, glds.lds.fit_exponential),
           "power-2": (ld_power2_model, glds.lds.fit_power2)}


def plot_intensity_profiles(star, intensity_profiles, ld_law="quadratic",
                            main_dir="", FT_idx=2):
    ld_model, fit_func = ld_laws[ld_law]
    FT_names = ["", "(discarded points with $\\mu \\leq 0.05$ $-$ Sing 2010)",
                "(100-point interpolation $-$ Claret & Bloemen 2011)"]
    fn_suffix = ["", "_mu>0.05", "_interp100"]
    kwargs = dict(s=5, alpha=.2, zorder=1)
    RF_list = list(intensity_profiles.keys())
    for RF in RF_list:
        rf_name = os.path.splitext(os.path.basename(RF))[0]
        fn = ("{}_intensity_profiles__{}{}.png"
              .format(star["Name"].replace(" ", "_"), rf_name,
                      fn_suffix[FT_idx]))
        if rf_name.lower().startswith("uniform"):
            rf_name = ("top-hat filter from {} A to {} A"
                       .format(*rf_name.split("_")[1:]))
        elif rf_name.lower().startswith("gaussian"):
            rf_name = ("gaussian filter centered on {} A (FWHM = {} A)"
                       .format(*rf_name.split("_")[1:]))
        ttl = ("Intensity profile for {}\nwith {}\n"
               .format(star["Name"], rf_name.replace("_", " ")))
        fig, ax = plt.subplots(figsize=(6, 4.5), gridspec_kw=dict(top=0.83))
        fig.suptitle(ttl, fontsize=13)
        fig.text(0.5, 0.87, FT_names[FT_idx], fontsize=10,
                 ha="center", va="center")
        mu_all, I_all = np.array([]), np.array([])
        n = 0
        for FT in intensity_profiles[RF]:
            FT_flags = [FT in ["A17", "P"],  # original data points
                        FT.endswith("S"),    # Sing 2010 (mu ≥ 0.05)
                        FT.endswith("100")]  # Claret & Bloemen 2011 (interp.)
            if not FT_flags[FT_idx]:
                continue
            if "A" in FT:
                mn, c, ls = "ATLAS", "C0", ":"
            elif "P" in FT:
                mn, c, ls = "PHOENIX", "C1", "--"
            else:
                mn, c, ls = "??", "C2", "-."
            mu, I = intensity_profiles[RF][FT]
            mu_all = np.append(mu_all, mu)
            I_all = np.append(I_all, I)
            ldc = [ldc_i.n for ldc_i in fit_func(mu, I)]
            ax.scatter(mu, I, c=c, label="{} data".format(mn), **kwargs)
            lbl_pattern = ("{{}} {{}} model: ({0})"
                           .format(", ".join(["{:.3f}" for ldc_i in ldc])))
            lbl = lbl_pattern.format(mn, ld_law, *ldc)
            ax.plot(mu, ld_model(mu, *ldc), c=LINECOLOR, ls=ls, label=lbl,
                    zorder=100)
            n += 1
        if n > 1:
            ldc = [ldc_i.n for ldc_i in fit_func(mu_all, I_all)]
            lbl_pattern = ("Joint {{}} model: ({0})"
                           .format(", ".join(["{:.3f}" for ldc_i in ldc])))
            lbl = lbl_pattern.format(ld_law, *ldc)
            ax.plot(mu, ld_model(mu, *ldc), c=LINECOLOR, ls="-", label=lbl,
                    zorder=100)
        i = 1.1
        ax.set_ylabel(r"$\frac{\mathcal{I}\left(\mu\right)}{\mathcal{I}_0}$",
                      fontsize=14*i, rotation=0, labelpad=15)
        ax.set_xlabel(r"$\mu=\cos\!\left(\theta\right)$", fontsize=10*i)
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlim(-0.01, 1.01)
        for i, txt in enumerate(["limb", "center"]):
            ax.text(i, -0.11*np.ptp(ax.get_ylim()), "({})".format(txt),
                    ha="center", va="center", fontsize=10, style="italic")
        ax.legend(loc="lower right", fontsize=8)
        fig.savefig(os.path.join(main_dir, fn), dpi=300)

    return(fig)
