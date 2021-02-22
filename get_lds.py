#! /usr/bin/env python
import sys
import os
import numpy as np
import glob
from urllib.request import urlopen
import argparse
import scipy.interpolate as si
from copy import copy

import astropy.io.fits as fits

from scipy.optimize import curve_fit
from uncertainties import ufloat


VERSION = 'v.1.1.deline'

ROOTDIR = os.path.dirname(os.path.realpath(__file__))

ATLAS_DIR = os.path.join(ROOTDIR, "atlas_models")
ATLAS_WEBSITE = "http://kurucz.harvard.edu/grids/"
ATLAS_Z = [-0.1, -0.2, -0.3, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0,
           -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
PHOENIX_DIR = os.path.join(ROOTDIR, "phoenix_models")
PHOENIX_WEBSITE = ("ftp://phoenix.astro.physik.uni-goettingen.de/"
                   "SpecIntFITS/PHOENIX-ACES-AGSS-COND-SPECINT-2011/")
PHOENIX_Z = [-0.0, -0.5, +0.5, -1.0, +1.0, -1.5, -2.0, -3.0, -4.0]
# PHOENIX_DIR = os.path.join(ROOTDIR, "phoenix_v3_models")
# PHOENIX_WEBSITE = ("ftp://phoenix.astro.physik.uni-goettingen.de/"
#                     "v3.0/SpecIntFITS/")
# PHOENIX_Z = [-0.0, -0.5, +0.5, -1.0, +1.0, -1.5, -2.0, -2.5, -3.0, -4.0]


def parse():
    """
    Parse command-line arguments.

    Returns
    -------
    input_filename: String
       Command-line input set by '-ifile'.
    output_filename: String
       Command-line input set by '-ofile'.  Output file where to store results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ifile', default=None)
    parser.add_argument('-ofile', default=None)
    args = parser.parse_args()

    # Set the input/output file names:
    input_filename = args.ifile
    output_filename = args.ofile
    if input_filename is None:
        input_filename = 'example_input_file.dat'
        output_filename = os.path.join(ROOTDIR, "results",
                                       "example_output_file.dat")
    elif output_filename is None:
        ifn, ext = os.path.splitext(os.path.basename(input_filename))
        if "input" in ifn:
            ofn = "output".join(ifn.split("input"))
        else:
            ofn = ifn+"_output"
        output_filename = os.path.join(ROOTDIR, "results", ofn+ext)
    if not os.path.isfile(input_filename):
        ifn = os.path.join(ROOTDIR, "input_files", input_filename)
        if os.path.isfile(ifn):
            input_filename = ifn
        else:
            raise FileNotFoundError("No such file: "
                                    + "'{}'".format(input_filename))

    return input_filename, output_filename


def FixSpaces(intervals):
    s = ''
    i = 0
    while True:
        if intervals[i] == '':
            intervals.pop(i)
        else:
            i = i+1
            if len(intervals) == i:
                break
        if len(intervals) == i:
            break
    for i in range(len(intervals)):
        if i != len(intervals)-1:
            s = s+str(np.double(intervals[i]))+'\t'
        else:
            s = s+str(np.double(intervals[i]))+'\n'
    return s


def getFileLines(fname):
    with open(fname, 'r') as f:
        line = f.readline()
        if line.find('\n') == -1:
            lines = line.split('\r')
        else:
            f.seek(0)
            line = f.read()
            lines = line.split('\n')
    return lines


def getATLASStellarParams(lines):
    for i in range(len(lines)):
        line = lines[i]
        idx = line.find('EFF')
        if idx != -1:
            idx2 = line.find('GRAVITY')
            TEFF = line[idx+4:idx2-1]
            GRAVITY = line[idx2+8:idx2+8+5]
            idx = line.find('L/H')
            if idx == -1:
                LH = '1.25'
            else:
                LH = line[idx+4:]
            break
    return (str(int(np.double(TEFF))), str(np.round(np.double(GRAVITY), 2)),
            str(np.round(np.double(LH), 2)))


def getIntensitySteps(lines):
    for j in range(len(lines)):
        line = lines[j]
        idx = line.find('intervals')
        if idx != -1:
            line = lines[j+1]
            intervals = line.split(' ')
            break

    s = FixSpaces(intervals)
    return j+2, s


def get_derivatives(rP, IP):
    """
    This function calculates the derivatives in an intensity profile I(r).
    For a detailed explaination, see Section 2.2 in Espinoza & Jordan (2015).

    INPUTS:
      rP:   Normalized radii, given by r = sqrt(1-mu**2)
      IP:   Intensity at the given radii I(r).

    OUTPUTS:
      rP:      Output radii at which the derivatives are calculated.
      dI/dr:   Measurement of the derivative of the intensity profile.
    """
    ri = rP[1:-1]   # Points
    rib = rP[:-2]   # Points inmmediately before
    ria = rP[2:]    # Points inmmediately after
    Ii = IP[1:-1]
    Iib = IP[:-2]
    Iia = IP[2:]

    rbar = (ri+rib+ria)/3.0
    Ibar = (Ii+Iib+Iia)/3.0
    num = (ri-rbar)*(Ii-Ibar) + (rib-rbar)*(Iib-Ibar) + (ria-rbar)*(Iia-Ibar)
    den = (ri-rbar)**2 + (rib-rbar)**2 + (ria-rbar)**2

    return rP[1:-1], num/den


def fix_spaces(the_string):
    """
    This function fixes some spacing issues in the ATLAS model files.
    """
    splitted = the_string.split(' ')
    for s in splitted:
        if(s != ''):
            return s
    return the_string


def fit_linear(mu, I, Ierr=None):
    """
    Calculate the coefficients for the linear LD law.
    It assumes input intensities are normalized.
    Least-squares solutions described in Espinoza & Jordan (2015) were
    replaced by a least-square curve fit returning coefficient
    uncertainties.

    INPUTS:
      mu:   Angles at which each intensity is calculated (numpy array).
      I:    Normalized intensities (i.e., I(mu)/I(1)) (numpy array).
      Ierr: Error/uncertinaties on the intensity values (numpy array).

    OUTPUTS:
      a:   Coefficient of the linear law.
    """
    def func(m, *coeffs):
        a, = coeffs
        return 1.0-a*(1.0-m)
    p0 = [.6, ]
    bounds = ((0, ), (1, ))
    popt, pcov = curve_fit(func, mu, I, p0=p0, bounds=bounds)
    psig = np.sqrt(np.diag(pcov))
    ld_coeffs = [ufloat(p, psig[i]) for i, p in enumerate(popt)]
    return ld_coeffs


def fit_square_root(mu, I, Ierr=None):
    """
    Calculates the coefficients for the square-root LD law.
    It assumes input intensities are normalized.
    Least-squares solutions described in Espinoza & Jordan (2015) were
    replaced by a least-square curve fit returning coefficient
    uncertainties.

    INPUTS:
      mu:   Angles at which each intensity is calculated (numpy array).
      I:    Normalized intensities (i.e., I(mu)/I(1)) (numpy array).
      Ierr: Error/uncertinaties on the intensity values (numpy array).

    OUTPUTS:
      s1:   Coefficient of the linear term of the square-root law.
      s2:   Coefficient of the square-root term of the square-root law.
    """
    def func(m, *coeffs):
        s1, s2 = coeffs
        return 1.0-s1*(1.0-m)-s2*(1.0-np.sqrt(m))
    p0 = [.3, .4]
    bounds = ((-1, 0), (1, 2))
    popt, pcov = curve_fit(func, mu, I, sigma=Ierr, p0=p0, bounds=bounds)
    psig = np.sqrt(np.diag(pcov))
    ld_coeffs = [ufloat(p, psig[i]) for i, p in enumerate(popt)]
    return ld_coeffs


def fit_quadratic(mu, I, Ierr=None):
    """
    Calculate the coefficients for the quadratic LD law.
    It assumes input intensities are normalized.
    Least-squares solutions described in Espinoza & Jordan (2015) were
    replaced by a least-square curve fit returning coefficient
    uncertainties.

    INPUTS:
      mu:   Angles at which each intensity is calculated (numpy array).
      I:    Normalized intensities (i.e., I(mu)/I(1)) (numpy array).
      Ierr: Error/uncertinaties on the intensity values (numpy array).

    OUTPUTS:
      u1:   Linear coefficient of the quadratic law.
      u2:   Quadratic coefficient of the quadratic law.
    """
    def func(m, *coeffs):
        u1, u2 = coeffs
        return 1.0-u1*(1.0-m)-u2*(1.0-m)**2
    p0 = [.5, .2]
    bounds = ((0, -1), (2, 1))
    bounds = ((-np.inf, -np.inf), (np.inf, np.inf))
    popt, pcov = curve_fit(func, mu, I, sigma=Ierr, p0=p0, bounds=bounds)
    psig = np.sqrt(np.diag(pcov))
    ld_coeffs = [ufloat(p, psig[i]) for i, p in enumerate(popt)]
    return ld_coeffs


def fit_three_parameter(mu, I, Ierr=None):
    """
    Calculate the coefficients for the three-parameter LD law.
    It assumes input intensities are normalized.
    Least-squares solutions described in Espinoza & Jordan (2015) were
    replaced by a least-square curve fit returning coefficient
    uncertainties.

    INPUTS:
      mu:   Angles at which each intensity is calculated (numpy array).
      I:    Normalized intensities (i.e., I(mu)/I(1)) (numpy array).
      Ierr: Error/uncertinaties on the intensity values (numpy array).

    OUTPUTS:
      b1:   Coefficient of the linear term of the three-parameter law.
      b2:   Coefficient of the (1-mu^{3/2}) part of the three-parameter law.
      b3:   Coefficient of the quadratic term of the three-parameter law.
    """
    def func(m, *coeffs):
        b1, b2, b3 = coeffs
        return 1.0-b1*(1.0-m)-b2*(1.0-np.sqrt(m)**3)-b3*(1-m**2)
    p0 = [1.5, -1.3, 0.5]
    bounds = ((-np.inf, -np.inf, -np.inf),
              (np.inf, np.inf, np.inf))
    popt, pcov = curve_fit(func, mu, I, sigma=Ierr, p0=p0, bounds=bounds)
    psig = np.sqrt(np.diag(pcov))
    ld_coeffs = [ufloat(p, psig[i]) for i, p in enumerate(popt)]
    return ld_coeffs


def fit_non_linear(mu, I, Ierr=None):
    """
    Calculate the coefficients for the non-linear LD law.
    It assumes input intensities are normalized.
    Least-squares solutions described in Espinoza & Jordan (2015) were
    replaced by a least-square curve fit returning coefficient
    uncertainties.

    INPUTS:
      mu:   Angles at which each intensity is calculated (numpy array).
      I:    Normalized intensities (i.e., I(mu)/I(1)) (numpy array).
      Ierr: Error/uncertinaties on the intensity values (numpy array).

    OUTPUTS:
      c1:   Coefficient of the square-root term of the non-linear law.
      c2:   Coefficient of the linear term of the non-linear law.
      c3:   Coefficient of the (1-mu^{3/2}) term of the non-linear law.
      c4:   Coefficient of the quadratic term of the non-linear law.
    """
    def func(m, *coeffs):
        c1, c2, c3, c4 = coeffs
        return (1.0-c1*(1.0-np.sqrt(m))-c2*(1.0-m)-c3*(1.0-np.sqrt(m)**3)
                - c4*(1-m**2))
    p0 = [1.1, -1.8, 2.6, -1.1]
    bounds = ((-np.inf, -np.inf, -np.inf, -np.inf),
              (np.inf, np.inf, np.inf, np.inf))
    popt, pcov = curve_fit(func, mu, I, sigma=Ierr, p0=p0, bounds=bounds)
    psig = np.sqrt(np.diag(pcov))
    ld_coeffs = [ufloat(p, psig[i]) for i, p in enumerate(popt)]
    return ld_coeffs


def fit_logarithmic(mu, I, Ierr=None):
    """
    Calculate the coefficients for the logarithmic LD law.
    It assumes input intensities are normalized.
    Least-squares solutions described in Espinoza & Jordan (2015) were
    replaced by a least-square curve fit returning coefficient
    uncertainties.

    INPUTS:
      mu:   Angles at which each intensity is calculated (numpy array).
      I:    Normalized intensities (i.e., I(mu)/I(1)) (numpy array).
      Ierr: Error/uncertinaties on the intensity values (numpy array).

    OUTPUTS:
      l1:   Coefficient of the linear term of the logarithmic law.
      l2:   Coefficient of the logarithmic term of the logarithmic law.
    """
    def func(m, *coeffs):
        l1, l2 = coeffs
        return 1.0-l1*(1.0-m)-l2*m*np.log(m)
    p0 = [.7, .2]
    bounds = ((0, 0), (1, 1))
    popt, pcov = curve_fit(func, mu, I, sigma=Ierr, p0=p0, bounds=bounds)
    psig = np.sqrt(np.diag(pcov))
    ld_coeffs = [ufloat(p, psig[i]) for i, p in enumerate(popt)]
    return ld_coeffs


def fit_exponential(mu, I, Ierr=None, mu_min=None):
    """
    Calculate the coefficients for the exponential LD law.
    It assumes input intensities are normalized.
    Least-squares solutions described in Espinoza & Jordan (2015) were
    replaced by a least-square curve fit returning coefficient
    uncertainties.

    INPUTS:
      mu:   Angles at which each intensity is calculated (numpy array).
      I:    Normalized intensities (i.e., I(mu)/I(1)) (numpy array).
      Ierr: Error/uncertinaties on the intensity values (numpy array).

    OUTPUTS:
      e1:   Coefficient of the linear term of the exponential law.
      e2:   Coefficient of the exponential term of the exponential law.
    """
    def func(m, *coeffs):
        e1, e2 = coeffs
        return 1.0-e1*(1.0-m)-e2/(1.0-np.exp(m))
    p0 = [.6, -2e-3]
    bounds = ((-np.inf, 1-np.e), (np.inf, np.inf))
    # Bounds to ensure {I > 0 and dI/dµ} for µ > µ_min
    if mu_min is None:
        mu_min = 0.05
    denominator = (1-np.e)**2+np.e*(1-np.exp(mu_min))*(1-mu_min)
    e1_min = np.e*(1-np.exp(mu_min))/denominator
    e2_min = (1-np.e)**2*(1-np.exp(mu_min))/denominator
    e1_max = np.exp(mu_min)/(1-mu_min*np.exp(mu_min))
    e2_max = (1-np.exp(mu_min))**2/(1-mu_min*np.exp(mu_min))
    e1_max = e1_max if e1_max >= 0 else np.inf
    e2_max = e2_max if e2_max >= 0 else np.inf
    bounds = ((e1_min, e2_min), (e1_max, e2_max))
    popt, pcov = curve_fit(func, mu, I, sigma=Ierr, p0=p0, bounds=bounds)
    psig = np.sqrt(np.diag(pcov))
    ld_coeffs = [ufloat(p, psig[i]) for i, p in enumerate(popt)]
    return ld_coeffs


def fit_power2(mu, I, Ierr=None):
    """
    Calculate the coefficients for the power2 LD law.
    It assumes input intensities are normalized.
    Least-squares solutions described in Espinoza & Jordan (2015) were
    replaced by a least-square curve fit returning coefficient
    uncertainties.

    INPUTS:
      mu:   Angles at which each intensity is calculated (numpy array).
      I:    Normalized intensities (i.e., I(mu)/I(1)) (numpy array).
      Ierr: Error/uncertinaties on the intensity values (numpy array).

    OUTPUTS:
      p1:   First coefficient of the power law.
      p2:   Second coefficient (exponent) of the power2 law.
    """
    def func(m, *coeffs):
        p1, p2 = coeffs
        return 1.0-p1*(1.0-m**p2)
    p0 = [.7, .7]
    bounds = ((0, 0), (1, np.inf))
    popt, pcov = curve_fit(func, mu, I, sigma=Ierr, p0=p0, bounds=bounds)
    psig = np.sqrt(np.diag(pcov))
    ld_coeffs = [ufloat(p, psig[i]) for i, p in enumerate(popt)]
    return ld_coeffs


def cmd_exists(cmd, path=None):
    """ test if path contains an executable file with name
    """
    if path is None:
        path = os.environ["PATH"].split(os.pathsep)
    for prefix in path:
        filename = os.path.join(prefix, cmd)
        if not os.path.isfile(filename):
            continue
        executable = os.access(filename, os.X_OK)
        if executable:
            return True
    return False


def downloader_wget(url, verbose=True):
    """
    This function downloads a file from the given url using curl.
    """
    file_name = url.split('/')[-1]
    if verbose:
        print('\t      + Downloading file {:s} from {:s}.'.format(file_name,
                                                                  url))
        os.system('wget '+url)
    else:
        os.system('wget -q '+url)


def downloader_curl(url, verbose=True):
    """
    This function downloads a file from the given url using curl.
    """
    file_name = url.split('/')[-1]
    if verbose:
        print('\t      + Downloading file {:s} from {:s}.'.format(file_name,
                                                                  url))
        os.system('curl -O '+url)
    else:
        os.system('curl -s -O '+url)


if cmd_exists("wget"):
    downloader = downloader_wget
elif cmd_exists("curl"):
    downloader = downloader_curl
else:
    def downloader(*args, **kwargs):
        raise RuntimeError("either 'Wget' or 'cURL' must be installed to "
                           "download file from a given URL")


def ATLAS_model_search(s_met, s_grav, s_teff, s_vturb, verbose=True,
                       verbose_download=None):
    """
    Given input metallicities, gravities, effective temperature and
    microturbulent velocity, this function estimates which model is
    the most appropiate (i.e., the closer one in parameter space).
    If the model is not present in the system, it downloads it from
    Robert L. Kurucz's website (kurucz.harvard.edu/grids.html).
    """
    if verbose_download is None:
        verbose_download = verbose

    # Path to the ATLAS models
    raw_path = os.path.join(ATLAS_DIR, "raw_models")

    if not os.path.exists(ATLAS_DIR):
        os.mkdir(ATLAS_DIR)
        os.mkdir(raw_path)

    # This is the list of all the available metallicities in Kurucz's website:
    possible_mets = np.array(ATLAS_Z, dtype=float)
    # And this is the list of all possible vturbs for Z==0:
    possible_vturb = np.array([0.0, 1.0, 2.0, 4.0, 8.0])

    if s_met not in possible_mets:
        # Check closest metallicity model for input star:
        m_diff = np.inf
        chosen_met = np.inf
        for met in possible_mets:
            # Estimate distance between current and input metallicity:
            c_m_diff = np.abs(met-s_met)
            if(c_m_diff < m_diff):
                chosen_met = met
                m_diff = copy(c_m_diff)
        if verbose:
            print('\t > For input metallicity {}, closest metallicity is {}.'
                  .format(s_met, chosen_met))
    else:
        chosen_met = s_met

    # Check if turbulent velocity is given. If not, set to 2 km/s:
    if s_vturb == -1:
        if verbose:
            print('\t > No known turbulent velocity. Setting it to 2 km/s.')
        s_vturb = 2.0
    if chosen_met != 0:
        # vturb can be different from 2 km/s only for [M/H]=0.0
        chosen_vturb = 2.0
    elif s_vturb not in possible_vturb:
        # Choose closest possible vturb:
        vturb_diff = np.abs(s_vturb-possible_vturb)
        vturb_idx = np.where(vturb_diff == np.min(vturb_diff))
        chosen_vturb = possible_vturb[vturb_idx][0]
        if verbose:
            print('\t > For input vturb {} km/s, closest vturb is '
                  '{} km/s.'.format(s_vturb, chosen_vturb))
    else:
        chosen_vturb = s_vturb

    # Check if the intensity file for the calculated metallicity and
    # vturb is on the atlas_models folder:
    if chosen_met == 0.0:
        met_dir = 'p00'
    elif chosen_met < 0:
        met_string = str(np.abs(chosen_met)).split('.')
        met_dir = 'm'+met_string[0]+met_string[1]
    else:
        met_string = str(np.abs(chosen_met)).split('.')
        met_dir = 'p'+met_string[0]+met_string[1]
    if verbose:
        print('\t    + Checking if ATLAS model file is on the system ...')
    # This will make the code below easier to follow:
    amodel = '{:s}k{:.0f}'.format(met_dir, chosen_vturb)
    afile = os.path.join(raw_path, "i"+amodel)

    if (os.path.exists(afile + 'new.pck')
            or os.path.exists(afile + '.pck19')
            or os.path.exists(afile + '.pck')):
        if verbose:
            print('\t    + Model file found.')
    else:
        # If not in the system, download it from Kurucz's website.
        # First, check all possible files to download:
        if verbose:
            print('\t    + Model file not found.')
        response = urlopen(ATLAS_WEBSITE+"grid"+met_dir+"/")
        html = str(response.read())
        ok = True
        filenames = []
        while(ok):
            idx = html.find('>i'+met_dir.lower())
            if(idx == -1):
                ok = False
            else:
                for i in range(30):
                    if(html[idx+i] == '<'):
                        filenames.append(html[idx+1:idx+i])
            html = html[idx+1:]

        hasnew = False
        gotit = False
        # Check that filenames have the desired vturb and prefer *new* models:
        for afname in filenames:
            if 'new' in afname and amodel in afname:
                hasnew = True
                gotit = True
                downloader(ATLAS_WEBSITE+"grid"+met_dir+"/"+afname,
                           verbose=verbose_download)
                if not os.path.exists(raw_path):
                    os.mkdir(raw_path)
                os.rename(afname, os.path.join(raw_path, afname))

        if not hasnew:
            for afname in filenames:
                if '.pck19' in afname and amodel in afname:
                    gotit = True
                    downloader(ATLAS_WEBSITE+"grid"+met_dir+"/"+afname,
                               verbose=verbose_download)
                    if not os.path.exists(raw_path):
                        os.mkdir(raw_path)
                    os.rename(afname, os.path.join(raw_path, afname))
            if not gotit:
                for afname in filenames:
                    if amodel+'.pck' in afname:
                        gotit = True
                        downloader(ATLAS_WEBSITE+"grid"+met_dir+"/"+afname,
                                   verbose=verbose_download)
                        if not os.path.exists(raw_path):
                            os.mkdir(raw_path)
                        os.rename(afname, os.path.join(raw_path, afname))
        if not gotit:
            if verbose:
                print('\t > No model with closest metallicity of {} and '
                      'closest vturb of {} km/s found.\n\t   Please, modify '
                      'the input values of the target and select other '
                      'stellar parameters for it.'.format(chosen_met,
                                                          chosen_vturb))
            sys.exit()

    # Check if the models in machine readable form have been generated.
    # If not, generate them:
    model_path = os.path.join(ATLAS_DIR, amodel)
    if not os.path.exists(os.path.join(ATLAS_DIR, amodel)):
        # Now read the files and generate machine-readable files:
        possible_paths = [afile+'new.pck', afile+'.pck19', afile+'.pck']

        for j in range(len(possible_paths)):
            possible_path = possible_paths[j]
            if os.path.exists(possible_path):
                lines = getFileLines(possible_path)
                # Create folder for current metallicity and turbulent
                # velocity if not created already:
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                # Save files in the folder:
                while True:
                    TEFF, GRAVITY, LH = getATLASStellarParams(lines)
                    if not os.path.exists(os.path.join(model_path, TEFF)):
                        os.mkdir(os.path.join(model_path, TEFF))
                    idx, mus = getIntensitySteps(lines)
                    save_mr_file = True
                    if os.path.exists(os.path.join(model_path, TEFF,
                                                   "grav_"+GRAVITY
                                                   + "_lh_"+LH+".dat")):
                        save_mr_file = False
                    if save_mr_file:
                        f = open(os.path.join(model_path, TEFF,
                                              "grav_"+GRAVITY
                                              + "_lh_"+LH+".dat"), 'w')
                        f.write('#TEFF:' + TEFF +
                                ' METALLICITY:' + met_dir +
                                ' GRAVITY:' + GRAVITY +
                                ' VTURB:' + str(int(chosen_vturb)) +
                                ' L/H: ' + LH + '\n')
                        f.write('#wav (nm) \t cos(theta):' + mus)
                    for i in range(idx, len(lines)):
                        line = lines[i]
                        idx = line.find('EFF')
                        idx2 = line.find('\x0c')
                        if(idx2 != -1 or line == ''):
                            pass
                        elif(idx != -1):
                            lines = lines[i:]
                            break
                        else:
                            wav_p_intensities = line.split(' ')
                            s = FixSpaces(wav_p_intensities)
                            if save_mr_file:
                                f.write(s+'\n')
                    if save_mr_file:
                        f.close()
                    if(i == len(lines)-1):
                        break

    # Now check closest Teff for input star:
    t_diff = np.inf
    chosen_teff = np.inf
    chosen_teff_folder = ''
    tefffolders = glob.glob(model_path+'/*')
    for tefffolder in tefffolders:
        fname = tefffolder.split('/')[-1]
        teff = np.double(fname)
        c_t_diff = abs(teff-s_teff)
        if(c_t_diff < t_diff):
            chosen_teff = teff
            chosen_teff_folder = tefffolder
            t_diff = c_t_diff
    if verbose:
        print('\t    + For input effective temperature {:.1f} K, closest '
              'value is {:.0f} K.'.format(s_teff, chosen_teff))
    # Now check closest gravity and turbulent velocity:
    grav_diff = np.inf
    chosen_grav = 0.0
    all_files = glob.glob(chosen_teff_folder+'/*')

    for filename in all_files:
        grav = np.double((filename.split('grav')[1]).split('_')[1])
        c_g_diff = abs(grav-s_grav)
        if c_g_diff < grav_diff:
            chosen_grav = grav
            grav_diff = c_g_diff
            chosen_filename = filename

    # Summary:
    model_root_len = len(raw_path)
    if verbose:
        print('\t + For input metallicity {}, effective temperature {} K, '
              'and\n'
              '\t   log-gravity {}, and turbulent velocity {} km/s, closest\n'
              '\t   combination is metallicity: {}, effective temperature: {} '
              'K,\n'
              '\t   log-gravity {} and turbulent velocity of {} km/s.\n\n'
              '\t + Chosen model file to be used:\n\t\t{:s}.\n'
              .format(s_met, s_teff, s_grav, s_vturb, chosen_met, chosen_teff,
                      chosen_grav, chosen_vturb,
                      chosen_filename[model_root_len:]))

    return chosen_filename, chosen_teff, chosen_grav, chosen_met, chosen_vturb


def PHOENIX_model_search(s_met, s_grav, s_teff, s_vturb, verbose=True,
                         verbose_download=None):
    """
    Given input metallicities, gravities, effective temperature and
    microtiurbulent velocity, this function estimates which model is
    the most appropiate (i.e., the closer one in parameter space).
    If the model is not present in the system, it downloads it from
    the PHOENIX public library (phoenix.astro.physik.uni-goettingen.de).
    """
    if verbose_download is None:
        verbose_download = verbose

    # Path to the PHOENIX models
    model_path = os.path.join(PHOENIX_DIR, "raw_models")

    if not os.path.exists(PHOENIX_DIR):
        os.mkdir(PHOENIX_DIR)
        os.mkdir(model_path)

    # In PHOENIX models, all of them are computed with vturb = 2 km/2
    if (s_vturb == -1) and verbose:
        print('\t    + No known turbulent velocity. Setting it to 2 km/s.')
    chosen_vturb = 2.0

    possible_mets = np.array(PHOENIX_Z, dtype=float)

    if s_met not in possible_mets:
        # Now check closest metallicity model for input star:
        m_diff = np.inf
        chosen_met = np.inf
        for met in possible_mets:
            # Estimate distance between current and input metallicity:
            c_m_diff = np.abs(met-s_met)
            if(c_m_diff < m_diff):
                chosen_met = met
                m_diff = copy(c_m_diff)
        if verbose:
            print('\t    + For input metallicity {}, closest value is {}.'.
                  format(s_met, chosen_met))
    else:
        chosen_met = s_met

    # Generate the folder name:
    abs_met = str(np.abs(chosen_met)).split('.')
    if chosen_met <= 0:
        met_folder = 'm'+abs_met[0]+abs_met[1]
        model = 'Z-'+abs_met[0]+'.'+abs_met[1]
    else:
        met_folder = 'p'+abs_met[0]+abs_met[1]
        model = 'Z+'+abs_met[0]+'.'+abs_met[1]

    chosen_met_folder = os.path.join(model_path, met_folder)

    # Check if folder exists. If it does not, create it and download the
    # PHOENIX models that are closer in temperature and gravity to the
    # user input values:
    if not os.path.exists(chosen_met_folder):
        os.mkdir(chosen_met_folder)
    cwd = os.getcwd()
    os.chdir(chosen_met_folder)

    # See if in a past call the file list for the given metallicity was
    # saved; if not, retrieve it from the PHOENIX website:
    if os.path.exists('file_list.dat'):
        with open('file_list.dat') as f:
            all_files = f.readlines()
        for i in np.arange(len(all_files)):
            all_files[i] = all_files[i].strip()
    else:
        response = urlopen(PHOENIX_WEBSITE+model+'/')
        html = str(response.read())
        all_files = []
        while True:
            idx = html.find('lte')
            if(idx == -1):
                break
            else:
                idx2 = html.find('.fits')
                all_files.append(html[idx:idx2+5])
            html = html[idx2+5:]
        f = open('file_list.dat', 'w')
        for file in all_files:
            f.write(file+'\n')
        f.close()
    # Now check closest Teff for input star:
    t_diff = np.inf
    chosen_teff = np.inf
    for file in all_files:
        teff = np.double(file[3:8])
        c_t_diff = abs(teff-s_teff)
        if(c_t_diff < t_diff):
            chosen_teff = teff
            t_diff = c_t_diff

    if verbose:
        print('\t    + For input effective temperature {:.1f} K, closest '
              'value is {:.0f} K.'.format(s_teff, chosen_teff))

    teff_files = []
    teff_string = "{:05.0f}".format(chosen_teff)
    for file in all_files:
        if teff_string in file:
            teff_files.append(file)

    # Now check closest gravity:
    grav_diff = np.inf
    chosen_grav = np.inf
    chosen_fname = ''
    for file in teff_files:
        grav = np.double(file[9:13])
        c_g_diff = abs(grav-s_grav)
        if(c_g_diff < grav_diff):
            chosen_grav = grav
            grav_diff = c_g_diff
            chosen_fname = file

    if verbose:
        print('\t    + Checking if PHOENIX model file is on the system...')
    # Check if file is already downloaded.
    #   If not, download it from the PHOENIX website:
    if not os.path.exists(chosen_fname):
        if verbose:
            print('\t    + Model file not found.')
        downloader(PHOENIX_WEBSITE+model+'/'+chosen_fname,
                   verbose=verbose_download)
    elif verbose:
        print('\t    + Model file found.')

    os.chdir(cwd)
    chosen_path = os.path.join(chosen_met_folder, chosen_fname)

    # Summary:
    if verbose:
        print('\t + For input metallicity {}, effective temperature {} K, '
              'and\n'
              '\t   log-gravity {}, closest combination is metallicity: {},\n'
              '\t   effective temperature: {} K, and log-gravity {}\n\n'
              '\t + Chosen model file to be used:\n\t\t{:s}\n'
              .format(s_met, s_teff, s_grav, chosen_met, chosen_teff,
                      chosen_grav, chosen_fname))

    return chosen_path, chosen_teff, chosen_grav, chosen_met, chosen_vturb


def get_response(min_w, max_w, response_function, verbose=True):
    root = os.path.join(ROOTDIR, "response_functions")
    # Standard response functions:
    if response_function.lower() == 'kphires':
        response_file = os.path.join(root, "standard",
                                     "kepler_response_hires1.txt")
    elif response_function.lower() == 'kplowres':
        response_file = os.path.join(root, "standard",
                                     "kepler_response_lowres1.txt")
    elif response_function.lower() == 'irac1':
        response_file = os.path.join(root, "standard",
                                     "IRAC1_subarray_response_function.txt")
    elif response_function.lower() == 'irac2':
        response_file = os.path.join(root, "standard",
                                     "RAC2_subarray_response_function.txt")
    elif response_function.lower() == 'wfc3':
        response_file = os.path.join(root, "standard",
                                     "WFC3_response_function.txt")
    # User-defined response functions:
    else:
        if os.path.exists(os.path.join(root, response_function)):
            response_file = os.path.join(root, response_function)
        elif os.path.exists(response_function):  # RF not in RF folder:
            response_file = response_function
        else:
            if verbose:
                print("Error: '{:s}' is not valid.".format(response_function))
            sys.exit()
    # Open the response file, which we assume has as first column wavelength
    # and second column the response:
    w, r = np.loadtxt(response_file, unpack=True)
    if('kepler' in response_file):
        w = 10*w
        if min_w is None:
            min_w = min(w)
        if max_w is None:
            max_w = max(w)
        if verbose:
            print('\t > Kepler response file detected.  Switch from '
                  'nanometers to Angstroms.')
            print('\t > Minimum wavelength: {} A.\n'
                  '\t > Maximum wavelength: {} A.'.format(min(w), max(w)))
    elif('IRAC' in response_file):
        w = 1e4*w
        if min_w is None:
            min_w = min(w)
        if max_w is None:
            max_w = max(w)
        if verbose:
            print('\t > IRAC response file detected.  Switch from microns to '
                  'Angstroms.')
            print('\t > Minimum wavelength: {} A.\n'
                  '\t > Maximum wavelength: {} A.'.format(min(w), max(w)))
    else:
        if min_w is None:
            min_w = min(w)
        if max_w is None:
            max_w = max(w)

    # Fit a univariate linear spline (k=1) with s=0 (a node in each point):
    S = si.UnivariateSpline(w, r, s=0, k=1)
    if type(min_w) is list:
        S_wav = []
        S_res = []
        for i in range(len(min_w)):
            c_idx = np.where((w > min_w[i]) & (w < max_w[i]))[0]
            c_S_wav = np.append(np.append(min_w[i], w[c_idx]), max_w[i])
            c_S_res = np.append(np.append(S(min_w[i]), r[c_idx]), S(max_w[i]))
            S_wav.append(np.copy(c_S_wav))
            S_res.append(np.copy(c_S_res))
    else:
        idx = np.where((w > min_w) & (w < max_w))[0]
        S_wav = np.append(np.append(min_w, w[idx]), max_w)
        S_res = np.append(np.append(S(min_w), r[idx]), S(max_w))

    return min_w, max_w, S_wav, S_res


def read_ATLAS(chosen_filename, model):
    # Define the ATLAS grid in mu = cos(theta):
    mu = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25,
                   0.2, 0.15, 0.125, 0.1, 0.075, 0.05, 0.025, 0.01])
    mu100 = np.arange(1.0, 0.0, -0.01)

    # Now prepare files and read data from the ATLAS models:
    with open(chosen_filename, 'r') as f:
        lines = f.readlines()
    # Remove comments and blank lines:
    for i in np.flipud(np.arange(len(lines))):
        if lines[i].strip() == "" or lines[i].strip().startswith("#"):
            lines.pop(i)

    nwave = len(lines)
    wavelengths = np.zeros(nwave)
    intensities = np.zeros((nwave, len(mu)))
    I100 = np.zeros((nwave, len(mu100)))
    for i in np.arange(nwave):
        # If no jump of line or comment, save the intensities:
        splitted = lines[i].split()
        if len(splitted) == 18:
            wavelengths[i] = np.double(splitted[0])*10  # nano to angstrom
            intensities[i] = np.array(splitted[1:], np.double)
            # Only if I(1) is different from zero, fit the LDs:
            if intensities[i, 0] != 0.0:
                # Kurucz doesn't put dots on his files (e.g.: 0.8013 is 8013)
                intensities[i, 1:] = intensities[i, 1:]/1e5
                # Normalzie intensities wrt the first one:
                intensities[i, 1:] = intensities[i, 1:]*intensities[i, 0]
                # If requested, extract the 100 mu-points, with cubic spline
                # interpolation (k=3) through all points (s=0) as CB11:
                if model == 'A100':
                    II = si.UnivariateSpline(mu[::-1], intensities[i, ::-1],
                                             s=0, k=3)
                    I100[i] = II(mu100)

    # Select only those with non-zero intensity:
    flag = intensities[:, 0] != 0.0
    if model == "A100":
        return wavelengths[flag], I100[flag], mu100
    else:
        return wavelengths[flag], intensities[flag], mu


def read_PHOENIX(chosen_path):
    mu = fits.getdata(chosen_path, 'MU')
    data = fits.getdata(chosen_path)
    CDELT1 = fits.getval(chosen_path, 'CDELT1')
    CRVAL1 = fits.getval(chosen_path, 'CRVAL1')
    wavelengths = np.arange(data.shape[1]) * CDELT1 + CRVAL1
    I = data.transpose()
    return wavelengths, I, mu


def integrate_response_ATLAS(wavelengths, I, mu, S_res, S_wav,
                             atlas_correction, photon_correction,
                             interpolation_order):
    # Define the number of mu angles at which we will perform the integrations:
    nmus = len(mu)

    # Integrate intensity through each angle:
    I_l = np.array([])
    for i in range(nmus):
        # Interpolate the intensities:
        Ifunc = si.UnivariateSpline(wavelengths, I[:, i], s=0,
                                    k=interpolation_order)
        # If several wavelength ranges where given, integrate through
        # each chunk one at a time.  If not, integrate the given chunk:
        if type(S_res) is list:
            integration_results = 0.0
            for j in range(len(S_res)):
                if atlas_correction and photon_correction:
                    integrand = (S_res[j]*Ifunc(S_wav[j])) / S_wav[j]
                elif atlas_correction and not photon_correction:
                    integrand = (S_res[j]*Ifunc(S_wav[j])) / (S_wav[j]**2)
                elif not atlas_correction and photon_correction:
                    integrand = (S_res[j]*Ifunc(S_wav[j])) * (S_wav[j])
                else:
                    integrand = S_res[j]*Ifunc(S_wav[j])*S_wav[j]
                integration_results += np.trapz(integrand, x=S_wav[j])
        else:
            if atlas_correction and photon_correction:
                integrand = (S_res*Ifunc(S_wav)) / S_wav
            elif atlas_correction and not photon_correction:
                integrand = (S_res*Ifunc(S_wav)) / (S_wav**2)
            elif not atlas_correction and photon_correction:
                integrand = S_res*Ifunc(S_wav) * S_wav
            else:
                integrand = S_res*Ifunc(S_wav)
            integration_results = np.trapz(integrand, x=S_wav)
        I_l = np.append(I_l, integration_results)

    I0 = I_l/(I_l[0])

    return I0


def integrate_response_PHOENIX(wavelengths, I, mu, S_res, S_wav, correction,
                               interpolation_order):
    I_l = np.array([])
    for i in range(len(mu)):
        Ifunc = si.UnivariateSpline(wavelengths, I[:, i], s=0,
                                    k=interpolation_order)
        if type(S_res) is list:
            integration_results = 0.0
            for j in range(len(S_res)):
                if correction:
                    integrand = S_res[j]*Ifunc(S_wav[j])*S_wav[j]
                else:
                    integrand = S_res[j]*Ifunc(S_wav[j])
                integration_results += np.trapz(integrand, x=S_wav[j])

        else:
            integrand = S_res * Ifunc(S_wav)  # lambda x,I,S: I(x)*S(x)
            if correction:
                integrand *= S_wav            # lambda x,I,S: (I(x)*S(x))*x
            # Integral of Intensity_nu*(Response Function*lambda)*c/lambda**2
            integration_results = np.trapz(integrand, x=S_wav)
        I_l = np.append(I_l, integration_results)

    return I_l/(I_l[-1])


def get_rmax(mu, I0):
    # Apply correction due to spherical extension. First, estimate the r:
    r = np.sqrt(1.0-(mu**2))
    # Estimate the derivatives at each point:
    rPi, m = get_derivatives(r, I0)
    # Estimate point of maximum (absolute) derivative:
    idx_max = np.argmax(np.abs(m))
    r_max = rPi[idx_max]
    # To refine this value, take 20 points to the left and 20 to the right
    # of this value, generate spline and search for roots:
    ndata = 20
    idx_lo = np.max([idx_max-ndata, 0])
    idx_hi = np.min([idx_max+ndata, len(mu)-1])
    r_maxes = rPi[idx_lo:idx_hi]
    m_maxes = m[idx_lo:idx_hi]
    spl = si.UnivariateSpline(r_maxes[::-1], m_maxes[::-1], s=0, k=4)
    fine_r_max = spl.derivative().roots()
    if(len(fine_r_max) > 1):
        abs_diff = np.abs(fine_r_max-r_max)
        iidx_min = np.where(abs_diff == np.min(abs_diff))[0]
        fine_r_max = fine_r_max[iidx_min]
    return r, fine_r_max


def get100_PHOENIX(wavelengths, I, new_mu, idx_new):
    mu100 = np.arange(0.01, 1.01, 0.01)
    I100 = np.zeros((len(wavelengths), len(mu100)))
    for i in range(len(wavelengths)):
        # Cubic splines (k=3), interpolation through all points (s=0) ala CB11.
        II = si.UnivariateSpline(new_mu, I[i, idx_new], s=0, k=3)
        I100[i] = II(mu100)
    return mu100, I100


def calc_lds(name, response_function, model, s_met, s_grav, s_teff,
             s_vturb, min_w=None, max_w=None, atlas_correction=True,
             photon_correction=True, interpolation_order=1, fout=None,
             verbose=True):
    """
    Generate the limb-darkening coefficients.  Note that response_function
    can be a string with the filename of a response function not in the
    list. The file has to be in the response_functions folder.

    Parameters
    ----------
    name: String
       Name of the object we are working on.
    response_function: String
       Number of a standard response function or filename of a response
       function under the response_functions folder.
    model: String
       Fitting technique model.
    s_met: Float
       Metallicity of the star.
    s_grav: Float
       log_g of the star (cgs).
    s_teff: Float
       Effective temperature of the star (K).
    s_vturb: Float
       Turbulent velocity in the star (km/s)
    min_w: Float
       Minimum wavelength to integrate (if None, use the minimum wavelength
       of the response function).
    max_w: Float
       Maximum wavelength to integrate (if None, use the maximum wavelength
       of the response function).
    atlas_correction: Bool
       True if corrections in the integrand of the ATLAS models should
       be applied (i.e., transformation of ATLAS intensities given in
       frequency to per wavelength)
    photon_correction: Bool
       If True, correction for photon-counting devices is used.
    interpolation_order: Integer
       Degree of the spline interpolation order.
    fout: FILE
       If not None, file where to save the LDCs.

    Returns
    -------
    LDC: dict
       The linear (a), square-root (s1, s2), quadratic (u1, u2),
       three-parameter (b1, b2, b3), non-linear (c1, c2, c3, c4),
       logarithmic (l1, l2), exponential (e1, e2), and power-2 laws (p1, p2).
    """
    if verbose:
        print('\n\t Reading response functions\n\t --------------------------')

    # Get the response file minimum and maximum wavelengths and all the
    # wavelengths and values:
    min_w, max_w, S_wav, S_res = get_response(min_w, max_w, response_function,
                                              verbose=verbose)

    ######################################################################
    # IF USING ATLAS MODELS....
    ######################################################################
    if 'A' in model:
        # Search for best-match ATLAS9 model for the input stellar parameters:
        if verbose:
            print('\n\t ATLAS modelling\n\t ---------------\n'
                  '\t > Searching for best-match Kurucz model ...')
        ATLAS_ms = ATLAS_model_search(s_met, s_grav, s_teff, s_vturb,
                                      verbose=verbose)
        chosen_filename = ATLAS_ms[0]
        chosen_teff, chosen_grav, chosen_met, chosen_vturb = ATLAS_ms[1:]

        # Read wavelengths and intensities (I) from ATLAS models.
        # If model is "A100", it also returns the interpolated
        # intensities (I100) and the associated mu values (mu100).
        # If not, those arrays are empty:
        wavelengths, I, mu = read_ATLAS(chosen_filename, model)

        # Now use these intensities to obtain the (normalized) integrated
        # intensities with the response function:
        I0 = integrate_response_ATLAS(wavelengths, I, mu, S_res, S_wav,
                                      atlas_correction, photon_correction,
                                      interpolation_order)

        # Finally, obtain the limb-darkening coefficients:
        if model == "AS":
            idx = mu >= 0.05    # Select indices as in Sing (2010)
        else:
            idx = mu >= 0.0     # Select all

    ######################################################################
    # IF USING PHOENIX MODELS....
    ######################################################################
    elif 'P' in model:
        # Search for best-match PHOENIX model for the input stellar parameters:
        if verbose:
            print('\n\t PHOENIX modelling\n\t -----------------\n'
                  '\t > Searching for best-match PHOENIX model ...')
        PHOENIX_ms = PHOENIX_model_search(s_met, s_grav, s_teff, s_vturb,
                                          verbose=verbose)
        chosen_path = PHOENIX_ms[0]
        chosen_teff, chosen_grav, chosen_met, chosen_vturb = PHOENIX_ms[1:]

        # Read PHOENIX model wavelenghts, intensities and mus:
        wavelengths, I, mu = read_PHOENIX(chosen_path)

        # Now use these intensities to obtain the (normalized) integrated
        # intensities with the response function:
        I0 = integrate_response_PHOENIX(wavelengths, I, mu, S_res, S_wav,
                                        photon_correction, interpolation_order)

        # Obtain correction due to spherical extension. First, get r_max:
        r, fine_r_max = get_rmax(mu, I0)

        # Now get r for each intensity point and leave out those that have r>1:
        new_r = r/fine_r_max
        idx_new = new_r <= 1.0
        new_r = new_r[idx_new]
        # Reuse variable names:
        mu = np.sqrt(1.0-(new_r**2))
        I0 = I0[idx_new]

        # Now, if the model requires it, obtain 100-mu points interpolated
        # in this final range of "usable" intensities:
        if model == 'P100':
            mu, I100 = get100_PHOENIX(wavelengths, I, mu, idx_new)
            I0 = integrate_response_PHOENIX(wavelengths, I100, mu,
                                            S_res, S_wav, photon_correction,
                                            interpolation_order)

        # Now define each possible model and fit LDs:
        if model == 'PQS':      # Quasi-spherical model (Claret et al. 2012)
            idx = mu >= 0.1
        elif model == 'PS':     # Sing method
            idx = mu >= 0.05
        else:
            idx = mu >= 0.0

    # Now compute each LD law:
    a, = fit_linear(mu[idx], I0[idx])
    s1, s2 = fit_square_root(mu[idx], I0[idx])
    u1, u2 = fit_quadratic(mu[idx], I0[idx])
    b1, b2, b3 = fit_three_parameter(mu[idx], I0[idx])
    c1, c2, c3, c4 = fit_non_linear(mu, I0)
    l1, l2 = fit_logarithmic(mu[idx], I0[idx])
    e1, e2 = fit_exponential(mu[idx], I0[idx])
    p1, p2 = fit_power2(mu[idx], I0[idx])

    # Stack all LD coefficients into one single dict:
    LDC = {"linear": (a,),
           "square-root": (s1, s2),
           "quadratic": (u1, u2),
           "three-parameter": (b1, b2, b3),
           "non-linear": (c1, c2, c3, c4),
           "logarithmic": (l1, l2),
           "exponential": (e1, e2),
           "power-2": (p1, p2)}

    # Save to the file:
    if fout is not None:
        fout.write(70*"#" + "\n")
        fout.write("{:s}  {:s}  {:s}\nTeff={:.1f}K  log(g)={:.1f}  "
                   "[M/H]={:.1f}  vturb={:.1f}\n\n".format(name, model,
                                                           response_function,
                                                           chosen_teff,
                                                           chosen_grav,
                                                           chosen_met,
                                                           chosen_vturb))
        cns = {"linear": "a", "square-root": "s", "quadratic": "u",
               "three-parameter": "b", "non-linear": "c", "logarithmic": "l",
               "exponential": "e", "power-2": "p"}
        for ldm in LDC:
            n_ldc = len(LDC[ldm])
            if n_ldc == 0:
                cn = cns[ldm]
            else:
                cn = ", ".join(["{}{}".format(cns[ldm], i+1)
                                for i in range(n_ldc)])
                cv = ", ".join(["{:11.9f}".format(c) for c in LDC[ldm]])
            fout.write("{:16}: {} = {}\n".format(ldm, cn, cv))
        fout.write("\n")

    if verbose:
        print('\t > Done! \n\t {:s}\n'.format(70*'#'))
    return LDC


def lds(Teff=None, grav=None, metal=None, vturb=-1,
        RF=None, FT=None, min_w=None, max_w=None,
        name="", ifile=None, ofile=None,
        interpolation_order=1,
        atlas_correction=True, photon_correction=True, verbose=True):
    """
    Compute limb-darkening coefficients.

    Parameters
    ----------
    Teff: Float
       Effective temperature of the star (K).
    grav: Float
       log_g of the star (cgs).
    metal: Float
       Metallicity of the star.
    vturb: Float
       Turbulent velocity in the star (km/s)
    RF: String
       A standard response function or filename of a response
       function under the response_functions folder.
    FT: String
       Limb-darkening fitting technique model.  Select one or more
       (comma separated, no blank spaces) model from the following list:
          A17:  LDs using ATLAS with all its 17 angles
          A100: LDs using ATLAS models interpolating 100 mu-points with a
                cubic spline (i.e., like Claret & Bloemen, 2011)
          AS:   LDs using ATLAS with 15 angles for linear, quadratic and
                three-parameter laws, bit 17 angles for the non-linear
                law (i.e., like Sing, 2010)
          P:    LDs using PHOENIX models (Husser et al., 2013).
          PS:   LDs using PHOENIX models using the methods of Sing (2010).
          PQS:  LDs using PHOENIX quasi-spherical models (mu>=0.1 only)
          P100: LDs using PHOENIX models and interpolating 100 mu-points
                with cubic spline (i.e., like Claret & Bloemen, 2011)
    min_w: Float
       Minimum wavelength to integrate (if None, use the minimum wavelength
       of the response function).
    max_w: Float
       Maximum wavelength to integrate (if None, use the maximum wavelength
       of the response function).
    name: String
       Name of the object we are working on (to write in ofile).
    ifile: String
       Filename with the user inputs.
    ofile: String
       If not None, filename where to write the LCDs.
    interpolation_order: Integer
       Degree of the spline interpolation order.
    atlas_correction: Bool
       If True, convert ATLAS intensities using c/lambda**2 (ATLAS
       intensities are given per frequency).
    photon_correction: Bool
       If True, apply photon counting correction (lambda/hc).

    Returns
    -------
    LDC: 1D list
       Each element in this list contains a dict of all the LD laws
       for a given parameter set.  The dictionaries of LD laws contain
       the linear (a), square-root (s1, s2), quadratic (u1, u2),
       three-parameter (b1, b2, b3), non-linear (c1, c2, c3, c4),
       logarithmic (l1, l2), exponential (e1, e2), and power-2 laws (p1, p2).

    Example
    -------
    >>> import get_lds as lds
    >>> ldc1 = lds.lds(ifile="input_files/example_input_file.dat")
    >>> ldc2 = lds.lds(5500.0, 4.5, 0.0, -1, "KpHiRes", "A100,P100")
    """
    if verbose:
        print('\n\t ##########################################################'
              '\n\n\t             Limb Darkening Calculations {:s}\n'
              '\n\t      Author: Nestor Espinoza (nespino@astro.puc.cl)\n'
              '\n\t DISCLAIMER: If you make use of this code for your '
              'research,\n'
              '\t please consider citing Espinoza & Jordan (2015)\n'
              '\n\t ##########################################################'
              .format(VERSION))

    if ofile is None:
        fout = None
    else:
        fout = open(ofile, 'w')
        fout.write(70*"#" + "\n"
                   "#\n# Limb Darkening Calculations {}\n"
                   "#\n# Limb-darkening coefficients for linear (a),"
                   " quadratic (u1,u2),\n"
                   "# three-parameter (b1,b2,b3), non-linear (c1,c2,c3,c4),\n"
                   "# logarithmic (l1,l2), exponential (e1,e2),"
                   " square-root (s1,s2)\n"
                   "# and power-2 laws (p1,p2).\n"
                   "#\n# Author:       Nestor Espinoza   "
                   "(nespino@astro.puc.cl) \n"
                   "#\n# Contributors: Benjamin Rackham  "
                   "(brackham@email.arizona.com) \n"
                   "#               Andres Jordan     "
                   "(ajordan@astro.puc.cl) \n"
                   "#               Ashley Villar     "
                   "(vvillar@cfa.harvard.edu) \n"
                   "#               Patricio Cubillos "
                   "(patricio.cubillos@oeaw.ac.at) \n"
                   "#\n# DISCLAIMER: If you make use of this code for your "
                   "research,\n"
                   "#          please consider citing "
                   "Espinoza & Jordan (2015).\n"
                   "#\n# MODIFIED BY Adrien Deline "
                   "(adrien.deline@unige.ch):\n"
                   "#\t1) replaced 'wget' URL donwloader by 'curl'\n"
                   "#\t2) included power-2 limb-darkening law"
                   " (Morello et al. 2017)\n"
                   "#\t3) corrected bug in the computation of closest vturb"
                   " for ATLAS models\n"
                   "#\t4) setting vturb to 2 km/s for [M/H]!=0 in ATLAS"
                   " models\n"
                   "#\t5) added missing -0.5 possible metallicity for PHOENIX"
                   " models\n"
                   "#\t6) changed the LD law fit to return coefficient"
                   " uncertainties\n"
                   "#\t7) added bounds to the possible LD coefficient values"
                   " to avoid negative stellar intensity or non monotically"
                   " increasing intensity toward center\n"
                   "\n"
                   .format(VERSION))

    # Read input parameters from file:
    if ifile is not None:
        input_set = []
        f = open(ifile, 'r')
        while True:
            line = f.readline()
            if line == '':
                break
            elif line[0] != '#':
                splitted = line.strip().split()
                name = fix_spaces(splitted[0])
                Teff = np.double(splitted[1])
                grav = np.double(splitted[2])
                metal = np.double(splitted[3])
                vturb = np.double(splitted[4])
                RF = fix_spaces(splitted[5])
                FT = fix_spaces(splitted[6])
                min_w = np.double(splitted[7])
                max_w = np.double(splitted[8])
                if(min_w == -1 or max_w == -1):
                    min_w = None
                    max_w = None
                input_set.append([name, RF, FT, metal, grav, Teff, vturb,
                                  min_w, max_w])
    # Else, take input parameters from the arguments:
    else:
        if (Teff is None or grav is None or metal is None
                or RF is None or FT is None):
            if verbose:
                print("Invalid input parameters.  Either define ifile, or "
                      "define Teff, grav, metal, RF, and FT.")
            return None
        input_set = [[name, RF, FT, metal, grav, Teff, vturb, min_w, max_w]]

    # Compute LDCs for each input set:
    LDC = []
    for i in np.arange(len(input_set)):
        iset = input_set[i] + [atlas_correction, photon_correction,
                               interpolation_order, fout]
        models = iset[2].split(',')
        for model in models:
            iset[2] = model
            LDC.append(calc_lds(*iset, verbose=verbose))

    if ofile is not None:
        fout.close()
        if verbose:
            print("\t > Program finished without problems.\n"
                  "\t   The results were saved in:\n"
                  "\t     '{:s}.\n".format(ofile))
    return LDC


if __name__ == "__main__":
    ifile, ofile = parse()
    lds(ifile=ifile, ofile=ofile)
