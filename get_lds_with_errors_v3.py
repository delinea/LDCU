import os
import sys
import glob
import pickle
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from uncertainties import ufloat, core as ucore
from scipy.special import erf
from scipy.optimize import root_scalar
from scipy.interpolate import UnivariateSpline, LinearNDInterpolator
import urllib.request as urlrequest
import warnings

import get_lds as lds


VERSION = 'v.1.1.deline'

ROOTDIR = os.path.dirname(os.path.realpath(__file__))

ATLAS_DIR = os.path.join(ROOTDIR, "atlas_models")
ATLAS_WEBSITE = "http://kurucz.harvard.edu/grids/"
ATLAS_Z = [-0.1, -0.2, -0.3, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0,
           -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
PHOENIX_DIR = os.path.join(ROOTDIR, "phoenix_models")
PHOENIX_WEBSITE = ("http://phoenix.astro.physik.uni-goettingen.de/data/v1.0/"
                   "SpecIntFITS/PHOENIX-ACES-AGSS-COND-SPECINT-2011/")
# PHOENIX_DIR = os.path.join(ROOTDIR, "phoenix_v3_models")
# PHOENIX_WEBSITE = ("http://phoenix.astro.physik.uni-goettingen.de/data/v3.0/"
#                    "SpecIntFITS/")


def wget_downloader(url, filename=None, verbose=False):
    if filename is None:
        cmd = "wget '{}'".format(url)
        filename = os.path.basename(url)
    else:
        dirname = os.path.dirname(filename)
        if len(dirname) > 0:
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
        cmd = "wget '{}' -O '{}'".format(url, filename)
    if not verbose:
        cmd += " -q"
    try:
        os.system(cmd)
    except BaseException as e:
        if os.path.isfile(filename):
            os.remove(filename)
        raise e
    if not os.path.isfile(filename):
        raise ConnectionError("could not download file from "
                              "url '{}'".format(url))


def curl_downloader(url, filename=None, verbose=False):
    if filename is None:
        cmd = "curl '{}' -O".format(url)
        filename = os.path.basename(url)
    else:
        dirname = os.path.dirname(filename)
        if len(dirname) > 0:
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
        cmd = "curl '{}' -o '{}'".format(url, filename)
    if not verbose:
        cmd += " -s"
    try:
        os.system(cmd)
    except BaseException as e:
        if os.path.isfile(filename):
            os.remove(filename)
        raise e
    if not os.path.isfile(filename):
        raise ConnectionError("could not download file from "
                              "url '{}'".format(url))


if lds.cmd_exists("wget"):
    downloader = wget_downloader
elif lds.cmd_exists("curl"):
    downloader = curl_downloader
else:
    def downloader(*args, **kwargs):
        raise RuntimeError("either 'Wget' or 'cURL' must be installed to "
                           "download file from a given URL")


def update_atlas_grid(force_download=False, remove_downloaded=False):
    website = urlrequest.urlopen(ATLAS_WEBSITE)
    assert(isinstance(website, urlrequest.http.client.HTTPResponse))
    website_list = [str(line, "utf-8") for line in website.readlines()]
    z_list = []
    for line in website_list:
        if ("gridm" not in line) and ("gridp" not in line):
            continue
        idx = line.index('grid')
        if line[idx+7] != "/":
            continue
        z_list.append(line[idx: idx+8])
    atlas_pck = {}
    for z in tqdm.tqdm(z_list, desc="Querying ATLAS website",
                       dynamic_ncols=True):
        website_z = urlrequest.urlopen(ATLAS_WEBSITE+z)
        assert(isinstance(website_z, urlrequest.http.client.HTTPResponse))
        website_z_list = [str(line, "utf-8") for line in website_z.readlines()]
        for line in website_z_list:
            fid = "i{}k".format(z[4:-1])
            if (".pck" not in line) or (">"+fid not in line):
                continue
            idx1 = line.index(">"+fid)+1
            idx2 = line[idx1:].index("<")+idx1
            filename = line[idx1:idx2]
            filesize = line.split()[-1]
            if filesize.endswith("K"):
                filesize = np.uint32(filesize[:-1])*2**10
            elif filesize.endswith("M"):
                filesize = np.uint32(filesize[:-1])*2**20
            elif filesize.endswith("G"):
                filesize = np.uint32(filesize[:-1])*2**30
            else:
                filesize = np.uint32(filesize)
            url = ATLAS_WEBSITE+z+filename
            atlas_pck[url] = filesize
    # Only keeps the most recent one (in order: *new.pck, *.pck19, *.pck)
    urls = [url for url in atlas_pck.keys() if url.endswith("new.pck")]
    for url in urls:
        for url_ in [url[:-7]+".pck19", url[:-7]+".pck"]:
            if url_ in atlas_pck.keys():
                atlas_pck.pop(url_)
    urls = [url for url in atlas_pck.keys() if url.endswith(".pck19")]
    for url in urls:
        if url[:-2] in atlas_pck.keys():
            atlas_pck.pop(url_)

    def filename_from_url(url):
        return(os.path.join(ATLAS_DIR, "raw_models", os.path.basename(url)))

    pck_list = atlas_pck.copy()
    if force_download:
        input_text = "All ATLAS PCK files"
    else:
        for url in atlas_pck:
            fn = filename_from_url(url)
            if os.path.isfile(fn) or os.path.isfile(fn+".tar.gz"):
                pck_list.pop(url)
        n_pck = len(pck_list)
        if n_pck > 0:
            input_text = "{} ATLAS PCK files".format(n_pck)
    if len(pck_list) > 0:
        while True:
            total_size = np.sum(list(pck_list.values()))
            unit = np.log2(total_size)//10
            total_size = total_size/2**(unit*10)
            unit = ["", "K", "M", "G"][int(unit)]
            input_text = (input_text+" will be downloaded"
                          + " ({:.1f} {}B).".format(total_size, unit)
                          + " Proceed? ([y]/n) : ")
            inp = input(input_text)
            if inp == "n":
                sys.exit()
            elif inp == "" or inp == "y":
                if not os.path.isdir(os.path.dirname(filename_from_url(""))):
                    os.makedirs(os.path.dirname(filename_from_url("")))
                for url in tqdm.tqdm(atlas_pck,
                                     desc="Downloading ATLAS PCK files",
                                     dynamic_ncols=True):
                    downloader(url, filename_from_url(url))
                break

    atlas_grid = {}
    for url in tqdm.tqdm(atlas_pck, desc="Building ATLAS grid",
                         dynamic_ncols=(True)):
        fn = filename_from_url(url)
        if not os.path.isfile(fn):
            if os.path.isfile(fn+".tar.gz"):
                with tarfile.open(fn+".tar.gz") as tar:
                    tar.extractall(os.path.dirname(fn))
            else:
                raise FileNotFoundError("could not find '{}'".format(fn))
        with open(fn, "r") as f:
            lines = f.readlines()
        if remove_downloaded and url in pck_list.keys():
            pass
        elif not os.path.isfile(fn+".tar.gz"):
            with tarfile.open(fn+".tar.gz", "w:gz") as tar:
                tar.add(fn, os.path.basename(fn))
        os.remove(fn)
        zv = os.path.basename(url)[1:6]
        if zv[0] == "p":
            z_ref = np.float32(0.1*int(zv[1:3]))
        elif zv[0] == "m":
            z_ref = np.float32(-0.1*int(zv[1:3]))
        vturb_ref = np.float32(zv[-1])
        teff = np.array([], dtype=np.uint16)
        logg = np.array([], dtype=np.float32)
        z = np.array([], dtype=np.float32)
        vturb = np.array([], dtype=np.float32)
        filesize = atlas_pck[url]
        for line in lines:
            if not line.startswith("EFF"):
                continue
            if ("[" not in line) or ("]" not in line):
                continue
            line_split = line.split()
            teff_ = np.uint16(float(line_split[line_split.index("EFF")+1]))
            logg_ = np.float32(line_split[line_split.index("GRAVITY")+1])
            z_ = np.float32(line.split("[")[1].split("]")[0])
            if z_ != z_ref:
                print("Discarded erroneous Z value in '{}'.".format(fn))
                continue
            vturb_ = np.float32(line_split[line_split.index("VTURB")+1])
            if vturb_ != vturb_ref:
                print("Discarded erroneous Vturb value in '{}'.".format(fn))
                continue
            assert np.all(vturb-vturb_ == 0)
            teff = np.append(teff, teff_)
            logg = np.append(logg, logg_)
            z = np.append(z, z_)
            vturb = np.append(vturb, vturb_)
        atlas_grid[url] = (teff, logg, z, vturb, filesize)

    atlas_teff = np.concatenate([atlas_grid[url][0] for url in atlas_grid])
    atlas_teff = np.sort(list(set(atlas_teff)))
    atlas_logg = np.concatenate([atlas_grid[url][1] for url in atlas_grid])
    atlas_logg = np.sort(list(set(atlas_logg)))
    atlas_z = np.concatenate([atlas_grid[url][2] for url in atlas_grid])
    atlas_z = np.sort(list(set(atlas_z)))
    atlas_vturb = np.concatenate([atlas_grid[url][3] for url in atlas_grid])
    atlas_vturb = np.sort(list(set(atlas_vturb)))
    url_length = np.max([len(url) for url in atlas_grid])
    name_length = np.max([len(filename_from_url(url)) for url in atlas_grid])
    atlas_grid_shape = (len(atlas_teff), len(atlas_logg), len(atlas_z),
                        len(atlas_vturb))
    atlas_urls = np.empty(atlas_grid_shape, dtype="|U{}".format(url_length))
    atlas_sizes = np.zeros(atlas_grid_shape, dtype=np.uint32)
    atlas_names = np.empty(atlas_grid_shape, dtype="|U{}".format(name_length))
    for url in atlas_grid:
        teff, logg, z, vturb, size = atlas_grid[url]
        for (teff_, logg_, z_, vturb_) in zip(teff, logg, z, vturb):
            i_teff = np.where(atlas_teff == teff_)[0][0]
            i_logg = np.where(atlas_logg == logg_)[0][0]
            i_z = np.where(atlas_z == z_)[0][0]
            i_vturb = np.where(atlas_vturb == vturb_)[0][0]
            atlas_urls[i_teff, i_logg, i_z, i_vturb] = url
            atlas_sizes[i_teff, i_logg, i_z, i_vturb] = size
            fn = filename_from_url(url)
            atlas_names[i_teff, i_logg, i_z, i_vturb] = fn
    atlas_grid = {"Teff": atlas_teff, "logg": atlas_logg,
                  "M/H": atlas_z, "vturb": atlas_vturb,
                  "url_grid": atlas_urls, "size_grid": atlas_sizes,
                  "name_grid": atlas_names}

    if not os.path.isdir(ATLAS_DIR):
        os.mkdir(ATLAS_DIR)
    pkl_fn = os.path.join(ATLAS_DIR, "atlas_grid.pkl")
    with open(pkl_fn, "wb") as f:
        pickle.dump(atlas_grid, f)
    with tarfile.open(pkl_fn+".tar.gz", "w:gz") as tar:
        tar.add(pkl_fn, os.path.basename(pkl_fn))
    os.remove(pkl_fn)


def update_phoenix_grid():
    website = urlrequest.urlopen(PHOENIX_WEBSITE)
    website_list = [str(line, "utf-8") for line in website.readlines()]
    z_list = []
    for line in website_list:
        if ">Z" not in line:
            continue
        idx = line.index(">Z") + 1
        z_list.append(line[idx: idx+5].strip())
    assert(len(set(z_list)) == len(z_list))
    phoenix_grid = {}
    for z in tqdm.tqdm(z_list, desc="Querying PHOENIX website",
                       dynamic_ncols=True):
        website_z = urlrequest.urlopen(PHOENIX_WEBSITE+z)
        website_z_list = [str(line, "utf-8") for line in website_z.readlines()]
        for i_line, line in enumerate(website_z_list):
            if (">lte" not in line) or (".fits" not in line):
                continue
            line_split = line.split()
            assert len(line_split) == 10
            filesize = np.uint32(line_split[4])
            filename = line_split[9]
            filename = filename[filename.index(">lte") + 1:]
            filename = filename[:filename.index(".fits") + 5]
            teff = np.uint16(filename[3:8])
            logg = np.float32(filename[9:13])
            z_ = np.float32(filename[13:17])
            assert z_ == np.float32(z[1:])
            url = PHOENIX_WEBSITE+z+"/"+filename
            assert(url not in phoenix_grid)
            phoenix_grid[url] = (teff, logg, z_, np.float32(2), filesize)

    def filename_from_url(url, sub_dir):
        return(os.path.join(PHOENIX_DIR, "raw_models", sub_dir,
                            os.path.basename(url)))

    phoenix_teff = np.sort(list(set(phoenix_grid[fn][0]
                                    for fn in phoenix_grid)))
    phoenix_logg = np.sort(list(set(phoenix_grid[fn][1]
                                    for fn in phoenix_grid)))
    phoenix_z = np.sort(list(set(phoenix_grid[fn][2]
                                 for fn in phoenix_grid)))
    phoenix_vturb = np.sort(list(set(phoenix_grid[fn][3]
                                     for fn in phoenix_grid)))
    url_length = np.max([len(url) for url in phoenix_grid])
    name_length = np.max([len(filename_from_url(url, "m00"))
                          for url in phoenix_grid])
    phoenix_grid_shape = (len(phoenix_teff), len(phoenix_logg), len(phoenix_z),
                          len(phoenix_vturb))
    phoenix_urls = np.empty(phoenix_grid_shape,
                            dtype="|U{}".format(url_length))
    phoenix_sizes = np.zeros(phoenix_grid_shape, dtype=np.uint32)
    phoenix_names = np.empty(phoenix_grid_shape,
                             dtype="|U{}".format(name_length))
    for url in phoenix_grid:
        teff, logg, z, vturb, size = phoenix_grid[url]
        i_teff = np.where(phoenix_teff == teff)[0][0]
        i_logg = np.where(phoenix_logg == logg)[0][0]
        i_z = np.where(phoenix_z == z)[0][0]
        i_vturb = np.where(phoenix_vturb == vturb)[0][0]
        phoenix_urls[i_teff, i_logg, i_z, i_vturb] = url
        phoenix_sizes[i_teff, i_logg, i_z, i_vturb] = size
        sub_dir = "{:02d}".format(int(abs(z*10)))
        sub_dir = "m"+sub_dir if z <= 0 else "p"+sub_dir
        fn = filename_from_url(url, sub_dir)
        phoenix_names[i_teff, i_logg, i_z, i_vturb] = fn
    phoenix_grid = {"Teff": phoenix_teff, "logg": phoenix_logg,
                    "M/H": phoenix_z, "vturb": phoenix_vturb,
                    "url_grid": phoenix_urls, "size_grid": phoenix_sizes,
                    "name_grid": phoenix_names}

    if not os.path.isdir(PHOENIX_DIR):
        os.mkdir(PHOENIX_DIR)
    pkl_fn = os.path.join(PHOENIX_DIR, "phoenix_grid.pkl")
    with open(pkl_fn, "wb") as f:
        pickle.dump(phoenix_grid, f)
    with tarfile.open(pkl_fn+".tar.gz", "w:gz") as tar:
        tar.add(pkl_fn, os.path.basename(pkl_fn))
    os.remove(pkl_fn)


def get_subgrids(Teff=(-np.inf, np.inf), logg=(-np.inf, np.inf),
                 M_H=(-np.inf, np.inf), vturb=(-np.inf, np.inf)):
    bounds = locals()
    pkl_fns = {"ATLAS": os.path.join(ATLAS_DIR, "atlas_grid.pkl"),
               "PHOENIX": os.path.join(PHOENIX_DIR, "phoenix_grid.pkl")}
    for name, pkl_fn in pkl_fns.items():
        if not os.path.isfile(pkl_fn) and not os.path.isfile(pkl_fn+".tar.gz"):
            raise FileNotFoundError("{} grid file could not be found: "
                                    "'update_{}_grid()' must be called first"
                                    .format(name, name.lower()))
    subgrids = {}
    for name, pkl_fn in pkl_fns.items():
        with tarfile.open(pkl_fn+".tar.gz") as tar:
            tar.extractall(os.path.dirname(pkl_fn))
        with open(pkl_fn, "rb") as f:
            grid = pickle.load(f)
        os.remove(pkl_fn)
        sg = {}
        idx = []
        for pn in bounds:
            pmin, pmax = bounds[pn]
            if pn == "M_H":
                pn = "M/H"
            g = grid[pn]
            pmin = pmin if np.all(g >= pmin) else np.max(g[g < pmin])
            pmax = pmax if np.all(g <= pmax) else np.min(g[g > pmax])
            idx_pn = np.where((g >= pmin) & (g <= pmax))[0]
            sg[pn] = g[idx_pn]
            idx.append(idx_pn)
        idx = tuple(np.meshgrid(*idx, indexing="ij"))
        for kw in grid:
            if kw not in sg:
                sg[kw] = grid[kw][idx]
        subgrids[name] = sg

    return(subgrids)


def download_files(Teff=(-np.inf, np.inf), logg=(-np.inf, np.inf),
                   M_H=(-np.inf, np.inf), vturb=(-np.inf, np.inf)):
    bounds = locals()
    subgrids = get_subgrids(**bounds)
    urls = []
    sizes = []
    ofns = []
    for model, subgrid in subgrids.items():
        names_ = np.ravel(subgrid["name_grid"])
        urls_ = np.ravel(subgrid["url_grid"])
        sizes_ = np.ravel(subgrid["size_grid"])
        for name, url, size in zip(names_, urls_, sizes_):
            if url == "" or url in urls:
                continue
            if os.path.isfile(name) or os.path.isfile(name+".tar.gz"):
                continue
            urls.append(url)
            sizes.append(size)
            ofns.append(name)
    n_urls = len(urls)
    if n_urls > 0:
        while True:
            total_size = np.sum(sizes)
            unit = np.log2(total_size)//10
            total_size = total_size/2**(unit*10)
            unit = ["", "K", "M", "G"][int(unit)]
            input_text = ("{} files will be downloaded".format(n_urls)
                          + " ({:.1f} {}B).".format(total_size, unit)
                          + " Proceed? ([y]/n) : ")
            inp = input(input_text)
            if inp == "n":
                sys.exit()
            elif inp == "" or inp == "y":
                for url, ofn in zip(tqdm.tqdm(urls,
                                              desc="Downloading files",
                                              dynamic_ncols=True), ofns):
                    downloader(url, ofn)
                    with tarfile.open(ofn+".tar.gz", "w:gz") as tar:
                        tar.add(ofn, arcname=os.path.basename(ofn))
                    os.remove(ofn)
                break


def extract_atlas_pck(pck_file, overwrite=True):
    pck_tarfile = pck_file+".tar.gz"
    if not os.path.isfile(pck_file):
        if os.path.isfile(pck_tarfile):
            with tarfile.open(pck_tarfile) as tar:
                tar.extractall(os.path.dirname(pck_tarfile))
        else:
            fn = os.path.join(ATLAS_DIR, "raw_models", pck_file)
            if os.path.isfile(fn):
                pck_file = fn
            elif os.path.isfile(fn+".tar.gz"):
                with tarfile.open(fn+".tar.gz") as tar:
                    tar.extractall(os.path.dirname(fn))
                pck_file = fn
            else:
                raise FileNotFoundError("could not find '{}': this PCK file "
                                        "must first be downloaded by calling "
                                        "'update_atlas_grid()'"
                                        .format(pck_file))

    metal = os.path.basename(pck_file)[1:4]
    vturb = os.path.basename(pck_file)[5:6]
    pck_dir = os.path.join(ATLAS_DIR, os.path.basename(pck_file)[1:6])
    if not os.path.isdir(pck_dir):
        os.mkdir(pck_dir)
    filenames = []
    lines = lds.getFileLines(pck_file)
    bar_format = "Extracting PCK file: {n:3} readable files processed"
    pbar = tqdm.tqdm(bar_format=bar_format, postfix=[dict(rate=0), ],
                     leave=False)
    pbar.bar_format += " ({postfix[0][rate]:.2f} files/second)"
    while True:
        TEFF, GRAVITY, LH = lds.getATLASStellarParams(lines)
        pck_subdir = os.path.join(pck_dir, TEFF)
        if not os.path.isdir(pck_subdir):
            os.mkdir(pck_subdir)
        idx, mus = lds.getIntensitySteps(lines)
        filename = os.path.join(pck_subdir, "grav_"+GRAVITY+"_lh_"+LH+".dat")
        tarname = filename+".tar.gz"
        filenames.append(filename)
        already_exists = os.path.isfile(tarname) or os.path.isfile(filename)
        to_be_saved = overwrite or not already_exists
        if to_be_saved:
            f = open(filename, "w")
            f.write("#TEFF:{} METALLICITY:{} GRAVITY:{} VTURB:{} L/H: {}\n"
                    .format(TEFF, metal, GRAVITY, vturb, LH))
            f.write("#wav (nm) \t cos(theta):" + mus)
        for i in range(idx, len(lines)):
            line = lines[i]
            j = line.find("EFF")
            k = line.find("\x0c")
            if(k != -1 or line == ""):
                pass
            elif(j != -1):
                lines = lines[i:]
                break
            else:
                wav_p_intensities = line.split(" ")
                s = lds.FixSpaces(wav_p_intensities)
                if to_be_saved:
                    f.write(s+"\n")
        if to_be_saved:
            f.close()
        if overwrite or not os.path.isfile(tarname):
            with tarfile.open(tarname, "w:gz") as tar:
                tar.add(filename, os.path.basename(filename))
        if os.path.isfile(filename):
            os.remove(filename)
        pbar.postfix[0]["rate"] = pbar.n/pbar.format_dict["elapsed"]
        pbar.update()
        if(i == len(lines)-1):
            break

    pbar.close()

    if not os.path.isfile(pck_tarfile):
        with tarfile.open(pck_tarfile, "w:gz") as tar:
            tar.add(pck_file, os.path.basename(pck_file))
    os.remove(pck_file)

    return filenames


def get_profile_interpolators(subgrids, RF, interpolation_order=1,
                              atlas_correction=True, photon_correction=True,
                              overwrite_pck=False):
    if type(RF) is str:
        RF_list = [RF, ]
        single_RF = True
    else:
        RF_list = list(RF)
        single_RF = False

    points = {}
    for kw, subgrid in subgrids.items():
        idx = dict(zip(["Teff", "logg", "M/H", "vturb"],
                       np.where(subgrid["size_grid"] > 0)))
        points[kw] = np.vstack([subgrid[kw][idx[kw]] for kw in idx]).T
        if kw == "ATLAS":
            atlas_pck_files = subgrid["name_grid"][subgrid["size_grid"] > 0]
        if kw == "PHOENIX":
            phoenix_filenames = subgrid["name_grid"][subgrid["size_grid"] > 0]

    # Response functions
    wl, S = {}, {}
    for RF in RF_list:
        try:
            wl[RF], S[RF] = lds.get_response(None, None, RF, verbose=False)[2:]
        except BaseException:
            raise FileNotFoundError("could not find {}".format(RF)) from None

    # ATLAS model
    for i, point in enumerate(tqdm.tqdm(points["ATLAS"],
                                        desc="Computing ATLAS LD curves",
                                        dynamic_ncols=True)):
        Teff, grav, metal, vturb = point
        # check if readable file available ("*.dat")
        pck_file = atlas_pck_files[i]
        filename = os.path.join(ATLAS_DIR, os.path.basename(pck_file)[1:6],
                                "{:.0f}".format(Teff),
                                "grav_{}_lh_*.dat".format(grav))
        filenames = glob.glob(filename)
        if len(filenames) == 0:
            tarnames = glob.glob(filename+".tar.gz")
            if len(tarnames) == 1:
                tarname = tarnames[0]
                with tarfile.open(tarname) as tar:
                    tar.extractall(os.path.dirname(tarname))
                filename = tarname[:-7]
            elif len(tarnames) == 0:
                extract_atlas_pck(pck_file, overwrite=overwrite_pck)
                tarnames = glob.glob(filename+".tar.gz")
                if len(tarnames) == 0:
                    raise FileNotFoundError("could not find '{}'"
                                            " after PCK extraction"
                                            .format(filename+".tar.gz"))
                elif len(tarnames) == 1:
                    tarname = tarnames[0]
                    with tarfile.open(tarname) as tar:
                        tar.extractall(os.path.dirname(tarname))
                    filename = tarname[:-7]
                else:
                    raise RuntimeError("found multiple files '{}'"
                                       " after PCK extraction"
                                       .format(filename+".tar.gz"))
            else:
                raise RuntimeError("found multiple files '{}'"
                                   .format(filename+".tar.gz"))
        elif len(filenames) == 1:
            filename = filenames[0]
            tarname = filename+".tar.gz"
        else:
            raise RuntimeError("found multiple files '{}'".format(filename))
        wavelengths, I, mu = lds.read_ATLAS(filename, None)
        if not os.path.isfile(tarname):
            with tarfile.open(tarname, "w:gz") as tar:
                tar.add(filename, os.path.basename(filename))
        os.remove(filename)
        mu100 = np.arange(1.0, 0.0, -0.01)
        I100 = np.full((len(I), len(mu100)), np.nan)
        for j, I_j in enumerate(I):
            I100[j] = UnivariateSpline(mu[::-1], I_j[::-1], s=0, k=3)(mu100)
        idx_AS = mu >= 0.05
        if i == 0:
            ATLAS_mu = {"A17": mu, "AS": mu[idx_AS], "A100": mu100}
            I_RF = {FT: np.full((len(points["ATLAS"]), len(mu_)), np.nan)
                    for FT, mu_ in ATLAS_mu.items()}
            ATLAS_I = {RF: I_RF for RF in RF_list}
        args = (atlas_correction, photon_correction, interpolation_order)
        for RF in RF_list:
            I0 = lds.integrate_response_ATLAS(wavelengths, I, mu,
                                              S[RF], wl[RF], *args)
            I0_100 = lds.integrate_response_ATLAS(wavelengths, I100, mu100,
                                                  S[RF], wl[RF], *args)
            ATLAS_I[RF]["A17"][i] = I0
            ATLAS_I[RF]["AS"][i] = I0[idx_AS]
            ATLAS_I[RF]["A100"][i] = I0_100

    # PHOENIX model
    for i, filename in enumerate(tqdm.tqdm(phoenix_filenames,
                                           desc="Computing PHOENIX LD curves",
                                           dynamic_ncols=True)):
        tarname = filename+".tar.gz"
        if not os.path.isfile(filename):
            if os.path.isfile(tarname):
                with tarfile.open(tarname) as tar:
                    tar.extractall(os.path.dirname(tarname))
            else:
                filenames = glob.glob(os.path.join(PHOENIX_DIR, "raw_models",
                                                   "*", filename))
                if len(filenames) == 1:
                    filename = filenames[0]
                    tarname = filename+".tar.gz"
                elif len(filenames) == 0:
                    raise FileNotFoundError("no such file '{}'"
                                            .format(filename))
                else:
                    raise RuntimeError("found multiple files '{}'"
                                       .format(filename))
        wavelengths, I, mu = lds.read_PHOENIX(filename)
        if not os.path.isfile(tarname):
            with tarfile.open(tarname, "w:gz") as tar:
                tar.add(filename, os.path.basename(filename))
        os.remove(filename)
        if i == 0:
            PHOENIX_sizes = {"P": len(mu), "PS": len(mu), "P100": 100}
            PHOENIX_mu = {RF: {FT: np.full((len(phoenix_filenames), n), np.nan)
                               for FT, n in PHOENIX_sizes.items()}
                          for RF in RF_list}
            PHOENIX_I = {RF: {FT: np.full((len(phoenix_filenames), n), np.nan)
                              for FT, n in PHOENIX_sizes.items()}
                         for RF in RF_list}
        for RF in RF_list:
            args = (S[RF], wl[RF], photon_correction, interpolation_order)
            I0 = lds.integrate_response_PHOENIX(wavelengths, I, mu, *args)
            r, fine_r_max = lds.get_rmax(mu, I0)
            new_r = r/fine_r_max
            idx_new = new_r <= 1.0
            new_r = new_r[idx_new]
            new_mu = np.sqrt(1.0-(new_r**2))
            I0 = I0[idx_new]
            mu100 = np.arange(0.01, 1.01, 0.01)
            I100 = np.zeros((len(wavelengths), len(mu100)))
            for j, I_j in enumerate(I):
                II = UnivariateSpline(new_mu, I_j[idx_new], s=0, k=3)
                I100[j] = II(mu100)
            I0_100 = lds.integrate_response_PHOENIX(wavelengths, I100, mu100,
                                                    *args)
            idx_PS = new_mu >= 0.05
            PHOENIX_mu[RF]["P"][i, -len(new_mu):] = new_mu
            PHOENIX_mu[RF]["PS"][i, -sum(idx_PS):] = new_mu[idx_PS]
            PHOENIX_mu[RF]["P100"][i] = mu100
            PHOENIX_I[RF]["P"][i, -len(new_mu):] = I0
            PHOENIX_I[RF]["PS"][i, -sum(idx_PS):] = I0[idx_PS]
            PHOENIX_I[RF]["P100"][i] = I0_100
    # removing/correcting NaN values
    for RF in RF_list:
        for FT in PHOENIX_mu[RF]:
            assert np.all(np.isfinite(PHOENIX_mu[RF][FT])
                          == np.isfinite(PHOENIX_I[RF][FT]))
            idx_ok = np.any(np.isfinite(PHOENIX_I[RF][FT]), axis=0)
            PHOENIX_mu[RF][FT] = PHOENIX_mu[RF][FT][:, idx_ok]
            PHOENIX_I[RF][FT] = PHOENIX_I[RF][FT][:, idx_ok]
            idx_nok = ~np.isfinite(PHOENIX_I[RF][FT])
            if np.any(idx_nok):
                idx_ok = np.argmin(idx_nok, axis=1)
                for i, idx in enumerate(idx_ok):
                    if idx == 0:
                        continue
                    PHOENIX_mu[RF][FT][i, :idx] = PHOENIX_mu[RF][FT][i, idx]
                    PHOENIX_I[RF][FT][i, :idx] = PHOENIX_I[RF][FT][i, idx]
            assert np.all(np.isfinite(PHOENIX_mu[RF][FT]))
            assert np.all(np.isfinite(PHOENIX_I[RF][FT]))

    # creating interpolators
    # TODO: to be implemented with scipy.interpolate.RBFInterpolator
    intensities = {}
    for RF in RF_list:
        intensities[RF] = {}
        for FT in ATLAS_mu:
            lndi_I = LinearNDInterpolator(points["ATLAS"], ATLAS_I[RF][FT])
            intensities[RF][FT] = (ATLAS_mu[FT], lndi_I)
        for FT in PHOENIX_mu[RF]:
            phoenix_points = points["PHOENIX"][:, :-1]  # removing vturb
            lndi_mu = LinearNDInterpolator(phoenix_points, PHOENIX_mu[RF][FT])
            lndi_I = LinearNDInterpolator(phoenix_points, PHOENIX_I[RF][FT])
            intensities[RF][FT] = (lndi_mu, lndi_I)

    # returning single value if input RF in not a list
    if single_RF:
        intensities = intensities[RF_list[0]]

    return intensities


def get_weights(distrib, grid, bounds=(-np.inf, np.inf)):
    if len(grid) == 1:
        return np.ones(1)
    elif len(grid) == 0:
        raise Exception
    assert np.all(np.diff(grid) > 0)
    assert np.diff(bounds) > 0
    midpoints = 0.5*(grid[1:]+grid[:-1])
    in_bounds = (midpoints >= bounds[0]) & (midpoints <= bounds[1])
    limits = np.concatenate((bounds[:1], midpoints[in_bounds], bounds[1:]))
    assert np.all(np.diff(limits) > 0)
    cdf = 0.5*(1+erf((limits-distrib.n)/distrib.s/np.sqrt(2)))
    if np.ptp(cdf) == 0:
        raise Exception("Grid is too far from distribution center!")
    weights = np.zeros_like(grid, dtype=np.float64)
    in_bounds = np.array(np.append((midpoints >= bounds[0]), 1)
                         * np.append(1, (midpoints <= bounds[1])), dtype=bool)
    weights[in_bounds] = np.diff(cdf)
    weights_norm = weights/np.ptp(cdf)
    return weights_norm


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


def get_all_ldc(mu, I, Ierr=None, mu_min=None, mcmc=False):
    params = (mu, I, Ierr)
    if mcmc:
        raise NotImplementedError
    else:
        ld_coeffs = {"linear": lds.fit_linear(*params),
                     "square-root": lds.fit_square_root(*params),
                     "quadratic": lds.fit_quadratic(*params),
                     "kipping": lds.fit_kipping2013(*params),
                     "three-parameter": lds.fit_three_parameter(*params),
                     "non-linear": lds.fit_non_linear(*params),
                     "logarithmic": lds.fit_logarithmic(*params),
                     "exponential": lds.fit_exponential(*params,
                                                        mu_min=mu_min),
                     "power-2": lds.fit_power2(*params)}
    return ld_coeffs


def log_prob_linear(ldc, mu, I, Ierr=None):
    if hasattr(ldc, "__iter__"):
        assert len(ldc) == 1
        ldc = ldc[0]
    if ldc < 0 or ldc > 1:
        return -np.inf
    model = 1-ldc*(1-mu)
    return -.5*np.sum(np.log(2*np.pi*Ierr**2)+((I-model)/Ierr)**2)


def log_prob_square_root(ldc, mu, I, Ierr=None):
    assert hasattr(ldc, "__iter__") and len(ldc) == 2
    if np.sum(ldc) > 1 or ldc[1] < 0 or ldc[0]+.5*ldc[1] < 0:
        return -np.inf
    model = 1-ldc[0]*(1-mu)-ldc[1]*(1-np.sqrt(mu))
    return -.5*np.sum(np.log(2*np.pi*Ierr**2)+((I-model)/Ierr)**2)


def log_prob_quadratic(ldc, mu, I, Ierr=None):
    assert hasattr(ldc, "__iter__") and len(ldc) == 2
    if np.sum(ldc) > 1 or ldc[0]+2*ldc[1] < 0 or ldc[0] < 0:
        return -np.inf
    model = 1-ldc[0]*(1-mu)-ldc[1]*(1-mu)**2
    return -.5*np.sum(np.log(2*np.pi*Ierr**2)+((I-model)/Ierr)**2)


def log_prob_kipping(ldc, mu, I, Ierr=None):
    assert hasattr(ldc, "__iter__") and len(ldc) == 2
    if ldc[0] < 0 or ldc[1] < 0 or ldc[0] > 1 or ldc[1] > 1:
        return -np.inf
    u1 = 2*np.sqrt(ldc[0])*ldc[1]
    u2 = np.sqrt(ldc[0])*(1-2*ldc[1])
    model = 1-u1*(1-mu)-u2*(1-mu)**2
    return -.5*np.sum(np.log(2*np.pi*Ierr**2)+((I-model)/Ierr)**2)


def log_prob_three_parameter(ldc, mu, I, Ierr=None):
    assert hasattr(ldc, "__iter__") and len(ldc) == 3
    if False:
        return -np.inf
    model = 1-ldc[0]*(1-mu)-ldc[1]*(1-np.sqrt(mu)**3)-ldc[2]*(1-mu**2)
    return -.5*np.sum(np.log(2*np.pi*Ierr**2)+((I-model)/Ierr)**2)


def log_prob_non_linear(ldc, mu, I, Ierr=None):
    assert hasattr(ldc, "__iter__") and len(ldc) == 4
    if False:
        return -np.inf
    model = (1-ldc[0]*(1-np.sqrt(mu))-ldc[1]*(1-mu)-ldc[2]*(1-np.sqrt(mu)**3)
             - ldc[3]*(1-mu**2))
    return -.5*np.sum(np.log(2*np.pi*Ierr**2)+((I-model)/Ierr)**2)


def log_prob_logarithmic(ldc, mu, I, Ierr=None):
    assert hasattr(ldc, "__iter__") and len(ldc) == 2
    if ldc[0] > 1 or ldc[1] < 0 or ldc[0]-ldc[1] < 0:
        return -np.inf
    model = 1-ldc[0]*(1-mu)-ldc[1]*mu*np.log(mu)
    return -.5*np.sum(np.log(2*np.pi*Ierr**2)+((I-model)/Ierr)**2)


def log_prob_exponential(ldc, mu, I, Ierr=None, mu_min=0.05):
    assert hasattr(ldc, "__iter__") and len(ldc) == 2
    if mu_min <= 0 or mu_min > 1:
        raise ValueError("'mu_min' must be >0 and <=1")
    if (ldc[1] < 0 or ldc[1]/(1-np.e) > 1
            or ldc[0]-ldc[1]*np.exp(mu_min)/(1-np.exp(mu_min))**2
            or ldc[0]-ldc[1]*np.e/(1-np.e)**2 < 0):
        return -np.inf
    model = 1-ldc[0]*(1-mu)-ldc[1]/(1-np.exp(mu))
    return -.5*np.sum(np.log(2*np.pi*Ierr**2)+((I-model)/Ierr)**2)


def log_prob_power2(ldc, mu, I, Ierr=None):
    assert hasattr(ldc, "__iter__") and len(ldc) == 2
    if ldc[0] > 1 or ldc[1] < 0 or ldc[0] < 0:
        return -np.inf
    model = 1-ldc[0]*(1-mu**ldc[1])
    return -.5*np.sum(np.log(2*np.pi*Ierr**2)+((I-model)/Ierr)**2)


# TODO:
#   include MCMC with log_prob_law
#   implement fit_noise (noise = s*(1-mu) or noise = s*(1-mu)+ss*(1-mu)^2)
#   implement as class
#   store differnet byproducts after init (e.g. interpolators)


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
            "#   1) added option for URL donwloader 'cURL'\n"
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
            "ensure stellar\n"
            "#          intensity is positive and monotonically decreasing "
            "toward the limb\n"
            "#   8) discard some erroneous metallicity values in the ATLAS PCK"
            " files\n"
            "#\n" + 79*"#" + "\n")
    text = text.format(VERSION)

    text += ("#\n# Limb-darkening laws\n"
             "#\n# Linear law:\n"
             "#   I(mu)/I(1) = 1 - a * (1 - mu)\n"
             "#\n# Square-root law:\n"
             "#   I(mu)/I(1) = 1 - s1 * (1 - mu) - s2 * (1 - mu^(1/2))\n"
             "#\n# Quadratic law:\n"
             "#   I(mu)/I(1) = 1 - u1 * (1 - mu) - u2 * (1 - mu)^2\n"
             "#\n# Quadratic law (Kipping 2013):\n"
             "#   I(mu)/I(1) = 1 - 2*sqrt(q1)*q2 * (1 - mu) "
             "- sqrt(q1)*(1-2*q2) * (1 - mu)^2\n"
             "#\n# Three-parameter law:\n"
             "#   I(mu)/I(1) = 1 - b1 * (1 - mu) - b2 * (1 - mu^(3/2)) "
             "- b3 * (1 - mu^2)\n"
             "#\n# Non-linear law:\n"
             # "#   I(mu)/I(1) = 1 - c1 * (1 - mu^(1/2)) - c2 * (1 - mu) "
             # "- c3 * (1 - mu^(3/2) - c4 * (1 - mu^2)\n"
             "#   I(mu)/I(1) = 1 - SUM[i=1..4]{c_i * (1 - mu^(i/2)}\n"
             "#\n# Logarithmic law:\n"
             "#   I(mu)/I(1) = 1 - l1 * (1 - mu) - l2 * mu * log(mu)\n"
             "#\n# Exponential law:\n"
             "#   I(mu)/I(1) = 1 - e1 * (1 - mu) - e2 / (1 - exp(mu))\n"
             "#\n# Power-2 law:\n"
             "#   I(mu)/I(1) = 1 - p1 * (1 - mu^p2)\n"
             "#\n" + 79*"#" + "\n")

    text += ("#\n# Description of the models:\n"
             "#   A17:     ATLAS intensity profiles with all 17 mu-values\n"
             "#   AS:      ATLAS intensity profiles for mu >= 0.05 "
             "(15 mu-values) (Sing 2010)\n"
             "#   A100:    ATLAS intensity profiles with 100 interpolated "
             "mu-points using a\n"
             "#              cubic spline (Claret & Bloemen 2011)\n"
             "#   P:       PHOENIX intensity profiles (Husser et al. 2013)\n"
             "#   PS:      PHOENIX intensity profiles for mu >= 0.05 "
             "(Sing 2010)\n"
             "#   P100:    PHOENIX intensity profiles with 100 interpolated "
             "mu-points using a\n"
             "#              cubic spline (Claret & Bloemen 2011)\n"
             "#\n# Description of the merged models:\n"
             "#   AP:      A17 + P\n"
             "#   APS:     AS + PS\n"
             "#   AP100:   A100 + P100\n"
             "#   ATLAS:   A17 + AS + A100\n"
             "#   PHOENIX: P + PS + P100\n"
             "#   ALL:     A17 + AS + A100 + P + PS + P100\n"
             "#\n# The merged models compute each LD coefficient "
             "(value and error) from a\n"
             "#   distribution that is a merging of normal distributions "
             "based on the\n"
             "#   LD coefficient values and errors from the different "
             "considered models.\n"
             "#\n# RECOMMENDATION: we recommend using the Merged/ALL "
             "coefficients, which have\n"
             "#   the most conservative estimated uncertainties.\n"
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
            if ldm == "kipping":
                text += "\n\nLD quadratic law (Kipping 2013):"
            else:
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
                        elif type(c) is ucore.Variable:
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


def get_lds_with_errors(Teff=None, logg=None, M_H=None, vturb=None,
                        RF="cheops_response_function.dat",
                        nsig=4, nsamples=2000, max_bad_points=0.30,
                        overwrite_pck=False):
    if vturb is None:
        vturb = ufloat(2, .5)
        warnings.warn("Microturbulent velocity 'vturb' not specified: "
                      "set to default value 2.0+/-0.5 km/s.",
                      UserWarning)
    for par in [Teff, logg, M_H, vturb]:
        if type(par) is not ucore.Variable:
            raise ValueError("input stellar parameters must all be of type "
                             "'uncertainties.ufloat'")
    if type(RF) is str:
        RF_list = [RF, ]
    else:
        RF_list = list(RF)

    bounds = {"Teff": (max(0, Teff.n-nsig*Teff.s), Teff.n+nsig*Teff.s),
              "logg": (logg.n-nsig*logg.s, logg.n+nsig*logg.s),
              "M_H": (M_H.n-nsig*M_H.s, M_H.n+nsig*M_H.s),
              "vturb": (max(0, vturb.n-nsig*vturb.s), vturb.n+nsig*vturb.s)}

    download_files(**bounds)
    subgrids = get_subgrids(**bounds)

    ip_interp = get_profile_interpolators(subgrids, RF_list,
                                          interpolation_order=1,
                                          atlas_correction=True,
                                          photon_correction=True,
                                          overwrite_pck=overwrite_pck)

    # Drawing stellar parameters from normal distributions
    vals = np.full((nsamples, 4), np.nan)
    for i, p in enumerate([Teff, logg, M_H, vturb]):
        vals[:, i] = np.random.normal(p.n, p.s, nsamples)
        if (i == 0) or (i == 3):    # removing negative values
            n_iter = 100
            for j in range(n_iter):
                if np.all(vals[:, i] >= 0):
                    break
                idx = np.where(vals[:, i] < 0, True, False)
                vals[:, i][idx] = np.random.normal(p.n, p.s, sum(idx))
                if j == n_iter-1:
                    # should never happen...
                    raise RuntimeError("failed to draw stellar parameters")

    # Interpolating ATLAS and PHOENIX LD curves
    FT_merged = {"AP": ["A17", "P"], "APS": ["AS", "PS"],
                 "AP100": ["A100", "P100"], "ATLAS": ["A17", "AS", "A100"],
                 "PHOENIX": ["P", "PS", "P100"],
                 "ALL": ["A17", "AS", "A100", "P", "PS", "P100"]}
    intensity_profiles = {}
    ldc_all = {RF: {} for RF in RF_list}
    n_ldc = len(get_all_ldc(np.linspace(.1, 1, 10), np.linspace(.1, 1, 10)))
    pbar = tqdm.tqdm(total=len(RF_list)*(len(ip_interp[RF_list[0]])+n_ldc),
                     desc="Computing LD coefficients", dynamic_ncols=True)
    for RF in RF_list:
        intensity_profiles[RF] = {}
        ldc_RF = {}
        for FT in ip_interp[RF]:
            lndi_mu, lndi_I = ip_interp[RF][FT]
            if FT.startswith("A"):
                mu = lndi_mu
                I = lndi_I(vals)
                mu = np.tile(mu, (len(I), 1))
            elif FT.startswith("P"):
                mu = lndi_mu(vals[:, :-1])
                I = lndi_I(vals[:, :-1])
            idx = np.isfinite(mu) & np.isfinite(I) & (I >= 0)
            bad_points = 1-np.sum(idx)/np.size(idx)
            if bad_points > 0:
                lib_name = "ATLAS" if FT.startswith("A") else "PHOENIX"
                txt = (" stellar parameter distribution"
                       " falls outside the interpolation range"
                       " covered by the {} library!".format(lib_name))
                txt_err = " (setting 'vturb = None' may solve this issue)"
                if bad_points > max_bad_points:
                    raise RuntimeError("{:.1f} % of the".format(bad_points*100)
                                       + txt + txt_err)
                elif bad_points > 0:
                    warnings.warn("{:.1f} % of the".format(bad_points*100)
                                  + txt)
            if bad_points == 1:
                continue
            mu = mu[idx]
            I = I[idx]
            idx = np.argsort(mu)
            mu = mu[idx]
            I = I[idx]
            intensity_profiles[RF][FT] = (mu, I)
            ldc_FT = get_all_ldc(mu, I, Ierr=None, mu_min=None)
            for ldm in ldc_FT:
                if ldm not in ldc_RF:
                    ldc_RF[ldm] = {FT: ldc_FT[ldm]}
                else:
                    ldc_RF[ldm][FT] = ldc_FT[ldm]
            pbar.update()

        ldc_merged_RF = {ldm: {} for ldm in ldc_RF}
        for ldm in ldc_RF:
            ldc_ldm = ldc_RF[ldm]
            n_ldc = [len(ldc) for ldc in ldc_ldm.values()]
            assert np.all(np.diff(n_ldc) == 0)
            n_ldc = n_ldc[0]
            for FT in FT_merged:
                ldc_FT = []
                for i in range(n_ldc):
                    distribs = [ldc_ldm[FT_][i] for FT_ in FT_merged[FT]
                                if FT_ in ldc_ldm]
                    if len(distribs) <= 1:
                        continue
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
                ldc_merged_RF[ldm][FT] = tuple(ldc_FT)
            pbar.update()

        for ldm in ldc_merged_RF:
            FTs = ldc_RF[ldm].keys()
            ldc_all[RF][ldm] = {"ATLAS": {FT: ldc_RF[ldm][FT] for FT in FTs
                                          if FT.startswith("A")},
                                "PHOENIX": {FT: ldc_RF[ldm][FT] for FT in FTs
                                            if FT.startswith("P")},
                                "Merged": ldc_merged_RF[ldm]}
    pbar.close()

    # TODO
    # check not too many NaNs in LD curves (especially due to vturb...)
    # store intensity profile interpolators
    # store sample curves
    # fit sample curves (curve_fit & MCMC) with each LD law
    # store coeff and errors

    return ldc_all
