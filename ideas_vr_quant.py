import os.path as op
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import cohen_kappa_score
from general.basic.helper_funcs import *


def vr_quant_kappa(vr, cl, min_cutoff=-20, max_cutoff=100, step=0.1):
    """Calculate Cohen's kappa between visual and quant. results.

    Tests a range of CL thresholds and calculates kappa for each.

    Parameters
    ----------
    vr : array vector
        Visual read result ("pos" or "neg")
    cl : array vector
                Vector of continuous Centiloids.
    min_cutoff : int, optional
                Minimum CL threshold to test, by default -20.
    max_cutoff : int, optional
                Maximum CL threshold to test, by default 100.
    step : int, optional
                Step size for CL thresholds, by default 1.

    Returns
    -------
    kappas : pandas.Series
            Series containing the kappa for each CL threshold.
    """
    cl_cutoffs = np.arange(min_cutoff, max_cutoff + step, step)
    vr_result = vr == "pos"
    kappas = []
    for cl_cutoff in cl_cutoffs:
        cl_result = cl > cl_cutoff
        kappa = cohen_kappa_score(vr_result, cl_result)
        kappas.append(kappa)

    kappas = pd.Series(index=cl_cutoffs, data=kappas)

    return kappas


def bootstrap_vr_quant_kappa(vr, cl, n_resamples=10000, method="BCa"):
    """Calculate bootstrap CIs for the kappa between visual and quant.

    Parameters
    ----------
    vr : array vector
        Visual read result ("pos" or "neg")
    cl : array vector
        Vector of continuous Centiloids.
    n_resamples : int, optional
        Number of bootstrap samples to draw, by default 10000.
    method : str, optional
        Method to use for calculating bootstrap CIs, by default "BCa".

    Returns
    -------
    res : BootstrapResult
        An object with attributes:
        confidence_interval : ConfidenceInterval
            The bootstrap confidence interval as an instance of
            collections.namedtuple with attributes low and high.
        bootstrap_distribution : ndarray
            The bootstrap distribution, that is, the value of statistic
            for each resample. The last dimension corresponds with the
            resamples (e.g. res.bootstrap_distribution.shape[-1] ==
            n_resamples).
        standard_error : float or ndarray
            The bootstrap standard error, that is, the sample standard
            deviation of the bootstrap distribution.
    """
    assert vr.size == cl.size
    assert vr.ndim == 1 and cl.ndim == 1
    res = stats.bootstrap(
        (vr, cl),
        vr_quant_kappa,
        n_resamples=n_resamples,
        paired=True,
        method=method,
    )
    return res


def calc_max_kappa(vr_quant):
    """Calculate max kappa and CL threshold within each data group.

    Also return the difference in optimal CL threshold between scans
    that used quantification in the visual read process and those that
    didn't.

    Parameters
    ----------
    vr_quant : pandas.DataFrame
        Dataframe containing the data.

    Returns
    -------
    max_kappa : dict
        Dictionary containing the max kappa ("kappa") and associated CL
        threshold ("cl_thresh") within each data group.
    cl_thresh_diff : float
        Difference in optimal CL threshold between scans that used
        quantification in the visual read process and those that didn't.
    """
    max_kappa = {}
    for k in ["all", "no_quant", "used_quant"]:
        max_kappa[k] = {
            "cl_thresh": np.round(
                vr_quant.query("(data=='{}')".format(k))["cl_thresh"].values[
                    vr_quant.query("(data=='{}')".format(k))["kappa"].argmax()
                ],
                1,
            ),
            "kappa": vr_quant.query("(data=='{}')".format(k))["kappa"].max(),
        }

    # Calculate the difference in optimal CL threshold between scans that
    # used quantification in the visual read process and those that didn't.
    cl_thresh_diff = np.round(
        max_kappa["used_quant"]["cl_thresh"] - max_kappa["no_quant"]["cl_thresh"], 1
    )

    return max_kappa, cl_thresh_diff


if __name__ == "__main__":
    timer = Timer("  Time from program start: ")

    # Hard-coded parameters
    n_resamples = 100
    save_output = True
    overwrite = True
    verbose = True

    # Load data
    proj_dir = op.join(op.expanduser("~"), "Box/projects/ideas_visual_quant")
    ssheet_dir = op.join(proj_dir, "data", "ssheets")
    infile = op.join(ssheet_dir, "dat_2023-11-15.csv")
    dat = pd.read_csv(infile)

    # Calculate Cohen's kappa between visual and quant. results for:
    # 1. All scans
    vr_quant = []
    _dat = dat
    if verbose:
        print("Calculating Cohen's kappa between visual and quant. results for:")
        print("  1. All {:,} IDEAS amyloid PET scans".format(_dat.shape[0]))
        print(timer)
    vr_quant.append(vr_quant_kappa(_dat["vr"], _dat["cl"]).reset_index())
    vr_quant[-1].columns = ["cl_thresh", "kappa"]
    vr_quant[-1].insert(0, "data", "all")
    boot_res = bootstrap_vr_quant_kappa(_dat["vr"], _dat["cl"], n_resamples=n_resamples)
    vr_quant[-1]["lower_ci"] = boot_res.confidence_interval.low
    vr_quant[-1]["upper_ci"] = boot_res.confidence_interval.high

    # 2. Scans that didn't use quantification in the visual read process
    _dat = dat.query("(read_used_quant==False)")
    if verbose:
        print(
            "  2. {:,} scans where quantification was not used for visual read".format(
                _dat.shape[0]
            )
        )
        print(timer)
    vr_quant.append(
        vr_quant_kappa(
            _dat["vr"],
            _dat["cl"],
        ).reset_index()
    )
    vr_quant[-1].columns = ["cl_thresh", "kappa"]
    vr_quant[-1].insert(0, "data", "no_quant")
    boot_res = bootstrap_vr_quant_kappa(_dat["vr"], _dat["cl"], n_resamples=n_resamples)
    vr_quant[-1]["lower_ci"] = boot_res.confidence_interval.low
    vr_quant[-1]["upper_ci"] = boot_res.confidence_interval.high

    # 3. Scans that used quantification in the visual read process
    _dat = dat.query("(read_used_quant==True)")
    if verbose:
        print(
            "  3. {:,} scans where quantification was used for visual read".format(
                _dat.shape[0]
            )
        )
        print(timer)
    vr_quant.append(
        vr_quant_kappa(
            _dat["vr"],
            _dat["cl"],
        ).reset_index()
    )
    vr_quant[-1].columns = ["cl_thresh", "kappa"]
    vr_quant[-1].insert(0, "data", "used_quant")
    boot_res = bootstrap_vr_quant_kappa(_dat["vr"], _dat["cl"], n_resamples=n_resamples)
    vr_quant[-1]["lower_ci"] = boot_res.confidence_interval.low
    vr_quant[-1]["upper_ci"] = boot_res.confidence_interval.high

    # Combine the dataframes
    vr_quant = pd.concat(vr_quant, axis=0).reset_index(drop=True)

    # Calculate the maximum kappa and CL threshold within each data grouping.
    max_kappa, cl_thresh_diff = calc_max_kappa(vr_quant)

    if save_output:
        outfile = op.join(ssheet_dir, f"vr_quant_{n_resamples}resamples_{today()}.csv")
        if overwrite or not op.exists(outfile):
            vr_quant.to_csv(outfile, index=False)
            print(f"Saved {outfile}")

    if verbose:
        print(f"vr_quant shape: {vr_quant.shape}")
        for k in max_kappa:
            print(
                "max_kappa['{}']: cl_thresh = {}, kappa = {:.4f}".format(
                    k, max_kappa[k]["cl_thresh"], max_kappa[k]["kappa"]
                )
            )
        print(f"cl_thresh_diff: {cl_thresh_diff}")
        print("Done!")
        print(timer)
