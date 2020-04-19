#! /usr/bin/env python3

import sys
import re
from pathlib import Path
import warnings

import docopt
import humanize
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pyBigWig
import vcfpy
from tqdm import tqdm
from cycler import cycler

doc = """
Uses VCF file, bigWig, and a set of positions to plot genome tracks for each allele.

Sample names are specified by Bigwig files and must exist in VCF file.

Usage:
    plotBigWigByGenotype.py --vcf=<vcf> --pos=<pos>
        [--output=<file> --left_pad=<w> --right_pad=<w> --debug --sample_regex=<regex> --ymax=<y>] <bigwig>...

Options:
    -h, --help         Show help.
    --vcf=<vcf>        VCF file.
    --pos=<pos>        BED file with SNP positions.
    --left_pad=<w>     Left window around SNP to plot (in bases) [default: 1000]
    --right_pad=<w>    Right window around SNP to plot (in bases) [default: 1000]
    --output=<file>    Output directory where PDF files are written. [default: output.pdf]
    --bigwig           Bigwig files corresponding to each sample.
    --ymax=<y>             Maximum y-limit. [default: auto]
    --sample_regex=<regex>     Regex to extract sample names from filenames [default: .*]
    --debug            Talk more.
"""


colors = ["#BA4225", "#424C2C", "#B8A159", "#B2684D"]
mpl.rcParams["axes.prop_cycle"] = cycler(color=colors)
mpl.rcParams["savefig.pad_inches"] = 0
mpl.rc("font", size=14)
mpl.rc("axes", titlesize=14)
mpl.rc("legend", fontsize=15)
mpl.rc("image", cmap="gray")


@ticker.FuncFormatter
def x_tick_formatter(x, pos):
    return humanize.filesize.naturalsize(x, format="%.3f").replace("B", "b")


def check_args(opts, exit=True):
    """Check if all arguments are sane."""
    status = 0
    if not opts["--vcf"].exists():
        print("error: vcf file not found.", file=sys.stderr)
        status = 1
    if not opts["--pos"].exists():
        print("error: bed file not found.", file=sys.stderr)
        status = 1
    for k in opts["<bigwig>"]:
        if not k.exists():
            print(f"error: {k} file not found.", file=sys.stderr)
            status = 1
    if status and exit:
        print("exit.", file=sys.stderr)
        sys.exit(status)

    return status


class BigwigObj:
    def __init__(self, url):
        myurl = url
        self.bw = pyBigWig.open(myurl)

    def get_scores(self, pos):
        return self.bw.values(*pos)


def get_value_from_pos(bw_object, pos, left_pad=1000, right_pad=1000):
    """Fetch values from BigWig object. See class defintion above."""
    scores = None
    try:
        scores = bw_object.get_scores(
            [pos[0], pos[1] - abs(left_pad), pos[2] + abs(right_pad)]
        )
    except Exception as e:
        print("error: was trying to get scores: {0}".format(e))

    if scores is not None:
        if len(scores) != 0:
            return [pos, np.mean(scores), scores]

    return None


def bed_reader(bed_file):
    """Simple BED file reader."""
    positions = []
    with open(bed_file) as f:
        # print("info: found 3 fields per line..")
        for line in f:
            split_line = line.strip().split("\t")
            if len(split_line) == 3:
                split_line[1] = int(split_line[1])
                split_line[2] = int(split_line[2])
                positions.append(split_line)
    return positions


if __name__ == "__main__":
    opts = docopt.docopt(doc)

    # process args
    opts["--vcf"] = Path(opts["--vcf"]).absolute()
    opts["--pos"] = Path(opts["--pos"]).absolute()
    opts["--left_pad"] = int(opts["--left_pad"])
    opts["--right_pad"] = int(opts["--right_pad"])
    opts["--sample_regex"] = re.compile(opts["--sample_regex"])
    debug = opts["--debug"]
    y_max = None
    if opts["--ymax"] != "auto":
        y_max = int(opts["--ymax"])

    for i, k in enumerate(opts["<bigwig>"]):
        opts["<bigwig>"][i] = Path(k).absolute()

    if not debug:
        warnings.filterwarnings("ignore")

    check_args(opts)
    sample_ids = [re.match(opts["--sample_regex"], x.stem)[1] for x in opts["<bigwig>"]]

    try:
        vcf_file = vcfpy.Reader.from_path(opts["--vcf"], parsed_samples=sample_ids)
    except AssertionError:
        vcf_file = vcfpy.Reader.from_path(opts["--vcf"])
        samples_vcf = vcf_file.header.samples.names
        not_found = [x for x in sample_ids if x not in samples_vcf]
        print(f"error: not all samples present in the vcf file.")
        print(f"error: following not found - {not_found}")
        sys.exit(1)

    # NOTE: BED is 0-based
    positions = bed_reader(opts["--pos"])

    data = {}
    bar = tqdm(total=len(sample_ids) + len(positions))
    for _, sample in enumerate(opts["<bigwig>"]):
        sample_id = re.match(opts["--sample_regex"], sample.stem)[1]
        if debug:
            print(f"info: processing sample {sample_id}")
        bw = BigwigObj(str(sample))

        slot = {}
        for pos in positions:
            key = f"{pos[0]}:{pos[2]}"

            # get bigwig values
            bw_values = get_value_from_pos(
                bw, pos, left_pad=opts["--left_pad"], right_pad=opts["--right_pad"]
            )

            # get genotype status
            if debug:
                print(f"info: querying {pos}..")
            # NOTE: 0-based query (but VCF is 1-based)
            # TODO: Check consistency of chr/no-chr for the chromosome field
            query = vcf_file.fetch(pos[0], pos[1], pos[2])
            for record in query:
                if not record.is_snv():
                    print("info: not a SNV!")
                    continue
                info = record.INFO
                sample_call = record.call_for_sample.get(sample_id)
                if debug:
                    print(sample_call)
                    print(
                        f"info: found: {record.CHROM}:{record.POS},"
                        f" ref/alt: {record.REF}/{record.ALT[0].value},"
                        f" maf: {info.get('MAF')},"
                        f" af: {info.get('AF')}"
                    )
                slot[key] = {
                    "gt": "/".join(sample_call.gt_bases),
                    "ref": record.REF,
                    "gt_type": sample_call.gt_type,
                    "het": sample_call.is_het,
                    "pos": bw_values[0],
                    "signal_mean": bw_values[1],
                    "signal_values": bw_values[2],
                }

        data[sample_id] = slot
        bar.update(1)

    fig, axs = plt.subplots(
        nrows=len(positions), ncols=1, figsize=(10, 3 * len(positions)), squeeze=False
    )

    for i, pos in enumerate(positions):
        key = f"{pos[0]}:{pos[2]}"
        genotype_signal = {}

        # Set plot ranges
        x_range = (
            int(pos[2]) - opts["--left_pad"],
            int(pos[2]) + opts["--right_pad"] + 1,
        )

        # Collect genotype signals from all samples
        for _, v in data.items():
            if key in v:
                vals = v[key]
                if vals["gt_type"] not in genotype_signal:
                    genotype_signal[vals["gt_type"]] = []
                genotype_signal[vals["gt_type"]].append(vals["signal_values"])

        # Sort so that lowest signal track is in the front of the plot
        genotype_signal = {
            k: v
            for k, v in sorted(
                genotype_signal.items(), key=lambda item: np.sum(item[0])
            )
        }

        for k, v in genotype_signal.items():
            # NOTE: Add signal across samples for each genotype and average
            # within each genotype group.
            genotype_signal[k] = np.sum(v, axis=0) / len(v)

            # Make an area plot with genotype as legend
            axs[i, 0].fill_between(range(*x_range), genotype_signal[k], label=k)

        # Prettify plots
        axs[i, 0].set_title(key)
        axs[i, 0].legend(loc=0, frameon=False)
        axs[i, 0].spines["right"].set_visible(False)
        axs[i, 0].spines["top"].set_visible(False)
        # axs[i, 0].spines["left"].set_visible(False)
        # axs[i, 0].get_yaxis().set_visible(True)
        axs[i, 0].set_xlim(*x_range)
        axs[i, 0].set_ylim(ymin=0)
        if y_max:
            axs[i, 0].set_ylim(ymax=y_max)
        axs[i, 0].yaxis.set_major_locator(ticker.LinearLocator(2))
        yticks = axs[i, 0].get_yticks()
        axs[i, 0].set_yticks([yticks[-1]])
        axs[i, 0].xaxis.set_major_locator(ticker.LinearLocator(4))
        axs[i, 0].xaxis.set_major_formatter(x_tick_formatter)

        # Update progress
        bar.update(1)

    fig.tight_layout()
    fig.savefig(opts["--output"], dpi=300)
    bar.close()
