#!/usr/bin/env python3
import os
import re
import argparse
import importlib
import sys
import textwrap
from pathlib import Path

np = None
pd = None
matplotlib = None
plt = None
MaxNLocator = None
FormatStrFormatter = None
gp = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KG = PROJECT_ROOT / "demo/BioNexKG_demo/BioNexKG_demo.csv"
DEFAULT_HEAD = "46220502"
DEFAULT_OUTDIR = "pathway_enrichment_results"
REQUIRED_KG_COLUMNS = {
    "head",
    "relation",
    "tail",
    "head_type",
    "tail_type",
}
RELATIONS = ("CGI-U", "CGI-D")
EMPTY_ENRICHMENT_COLUMNS = [
    "Gene_set",
    "Term",
    "Overlap",
    "P_value",
    "Adjusted_P_value",
    "Odds_Ratio",
    "Combined_Score",
    "Genes",
    "Gene_Count",
    "Gene_Ratio",
    "minus_log10_FDR",
    "Term_key",
]


class WorkflowError(RuntimeError):
    """Expected input, dependency, or remote-service failure."""


def load_data_dependencies():
    global np, pd

    missing = []
    try:
        np = importlib.import_module("numpy")
    except ImportError:
        missing.append("numpy")
    try:
        pd = importlib.import_module("pandas")
    except ImportError:
        missing.append("pandas")

    if missing:
        raise WorkflowError(
            "Missing Python dependencies: "
            + ", ".join(missing)
            + ". Install them before running pathway enrichment."
        )


def load_enrichment_dependency():
    global gp

    try:
        gp = importlib.import_module("gseapy")
    except ImportError as exc:
        raise WorkflowError(
            "Missing Python dependency: gseapy. Install it with "
            "`python -m pip install gseapy`."
        ) from exc


def load_plotting_dependencies():
    global matplotlib, plt, MaxNLocator, FormatStrFormatter

    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
        ticker = importlib.import_module("matplotlib.ticker")
    except ImportError as exc:
        raise WorkflowError(
            "Missing Python dependency: matplotlib. Install it before "
            "generating enrichment plots."
        ) from exc

    MaxNLocator = ticker.MaxNLocator
    FormatStrFormatter = ticker.FormatStrFormatter


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def cm_to_inch(cm):
    return cm / 2.54


def sanitize_filename(s):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(s),
    )


def set_plot_style(base_fontsize=5.5):
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
    ]

    matplotlib.rcParams["font.size"] = base_fontsize
    matplotlib.rcParams["axes.titlesize"] = base_fontsize
    matplotlib.rcParams["axes.labelsize"] = base_fontsize
    matplotlib.rcParams["xtick.labelsize"] = base_fontsize
    matplotlib.rcParams["ytick.labelsize"] = base_fontsize
    matplotlib.rcParams["legend.fontsize"] = base_fontsize

    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    matplotlib.rcParams["axes.linewidth"] = 0.5
    matplotlib.rcParams["xtick.major.width"] = 0.5
    matplotlib.rcParams["ytick.major.width"] = 0.5
    matplotlib.rcParams["xtick.major.size"] = 2.0
    matplotlib.rcParams["ytick.major.size"] = 2.0

    matplotlib.rcParams["savefig.dpi"] = 600
    matplotlib.rcParams["figure.dpi"] = 150


class CountSizeMapper:
    """
    Map enrichment gene counts to bubble areas.

    The original bubble-size range is retained:
        minimum bubble area = 12
        maximum bubble area = 85

    Each CGI-U or CGI-D figure constructs its own mapper using
    only the count values displayed in that figure.
    """

    def __init__(
        self,
        values,
        min_size=12,
        max_size=85,
    ):
        values = np.asarray(
            pd.Series(values).dropna(),
            dtype=float,
        )

        values = values[
            np.isfinite(values)
        ]

        values = values[
            values > 0
        ]

        self.min_size = min_size
        self.max_size = max_size

        self.vmin = (
            float(np.nanmin(values))
            if len(values)
            else 1.0
        )

        self.vmax = (
            float(np.nanmax(values))
            if len(values)
            else 1.0
        )

    def __call__(self, values):
        values = np.asarray(
            values,
            dtype=float,
        )

        values = np.nan_to_num(
            values,
            nan=self.vmin,
            posinf=self.vmax,
            neginf=self.vmin,
        )

        values = np.maximum(
            values,
            self.vmin,
        )

        if self.vmax == self.vmin:
            return np.full_like(
                values,
                (
                    self.min_size
                    + self.max_size
                ) / 2,
                dtype=float,
            )

        scaled = (
            np.sqrt(values)
            - np.sqrt(self.vmin)
        ) / (
            np.sqrt(self.vmax)
            - np.sqrt(self.vmin)
        )

        return (
            self.min_size
            + scaled
            * (
                self.max_size
                - self.min_size
            )
        )


def remove_accession(term):
    """
    Remove accession-like suffixes from enrichment term names.
    """
    term = str(term).strip()

    term = re.sub(
        r"\s*\((GO|KEGG|R-HSA|WP|REAC|M\d+|HALLMARK)[^)]*\)\s*$",
        "",
        term,
        flags=re.IGNORECASE,
    )

    term = re.sub(
        r"\s*\(([A-Za-z0-9:_-]{4,})\)\s*$",
        "",
        term,
    )

    return re.sub(
        r"\s+",
        " ",
        term,
    ).strip()


def make_term_key(term):
    """
    Create a normalized enrichment-term key for deduplication.
    """
    return remove_accession(
        term
    ).casefold()


def wrap_term(
    term,
    width=28,
    tolerance=2,
):
    """
    Wrap long pathway names without breaking words.
    """
    term = remove_accession(term)

    if len(term) <= width + tolerance:
        return term

    return "\n".join(
        textwrap.wrap(
            term,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def parse_overlap(overlap):
    """
    Parse an Enrichr overlap string such as '8/200'.

    Returns:
        Gene_Count = 8
        Gene_Ratio = 8 / 200
    """
    try:
        numerator, denominator = str(overlap).split("/")

        numerator = int(numerator)
        denominator = int(denominator)

        ratio = (
            numerator / denominator
            if denominator > 0
            else np.nan
        )

        return numerator, ratio

    except Exception:
        return np.nan, np.nan


def adaptive_count_breaks(
    values,
    narrow_range_threshold=30,
):
    """
    Generate natural Count legend values.

    Rules:
        1. Every legend value is a positive multiple of 5.
        2. Use 3 circles when the rounded count range is narrow.
        3. Use 4 circles when the rounded count range is wide.
        4. Legend values are evenly spaced.
        5. The legend range covers the observed count range whenever
           possible.

    If the observed count range is too narrow to provide three
    different multiples of 5, the legend upper limit is extended
    slightly. The same CountSizeMapper is then used for both the
    plotted bubbles and legend bubbles, so their sizes remain
    numerically consistent.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    values = values[
        values > 0
    ]

    if len(values) == 0:
        return []

    observed_min = float(
        np.min(values)
    )

    observed_max = float(
        np.max(values)
    )

    # Round the lower and upper bounds to multiples of 5.
    lower = int(
        np.floor(
            observed_min / 5.0
        ) * 5
    )

    upper = int(
        np.ceil(
            observed_max / 5.0
        ) * 5
    )

    # Count legends should not contain zero or negative values.
    lower = max(
        5,
        lower,
    )

    upper = max(
        lower,
        upper,
    )

    rounded_range = (
        upper
        - lower
    )

    # Use three circles for relatively narrow count ranges and
    # four circles for wider count ranges.
    number_of_breaks = (
        3
        if rounded_range
        <= narrow_range_threshold
        else 4
    )

    # At least 5 units are required between adjacent legend values.
    minimum_range = (
        number_of_breaks
        - 1
    ) * 5

    if rounded_range < minimum_range:
        upper = (
            lower
            + minimum_range
        )

    # Calculate an evenly spaced step and round it upward to the
    # nearest multiple of 5.
    raw_step = (
        upper
        - lower
    ) / (
        number_of_breaks
        - 1
    )

    step = int(
        np.ceil(
            raw_step / 5.0
        ) * 5
    )

    step = max(
        5,
        step,
    )

    count_breaks = [
        lower
        + step * i
        for i in range(
            number_of_breaks
        )
    ]

    return count_breaks


def compute_color_limits(values):
    """
    Calculate the Gene Ratio color range using only the values
    in the current CGI-U or CGI-D panel.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return 0.0, 1.0

    vmin = max(
        0.0,
        float(np.min(values)),
    )

    vmax = min(
        1.0,
        float(np.max(values)),
    )

    if vmax <= vmin:
        return (
            max(
                0.0,
                vmin - 0.01,
            ),
            min(
                1.0,
                vmax + 0.01,
            ),
        )

    pad = (
        vmax - vmin
    ) * 0.08

    return (
        max(
            0.0,
            vmin - pad,
        ),
        min(
            1.0,
            vmax + pad,
        ),
    )


def compute_adaptive_xlim(
    values,
    step=0.5,
):
    """
    Calculate the -log10(FDR) x-axis range using only the values
    in the current CGI-U or CGI-D panel.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return 0.0, 1.0

    vmin = float(
        np.min(values)
    )

    vmax = float(
        np.max(values)
    )

    value_range = vmax - vmin

    pad = max(
        0.15,
        value_range * 0.12,
    )

    xmin = max(
        0.0,
        vmin - pad,
    )

    xmax = vmax + pad

    xmin = (
        np.floor(xmin / step)
        * step
    )

    xmax = (
        np.ceil(xmax / step)
        * step
    )

    if xmax <= xmin:
        xmax = xmin + step

    return xmin, xmax


def set_full_box(
    ax,
    lw=0.5,
):
    """
    Draw a complete rectangular border around an axes.
    """
    for side in [
        "left",
        "right",
        "top",
        "bottom",
    ]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(lw)
        ax.spines[side].set_color("black")


def draw_right_aligned_ylabels(
    ax,
    y_positions,
    labels,
    x_axes=-0.10,
):
    """
    Draw right-aligned pathway names to the left of a panel.
    """
    transform = ax.get_yaxis_transform()

    for y, label in zip(
        y_positions,
        labels,
    ):
        ax.text(
            x_axes,
            y,
            label,
            transform=transform,
            ha="right",
            va="center",
            multialignment="right",
            linespacing=0.92,
            fontsize=matplotlib.rcParams[
                "ytick.labelsize"
            ],
            color="black",
            fontweight="normal",
            clip_on=False,
        )


def read_observed_cgi_rows(
    kg_file,
    target_head,
    chunksize,
):
    """
    Read observed CGI-U and CGI-D rows for one compound from the KG.
    """
    usecols = [
        "head",
        "relation",
        "tail",
        "head_type",
        "tail_type",
    ]

    rows = []
    compound_present = False

    reader = pd.read_csv(
        kg_file,
        usecols=usecols,
        dtype={
            "head": "string",
            "relation": "string",
            "tail": "string",
            "head_type": "string",
            "tail_type": "string",
        },
        chunksize=chunksize,
        low_memory=False,
    )

    for i, chunk in enumerate(
        reader,
        start=1,
    ):
        print(
            f"[INFO] Reading observed chunk {i}",
            flush=True,
        )

        for col in usecols:
            chunk[col] = (
                chunk[col]
                .astype("string")
                .str.strip()
            )

        compound_present = compound_present or bool(
            (
                (chunk["head"] == str(target_head))
                & (chunk["head_type"] == "compound")
            ).any()
        )

        sub = chunk[
            (chunk["head"] == str(target_head))
            & chunk["relation"].isin(
                ["CGI-U", "CGI-D"]
            )
            & (
                chunk["head_type"]
                == "compound"
            )
            & (
                chunk["tail_type"]
                == "gene"
            )
        ].copy()

        if not sub.empty:
            rows.append(sub)

    if not rows:
        return (
            pd.DataFrame(columns=usecols),
            compound_present,
        )

    observed_df = (
        pd.concat(
            rows,
            ignore_index=True,
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return observed_df, compound_present


def extract_gene_lists(cgi_df):
    """
    Extract unique uppercase gene lists for CGI-U and CGI-D.
    """
    gene_lists = {}

    for relation in [
        "CGI-U",
        "CGI-D",
    ]:
        genes = (
            cgi_df.loc[
                cgi_df["relation"] == relation,
                "tail",
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
        )

        genes = (
            genes[
                genes != ""
            ]
            .drop_duplicates()
            .tolist()
        )

        gene_lists[relation] = genes

    return gene_lists


def check_libraries(
    libraries,
    organism,
):
    """
    Check whether the requested Enrichr libraries are available.
    """
    try:
        available = gp.get_library_name(
            organism=organism
        )
    except Exception as exc:
        raise WorkflowError(
            "Could not retrieve Enrichr libraries. Check the organism "
            "name and network access to Enrichr, then retry."
        ) from exc

    missing = [
        library
        for library in libraries
        if library not in available
    ]

    if missing:
        raise WorkflowError(
            f"Unavailable Enrichr libraries for organism '{organism}': "
            + ", ".join(missing)
            + ". Use a library name returned by "
            "gseapy.get_library_name() for the same organism."
        )


def empty_enrichment_result():
    return pd.DataFrame(columns=EMPTY_ENRICHMENT_COLUMNS)


def run_enrichr(
    gene_list,
    gene_set,
    organism,
):
    """Run Enrichr and standardize the columns used for plotting."""
    if not gene_list:
        return empty_enrichment_result()

    try:
        enrichment = gp.enrichr(
            gene_list=list(gene_list),
            gene_sets=gene_set,
            organism=organism,
            outdir=None,
            cutoff=1.0,
            no_plot=True,
        )
    except Exception as exc:
        message = str(exc).casefold()
        no_result_markers = (
            "no hits returned",
            "no enriched terms",
            "no results",
            "no genes matched",
        )
        if any(marker in message for marker in no_result_markers):
            print(
                f"[WARN] Enrichr returned no results for {gene_set}.",
                flush=True,
            )
            return empty_enrichment_result()
        raise WorkflowError(
            f"Enrichr request failed for library '{gene_set}'. Check "
            "network access and the Enrichr service, then retry."
        ) from exc

    raw_result = getattr(enrichment, "results", None)
    if raw_result is None or raw_result.empty:
        print(
            f"[WARN] Enrichr returned no results for {gene_set}.",
            flush=True,
        )
        return empty_enrichment_result()

    result = raw_result.copy().rename(
        columns={
            "Adjusted P-value": "Adjusted_P_value",
            "P-value": "P_value",
            "Odds Ratio": "Odds_Ratio",
            "Combined Score": "Combined_Score",
        }
    )

    required = {"Term", "Adjusted_P_value", "P_value", "Overlap"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise WorkflowError(
            "Enrichr response is missing required columns: "
            + ", ".join(missing)
        )

    parsed = result["Overlap"].apply(parse_overlap)
    result["Gene_Count"] = parsed.apply(lambda value: value[0])
    result["Gene_Ratio"] = parsed.apply(lambda value: value[1])

    for col in (
        "Adjusted_P_value",
        "P_value",
        "Gene_Count",
        "Gene_Ratio",
    ):
        result[col] = pd.to_numeric(result[col], errors="coerce")

    result["minus_log10_FDR"] = -np.log10(
        result["Adjusted_P_value"].clip(lower=1e-300)
    )
    result["Term_key"] = result["Term"].apply(make_term_key)

    return result.sort_values(
        ["Adjusted_P_value", "P_value"],
        ascending=[True, True],
    ).reset_index(drop=True)


def validate_enrichment_df(df):
    required_columns = [
        "Term",
        "Adjusted_P_value",
        "P_value",
        "Gene_Count",
        "Gene_Ratio",
        "minus_log10_FDR",
    ]

    if df is None or df.empty:
        return False

    return all(
        col in df.columns
        for col in required_columns
    )


def prepare_observed_top_df(
    df,
    top_terms,
    term_width,
):
    """
    Select the observed Top N pathways and reverse their order so that
    the most significant pathway is displayed at the top.
    """
    if not validate_enrichment_df(df):
        return pd.DataFrame()

    out = df.copy()

    if "Term_key" not in out.columns:
        out["Term_key"] = out[
            "Term"
        ].apply(make_term_key)

    out = (
        out.dropna(
            subset=[
                "Term",
                "Adjusted_P_value",
                "P_value",
                "Gene_Count",
                "Gene_Ratio",
                "minus_log10_FDR",
            ]
        )
        .sort_values(
            [
                "Adjusted_P_value",
                "P_value",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .drop_duplicates(
            subset=["Term_key"],
            keep="first",
        )
        .head(top_terms)
        .copy()
    )

    if out.empty:
        return out

    out["Term_display"] = out[
        "Term"
    ].apply(
        lambda term: wrap_term(
            term,
            width=term_width,
        )
    )

    return (
        out.iloc[::-1]
        .reset_index(drop=True)
    )


def draw_one_dotplot_panel(
    ax,
    plot_df,
    title,
    xmin,
    xmax,
    cmin,
    cmax,
    cmap,
    size_mapper,
    ylabel_x=-0.10,
):
    """
    Draw one observed enrichment bubble-plot panel.
    """
    if (
        plot_df is None
        or plot_df.empty
    ):
        ax.text(
            0.5,
            0.5,
            "No enriched terms",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.set_title(
            title,
            loc="center",
            pad=2,
            fontweight="bold",
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        set_full_box(
            ax,
            lw=0.5,
        )

        return None

    y_positions = np.arange(
        len(plot_df),
        dtype=float,
    )

    x_values = pd.to_numeric(
        plot_df[
            "minus_log10_FDR"
        ],
        errors="coerce",
    )

    count_values = pd.to_numeric(
        plot_df[
            "Gene_Count"
        ],
        errors="coerce",
    )

    ratio_values = pd.to_numeric(
        plot_df[
            "Gene_Ratio"
        ],
        errors="coerce",
    )

    valid_mask = (
        x_values.notna()
        & count_values.notna()
        & ratio_values.notna()
        & np.isfinite(
            x_values.to_numpy()
        )
        & np.isfinite(
            count_values.to_numpy()
        )
        & np.isfinite(
            ratio_values.to_numpy()
        )
    )

    scatter = None

    if valid_mask.any():
        valid_y = y_positions[
            valid_mask.to_numpy()
        ]

        scatter = ax.scatter(
            x_values.loc[
                valid_mask
            ].astype(float).values,
            valid_y,
            s=size_mapper(
                count_values.loc[
                    valid_mask
                ].astype(float).values
            ),
            c=ratio_values.loc[
                valid_mask
            ].astype(float).values,
            cmap=cmap,
            vmin=cmin,
            vmax=cmax,
            edgecolors="#4d4d4d",
            linewidth=0.28,
            zorder=3,
        )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        [""] * len(y_positions)
    )

    draw_right_aligned_ylabels(
        ax=ax,
        y_positions=y_positions,
        labels=plot_df[
            "Term_display"
        ].tolist(),
        x_axes=ylabel_x,
    )

    ax.set_xlim(
        xmin,
        xmax,
    )

    ax.set_ylim(
        -0.7,
        len(plot_df) - 0.3,
    )

    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=3)
    )

    ax.xaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

    set_full_box(
        ax,
        lw=0.5,
    )

    ax.tick_params(
        axis="x",
        direction="out",
        length=2.0,
        width=0.5,
        pad=1.2,
    )

    ax.tick_params(
        axis="y",
        which="major",
        direction="out",
        left=True,
        right=False,
        length=2.0,
        width=0.5,
        pad=0,
        labelleft=False,
    )

    ax.grid(False)

    ax.set_title(
        title,
        loc="center",
        pad=2,
        fontweight="bold",
    )

    ax.set_xlabel(
        "-log10(FDR)",
        labelpad=1.2,
    )

    return scatter


def plot_observed_single_dotplot(
    enrichment_df,
    relation,
    library,
    outdir,
    target_head,
    top_terms,
    fig_width_cm,
    fig_height_cm,
    term_width,
    cmap="plasma",
):
    """
    Draw and save one independent CGI-U or CGI-D SVG figure.

    Each figure independently determines:
        - the -log10(FDR) x-axis range;
        - the Count-to-bubble-size mapping;
        - the Count legend values;
        - the Gene Ratio color range;
        - the Gene Ratio colorbar ticks.

    The physical bubble-panel dimensions from the original script
    are retained:
        panel_width = 0.075
        panel_height = 0.30
    """

    plot_df = prepare_observed_top_df(
        df=enrichment_df,
        top_terms=top_terms,
        term_width=term_width,
    )

    if plot_df.empty:
        print(
            f"[WARN] No plottable enriched terms for "
            f"{relation}, {library}. Skip plotting.",
            flush=True,
        )
        return

    x_values = pd.to_numeric(
        plot_df.get(
            "minus_log10_FDR",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    ).to_numpy()

    count_values = pd.to_numeric(
        plot_df.get(
            "Gene_Count",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    ).to_numpy()

    ratio_values = pd.to_numeric(
        plot_df.get(
            "Gene_Ratio",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    ).to_numpy()

    finite_x = x_values[
        np.isfinite(x_values)
    ]

    finite_count = count_values[
        np.isfinite(count_values)
        & (count_values > 0)
    ]

    finite_ratio = ratio_values[
        np.isfinite(ratio_values)
    ]

    xmin, xmax = compute_adaptive_xlim(
        finite_x,
        step=0.5,
    )

    cmin, cmax = compute_color_limits(
        finite_ratio
    )

    count_breaks = adaptive_count_breaks(
        finite_count,
        narrow_range_threshold=30,
    )

    # Construct one shared size-mapping range using both the observed
    # counts and the displayed legend values. This guarantees that
    # every legend circle has exactly the same size mapping as a data
    # bubble with the corresponding Count value.
    if count_breaks:
        size_reference_values = np.concatenate(
            [
                finite_count,
                np.asarray(
                    count_breaks,
                    dtype=float,
                ),
            ]
        )
    else:
        size_reference_values = finite_count

    size_mapper = CountSizeMapper(
        size_reference_values,
        min_size=12,
        max_size=85,
    )

    print(
        f"[INFO] {relation}, {library}: "
        f"-log10(FDR) range = {xmin:.2f} to {xmax:.2f}",
        flush=True,
    )

    if len(finite_count) > 0:
        print(
            f"[INFO] {relation}, {library}: "
            f"Observed Count range = "
            f"{np.min(finite_count):.0f} to "
            f"{np.max(finite_count):.0f}",
            flush=True,
        )

    print(
        f"[INFO] {relation}, {library}: "
        f"Count legend values = "
        f"{', '.join(str(value) for value in count_breaks)}",
        flush=True,
    )

    print(
        f"[INFO] {relation}, {library}: "
        f"Count size-mapping range = "
        f"{size_mapper.vmin:.0f} to "
        f"{size_mapper.vmax:.0f}",
        flush=True,
    )

    print(
        f"[INFO] {relation}, {library}: "
        f"Gene Ratio range = {cmin:.4f} to {cmax:.4f}",
        flush=True,
    )

    fig = plt.figure(
        figsize=(
            cm_to_inch(fig_width_cm),
            cm_to_inch(fig_height_cm),
        ),
        facecolor="white",
    )

    # Retain the original single-panel physical dimensions.
    panel_bottom = 0.13
    panel_height = 0.30
    panel_width = 0.075

    # Both CGI-U and CGI-D use the same position in their own
    # independent figures.
    ax_panel = fig.add_axes(
        [
            0.285,
            panel_bottom,
            panel_width,
            panel_height,
        ]
    )

    # Use separate axes for the Count legend and Gene Ratio colorbar
    # to prevent them from overlapping.
    ax_count_legend = fig.add_axes(
        [
            0.500,
            0.43,
            0.120,
            0.30,
        ]
    )

    ax_count_legend.axis("off")

    colorbar_ax = fig.add_axes(
        [
            0.525,
            0.16,
            0.012,
            0.20,
        ]
    )

    draw_one_dotplot_panel(
        ax=ax_panel,
        plot_df=plot_df,
        title=relation,
        xmin=xmin,
        xmax=xmax,
        cmin=cmin,
        cmax=cmax,
        cmap=cmap,
        size_mapper=size_mapper,
        ylabel_x=-0.10,
    )

    fig.text(
        0.02,
        0.975,
        (
            f"Compound {target_head}: observed {relation} "
            f"enrichment ({library})"
        ),
        ha="left",
        va="top",
        fontsize=matplotlib.rcParams[
            "axes.titlesize"
        ],
        fontweight="bold",
    )

    count_handles = []
    count_labels = []

    for count_value in count_breaks:
        handle = ax_count_legend.scatter(
            [],
            [],
            s=size_mapper(
                [count_value]
            )[0],
            facecolors="black",
            edgecolors="black",
            linewidths=0.25,
        )

        count_handles.append(
            handle
        )

        count_labels.append(
            str(int(count_value))
        )

    if count_handles:
        ax_count_legend.legend(
            count_handles,
            count_labels,
            title="Count",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(
                0.00,
                1.00,
            ),
            borderaxespad=0.0,
            handletextpad=0.45,
            labelspacing=0.85,
            handleheight=1.20,
            handlelength=1.2,
            markerscale=1.0,
        )



    norm = matplotlib.colors.Normalize(
        vmin=cmin,
        vmax=cmax,
    )

    scalar_mappable = matplotlib.cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    scalar_mappable.set_array([])

    colorbar = fig.colorbar(
        scalar_mappable,
        cax=colorbar_ax,
    )

    colorbar.set_label(
        "Gene ratio",
        labelpad=1.2,
    )

    colorbar.ax.tick_params(
        labelsize=5,
        width=0.5,
        length=1.7,
        pad=1,
    )

    colorbar.outline.set_linewidth(
        0.5
    )

    colorbar.locator = MaxNLocator(
        nbins=4
    )

    colorbar.update_ticks()

    colorbar.ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.2f")
    )

    out_prefix = os.path.join(
        outdir,
        (
            f"head_{sanitize_filename(target_head)}_"
            f"observed_{sanitize_filename(relation)}_"
            f"{sanitize_filename(library)}_"
            f"single_dotplot"
        ),
    )

    fig.savefig(
        out_prefix + ".svg",
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )

    plt.close(fig)

    print(
        f"[INFO] Saved {out_prefix}.svg",
        flush=True,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run Enrichr pathway analysis for observed BioNexKG CGI-U "
            "and CGI-D genes and create separate SVG bubble plots."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kg",
        "--kg-file",
        dest="kg",
        default=str(DEFAULT_KG),
        help=(
            "BioNexKG CSV containing head, relation, tail, head_type, "
            "and tail_type columns."
        ),
    )
    parser.add_argument(
        "--head",
        "--compound-id",
        dest="head",
        default=DEFAULT_HEAD,
        help="Compound/head ID; matched exactly after trimming whitespace.",
    )
    parser.add_argument(
        "--outdir",
        "--output-directory",
        dest="outdir",
        default=DEFAULT_OUTDIR,
        help="Directory for observed rows, gene lists, enrichment CSVs, and SVGs.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500000,
        help="KG rows to read per chunk.",
    )
    parser.add_argument(
        "--organism",
        default="human",
        help="Organism name accepted by gseapy/Enrichr.",
    )
    parser.add_argument(
        "--libraries",
        nargs="+",
        default=["MSigDB_Hallmark_2020"],
        help="One or more Enrichr library names.",
    )
    parser.add_argument(
        "--top_terms",
        "--top-terms",
        dest="top_terms",
        type=int,
        default=5,
        help="Maximum pathways shown in each SVG.",
    )
    parser.add_argument(
        "--cmap",
        default="plasma",
        help="Matplotlib colormap used for Gene Ratio.",
    )
    parser.add_argument(
        "--font_pt",
        "--font-pt",
        dest="font_pt",
        type=float,
        default=5.5,
        help="Base plot font size in points.",
    )
    parser.add_argument(
        "--fig_width_cm",
        "--fig-width-cm",
        dest="fig_width_cm",
        type=float,
        default=18.0,
        help="SVG figure width in centimeters.",
    )
    parser.add_argument(
        "--fig_height_cm",
        "--fig-height-cm",
        dest="fig_height_cm",
        type=float,
        default=7.2,
        help="SVG figure height in centimeters.",
    )
    parser.add_argument(
        "--term_width",
        "--term-width",
        dest="term_width",
        type=int,
        default=24,
        help="Approximate pathway-label wrap width.",
    )
    return parser


def validate_args(args):
    args.head = str(args.head).strip()
    args.organism = str(args.organism).strip().lower()
    args.libraries = list(
        dict.fromkeys(
            library.strip()
            for library in args.libraries
            if library.strip()
        )
    )

    if not args.head:
        raise WorkflowError("--head/--compound-id must not be empty.")
    if not args.organism:
        raise WorkflowError("--organism must not be empty.")
    if not args.libraries:
        raise WorkflowError("--libraries must contain at least one library name.")
    if args.chunksize < 1:
        raise WorkflowError("--chunksize must be at least 1.")
    if args.top_terms < 1:
        raise WorkflowError("--top_terms must be at least 1.")
    if args.font_pt <= 0:
        raise WorkflowError("--font_pt must be greater than 0.")
    if args.fig_width_cm <= 0 or args.fig_height_cm <= 0:
        raise WorkflowError("Figure width and height must be greater than 0.")
    if args.term_width < 1:
        raise WorkflowError("--term_width must be at least 1.")

    kg_file = Path(args.kg).expanduser().resolve()
    if not kg_file.is_file():
        raise WorkflowError(f"KG file does not exist: {kg_file}")

    outdir = Path(args.outdir).expanduser().resolve()
    if outdir.exists() and not outdir.is_dir():
        raise WorkflowError(f"Output path exists but is not a directory: {outdir}")

    return kg_file, outdir


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        kg_file, outdir = validate_args(args)
        load_data_dependencies()

        print(f"[INFO] Input observed KG: {kg_file}", flush=True)
        print(f"[INFO] Target head: {args.head}", flush=True)

        try:
            observed_df, compound_present = read_observed_cgi_rows(
                kg_file=kg_file,
                target_head=args.head,
                chunksize=args.chunksize,
            )
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            required = ", ".join(sorted(REQUIRED_KG_COLUMNS))
            raise WorkflowError(
                f"Could not read KG CSV. Required columns: {required}."
            ) from exc

        if not compound_present:
            raise WorkflowError(
                f"Compound/head ID '{args.head}' was not found as a compound "
                f"head in {kg_file}."
            )

        observed_gene_lists = extract_gene_lists(observed_df)
        has_observed_genes = any(observed_gene_lists.values())

        if has_observed_genes:
            load_enrichment_dependency()
            check_libraries(args.libraries, args.organism)
            load_plotting_dependencies()
            try:
                plt.get_cmap(args.cmap)
            except ValueError as exc:
                raise WorkflowError(
                    f"Unknown Matplotlib colormap: {args.cmap}"
                ) from exc
            set_plot_style(base_fontsize=args.font_pt)

        ensure_dir(outdir)
        observed_file = outdir / (
            f"head_{sanitize_filename(args.head)}_observed_CGI_U_D_rows.csv"
        )
        observed_df.to_csv(observed_file, index=False)
        print(
            f"[INFO] Observed rows: {len(observed_df)}. Saved {observed_file}",
            flush=True,
        )

        for relation in RELATIONS:
            genes = observed_gene_lists[relation]
            gene_file = outdir / (
                f"head_{sanitize_filename(args.head)}_"
                f"observed_{sanitize_filename(relation)}_genes.txt"
            )
            with gene_file.open("w", encoding="utf-8") as file_handle:
                if genes:
                    file_handle.write("\n".join(genes) + "\n")
            print(
                f"[INFO] Observed {relation}: {len(genes)} unique genes. "
                f"Saved {gene_file}",
                flush=True,
            )

        all_results = {relation: {} for relation in RELATIONS}
        for relation in RELATIONS:
            genes = observed_gene_lists[relation]
            if not genes:
                print(
                    f"[WARN] No observed genes found for {relation}. "
                    "Skip enrichment and plotting for this relation.",
                    flush=True,
                )
                continue

            for library in args.libraries:
                print(
                    f"[INFO] Running Enrichr: observed {relation}, {library}",
                    flush=True,
                )
                result = run_enrichr(
                    gene_list=genes,
                    gene_set=library,
                    organism=args.organism,
                )
                output_csv = outdir / (
                    f"head_{sanitize_filename(args.head)}_"
                    f"observed_{sanitize_filename(relation)}_"
                    f"{sanitize_filename(library)}_enrichr.csv"
                )
                result.to_csv(output_csv, index=False)
                all_results[relation][library] = result
                print(f"[INFO] Saved {output_csv}", flush=True)

        if has_observed_genes:
            for library in args.libraries:
                for relation in RELATIONS:
                    plot_observed_single_dotplot(
                        enrichment_df=all_results[relation].get(
                            library,
                            empty_enrichment_result(),
                        ),
                        relation=relation,
                        library=library,
                        outdir=outdir,
                        target_head=args.head,
                        top_terms=args.top_terms,
                        fig_width_cm=args.fig_width_cm,
                        fig_height_cm=args.fig_height_cm,
                        term_width=args.term_width,
                        cmap=args.cmap,
                    )

        print("[INFO] Done.", flush=True)
        return 0
    except WorkflowError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 2
    except OSError as exc:
        print(f"[ERROR] File operation failed: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
