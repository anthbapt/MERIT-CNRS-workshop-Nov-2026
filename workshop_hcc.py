"""Helpers for the MERIT workshop hands-on (HCC core, Xenium + COMET + H&E).

Built on multispace (cell level: data contract, registration, cell typing, spatial
analysis) and spatiomic (pixel level: SOM + Leiden clustering, vicinity graphs).

Everything here takes and returns plain objects so the notebook stays readable.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*validate_table_in_spatialdata.*")
warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger("workshop_hcc")

# Xenium µm → COMET px scale is ~3.581 (from the fitted affine), so 1 COMET px ≈ 0.2792 µm.
COMET_UM_PER_PX = 1 / 3.58111

# RNA–protein pairs with a matching gene in the Xenium Prime 5K panel
# (COMET channel name → gene). 9 of the 15 markers have a direct counterpart in the panel.
RNA_PROTEIN_PAIRS = {
    "CD20 - TRITC": "MS4A1",
    "CD206 - TRITC": "MRC1",
    "HepPar1 - TRITC": "CPS1",
    "E-Cadherin - TRITC": "CDH1",
    "podoplanin - FITC": "PDPN",
    "VEGF - TRITC": "VEGFA",
    "CD45 - TRITC": "PTPRC",
    "CD1c - TRITC": "CD1C",
    "IL-10 - TRITC": "IL10",
    "FXIIIa - FITC": "F13A1",
    "CD66 - FITC": "CEACAM1",
    "αSMA - TRITC": "ACTA2",
    "vimentin - TRITC": "VIM",
    "CK8 - TRITC": "KRT8",
    "CD45RA - TRITC": "PTPRC",
}

# Coarse liver / HCC markers restricted to genes present in the Prime 5K panel
# (ALB, APOA1, KRT19, COL1A1, VWF, PTPRC, ... are not in the panel).
LIVER_MARKERS = {
    "Hepatocyte-tumour": ["APOB", "AHSG", "HNF4A", "ARG1", "IGF2", "DLK1", "PEG10"],
    "Cholangiocyte-EPCAM": ["EPCAM", "SOX9", "CDH1", "CFTR"],
    "Endothelial": ["PECAM1", "PLVAP", "CD34", "KDR", "ENG", "CDH5", "STAB1"],
    "Fibroblast-stellate": ["PDGFRB", "PDGFRA", "RGS5", "THY1", "COL4A1", "RELN", "FAP"],
    "Macrophage": ["CD68", "CD163", "CD14", "MRC1", "MSR1", "CSF1R"],
    "T cell": ["CD3E", "CD2", "CD8A", "CD4", "CD247", "LCK"],
    "B cell": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1"],
    "Plasma cell": ["MZB1", "DERL3", "TNFRSF17"],
    "NK cell": ["KLRD1", "KLRF1", "PRF1", "GZMB"],
    "Neutrophil": ["FCGR3B", "CSF3R", "CEACAM8"],
    "Dendritic cell": ["CD1C", "CLEC10A", "CLEC9A", "LAMP3"],
}


# Signed markers for the cluster-level marker strategy (multispace celltyping, Pillar 1:
# annotate_clusters_by_markers). Net score = positive score − negative score. Only genes in
# the Xenium Prime 5K panel; ALB, APOA1, KRT19, COL1A1, VWF, PTPRC, ... are not in it.
_EPI, _IMM = ["APOB", "AHSG", "EPCAM"], ["CD3E", "CD68", "MS4A1", "CD79A"]
LIVER_MARKERS_SIGNED = {
    "Hepatocyte": {"positive": ["APOB", "AHSG", "ARG1", "CPS1", "HNF4A", "HNF1A", "PCK1", "CYP2C9", "SLC10A1"],
                   "negative": _IMM},
    "HCC tumour": {"positive": ["IGF2", "DLK1", "PEG10", "LGR5", "H19"], "negative": _IMM + ["CPS1", "ARG1"]},
    "Cholangiocyte": {"positive": ["EPCAM", "SOX9", "CFTR", "CDH1"], "negative": _IMM + ["APOB"]},
    "Endothelial": {"positive": ["PECAM1", "PLVAP", "CD34", "KDR", "ENG", "CDH5", "EGFL7", "CLDN5"],
                    "negative": _EPI + _IMM},
    "Fibroblast / stellate": {"positive": ["PDGFRB", "PDGFRA", "RGS5", "THY1", "COL4A1", "RELN", "FAP", "HGF"],
                              "negative": _EPI + _IMM},
    "Macrophage": {"positive": ["CD68", "CD163", "CD14", "MRC1", "MSR1", "CSF1R", "MARCO", "VSIG4"],
                   "negative": _EPI + ["CD3E", "MS4A1"]},
    "T cell": {"positive": ["CD3E", "CD2", "CD8A", "CD4", "CD247", "LCK", "GZMK"],
               "negative": _EPI + ["CD68", "MS4A1", "CD79A"]},
    "B cell": {"positive": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "PAX5", "CR2"],
               "negative": _EPI + ["CD3E", "CD68"]},
    "Plasma cell": {"positive": ["MZB1", "DERL3", "TNFRSF17"], "negative": _EPI + ["MS4A1"]},
    "NK cell": {"positive": ["KLRD1", "KLRF1", "PRF1", "GZMB", "NCAM1"], "negative": _EPI + ["CD3E"]},
    "Neutrophil": {"positive": ["FCGR3B", "CSF3R", "CXCR2", "CEACAM8"], "negative": _EPI},
    "Dendritic cell": {"positive": ["CD1C", "CLEC10A", "CLEC9A", "LAMP3", "FCER1A", "XCR1"], "negative": _EPI},
}


def annotate_by_markers(adata, markers=LIVER_MARKERS_SIGNED, resolution: float = 2.0, out_dir=None,
                        min_score: float = 0.0, ambiguity_threshold: float = 0.5, seed: int = 0):
    """Marker strategy: Leiden clusters on RNA, signed marker scores per cluster, one label per cluster.

    Uses ``multispace.celltyping.cluster_annotation.annotate_clusters_by_markers`` and writes the
    multispace contract columns: ``cell_type`` (cluster label), ``cell_type_confidence``
    (winner / runner-up score ratio of the cluster), ``cell_type_cluster`` and ``cell_type_score``.
    Resolution 2.0 is needed on core01 to separate the B-cell follicle from the T cells.
    Returns the cluster → label table.
    """
    import scanpy as sc
    from multispace.celltyping.cluster_annotation import annotate_clusters_by_markers

    a = adata.copy()
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    a.layers["lognorm"] = a.X.copy()
    sc.pp.pca(a, n_comps=30, random_state=seed)
    sc.pp.neighbors(a, n_neighbors=15, random_state=seed)
    sc.tl.umap(a, random_state=seed)
    a.obsm["X_umap_corrected"] = a.obsm["X_umap"]  # the function plots this basis
    sc.tl.leiden(a, resolution=resolution, key_added="leiden_markers", flavor="igraph", n_iterations=2,
                 random_state=seed)
    a, table = annotate_clusters_by_markers(a, str(out_dir or "cell_typing_markers"), marker_dict=markers,
                                            cluster_key="leiden_markers", output_col="cell_type_markers",
                                            min_score=min_score, ambiguity_threshold=ambiguity_threshold,
                                            layer="lognorm")
    cl = a.obs["leiden_markers"].astype(str)
    table.index = table.index.astype(str)
    adata.obs["cell_type"] = pd.Categorical(a.obs["cell_type_markers"].astype(str).to_numpy())
    adata.obs["cell_type_cluster"] = pd.Categorical(cl.to_numpy())
    adata.obs["cell_type_confidence"] = cl.map(table["ambiguity_ratio"]).astype(float).to_numpy()
    adata.obs["cell_type_score"] = cl.map(table["winner_score"]).astype(float).to_numpy()
    adata.obsm["X_umap"] = a.obsm["X_umap"]
    return table


# Protein evidence for the multi-omics marker strategy (COMET short channel names).
# Lineage: the compartment, where the 5K panel is weakest (no PTPRC, ALB, KRT19, COL1A1). Each cell type
# gets  lineage(own) - mean(lineage(others)), a contrast that cancels a cell's overall brightness.
# Specific: one extra protein where the panel has one; scored as own - mean(specific of the same lineage),
# so a type without a specific protein (T, NK, plasma, endothelial, all epithelial types) is "none of the others".
PROTEIN_LINEAGES = {"immune": ["CD45"], "epithelial": ["CK8", "E-Cadherin", "HepPar1"], "stromal": ["vimentin"]}
CELL_TYPE_LINEAGE = {"Hepatocyte": "epithelial", "HCC tumour": "epithelial", "Cholangiocyte": "epithelial",
                     "Endothelial": "stromal", "Fibroblast / stellate": "stromal",
                     "Macrophage": "immune", "T cell": "immune", "B cell": "immune", "Plasma cell": "immune",
                     "NK cell": "immune", "Neutrophil": "immune", "Dendritic cell": "immune"}
# CD66 is left out on purpose: the antibody also sees CEACAM1 (activated T cells, hepatocyte canaliculi). With it
# as a neutrophil marker, ~750 cells of the lymphoid aggregate (T cells by RNA) were called neutrophils.
PROTEIN_SPECIFIC = {"B cell": ["CD20", "CD45RA"], "Macrophage": ["CD206", "FXIIIa"], "Dendritic cell": ["CD1c"],
                    "Fibroblast / stellate": ["αSMA"]}


def protein_marker_scores(pz: pd.DataFrame, lineages=PROTEIN_LINEAGES, cell_lineage=CELL_TYPE_LINEAGE,
                          specific=PROTEIN_SPECIFIC, specific_weight: float = 0.5) -> pd.DataFrame:
    """Signed protein score per cell and cell type (see PROTEIN_LINEAGES). ``pz``: output of protein_z.
    Each cell's mean z over all channels is removed first: bright or autofluorescent cells are high in every
    channel, and without this a type defined by "none of the other markers" (T cells) loses to any type with
    a specific marker in those cells."""
    pz = pz.sub(pz.mean(axis=1), axis=0)
    lin = pd.DataFrame({k: pz[v].mean(axis=1) for k, v in lineages.items()})
    spec = pd.DataFrame({k: pz[v].mean(axis=1) for k, v in specific.items()})
    out = {}
    for ct, l in cell_lineage.items():
        score = lin[l] - lin.drop(columns=l).mean(axis=1)
        group = [c for c in spec if cell_lineage.get(c) == l]
        if group:
            own = spec[ct] if ct in spec else 0.0
            score = score + specific_weight * (own - spec[group].mean(axis=1))
        out[ct] = score
    return pd.DataFrame(out, index=pz.index)


def protein_z(adata, clip=(-3.0, 5.0)) -> pd.DataFrame:
    """Per-cell COMET mean intensities (``obsm['protein']``, DAPI dropped): log1p, z-score per channel, clip.
    Columns are renamed to the short channel name (``'CD45 - TRITC'`` → ``'CD45'``)."""
    p = adata.obsm["protein"].drop(columns=["DAPI"], errors="ignore").astype(float)
    p.columns = [c.split(" - ")[0] for c in p.columns]
    p = np.log1p(p.clip(lower=0))
    return ((p - p.mean()) / p.std()).clip(*clip)


def annotate_multiomic(adata, rna_markers=LIVER_MARKERS_SIGNED,
                       resolution: float = 2.0, protein_weight: float = 0.5, ambiguity_threshold: float = 0.5,
                       min_score: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Marker strategy on RNA + protein: the multispace marker logic, with protein evidence added.

    1. Joint clusters: RNA PCs (30, log-normalised counts) and protein z-scores, each block scaled to unit
       total variance, concatenated; kNN graph and Leiden at the same resolution as the RNA-only typing.
    2. Per cell and cell type: the signed RNA score from multispace (``_score_signed_markers``: score_genes on
       positive minus negative genes) and a signed protein score (``protein_marker_scores``: lineage
       contrast plus a specific protein where the panel has one).
       Each score matrix is divided by its overall standard deviation so the two modalities share a scale.
    3. Combined score = (1 - w) · RNA + w · protein (w = ``protein_weight``).
    4. Per cluster: mean combined score, the same quality gates as ``annotate_clusters_by_markers``
       (``min_score``, winner / runner-up ``ambiguity_threshold``), one label per cluster.

    Writes ``obs['cell_type_multiomic']``, ``obs['multiomic_cluster']``, ``obs['multiomic_confidence']``
    and ``obsm['marker_scores_rna' | 'marker_scores_protein']``. Returns the cluster table, with the winner
    of each modality alone so you can see where RNA and protein disagree.
    """
    import scanpy as sc
    from sklearn.decomposition import PCA
    from multispace.celltyping.cluster_annotation import _score_signed_markers

    a = adata.copy()
    a.X = a.layers["counts"].copy()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    a.layers["lognorm"] = a.X.copy()
    pz = protein_z(adata)

    # 1 · joint clusters
    rna_pcs = PCA(30, random_state=seed).fit_transform(np.asarray(a.X.todense()) if hasattr(a.X, "todense")
                                                         else np.asarray(a.X))
    blocks = [rna_pcs / np.sqrt(rna_pcs.var(axis=0).sum()), pz.to_numpy() / np.sqrt(pz.var(axis=0).sum())]
    a.obsm["X_rna_protein"] = np.hstack(blocks)
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X_rna_protein", random_state=seed)
    sc.tl.leiden(a, resolution=resolution, key_added="multiomic_cluster", flavor="igraph", n_iterations=2,
                 random_state=seed)

    # 2 · per-cell scores
    rna = {}
    for ct, entry in rna_markers.items():
        if _score_signed_markers(a, ct, entry, prefix="_mk_", layer="lognorm"):
            rna[ct] = a.obs.pop(f"_mk_{ct}").to_numpy()
    rna = pd.DataFrame(rna, index=a.obs_names)
    prot = protein_marker_scores(pz).loc[:, lambda d: [c for c in d if c in rna]]
    rna_s, prot_s = rna / np.nanstd(rna.to_numpy()), prot / np.nanstd(prot.to_numpy())

    # 3 · combined score
    comb = rna_s.copy()
    for ct in prot_s:
        if ct in comb:
            comb[ct] = (1 - protein_weight) * rna_s[ct] + protein_weight * prot_s[ct]

    # 4 · one label per cluster, same gates as multispace
    cl = a.obs["multiomic_cluster"].astype(str)
    means = comb.groupby(cl.to_numpy()).mean()
    rows = {}
    for c, row in means.iterrows():
        srt = row.sort_values(ascending=False)
        w, r = srt.iloc[0], srt.iloc[1]
        ratio = w / r if r > 1e-6 else np.inf
        label = "Unresolved" if w < min_score else ("Ambiguous" if ratio < ambiguity_threshold else srt.index[0])
        rows[c] = {"cell_type_multiomic": label, "winner_score": w, "runner_up_score": r,
                   "ambiguity_ratio": ratio if np.isfinite(ratio) else np.nan,
                   "rna_winner": rna_s.groupby(cl.to_numpy()).mean().loc[c].idxmax(),
                   "protein_winner": prot_s.groupby(cl.to_numpy()).mean().loc[c].idxmax(),
                   "n_cells": int((cl == c).sum())}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "multiomic_cluster"
    adata.obs["multiomic_cluster"] = pd.Categorical(cl.to_numpy())
    adata.obs["cell_type_multiomic"] = pd.Categorical(cl.map(table["cell_type_multiomic"]).to_numpy())
    adata.obs["multiomic_confidence"] = cl.map(table["ambiguity_ratio"]).astype(float).to_numpy()
    adata.obsm["marker_scores_rna"], adata.obsm["marker_scores_protein"] = rna_s, prot_s
    return table


def segmentation_match(sdata, xenium="cell_boundaries", comet="comet_cell_masks") -> pd.DataFrame:
    """For each Xenium cell, its best-overlapping COMET cell (IoU) and whether the pair is a mutual best
    match with IoU > 0.5 (``one_to_one``). Both segmentations are compared in the global coordinate system."""
    from collections import defaultdict
    from shapely.affinity import affine_transform
    from shapely.strtree import STRtree
    from spatialdata.transformations import get_transformation

    def to_global(key):
        g = sdata.shapes[key]
        M = get_transformation(g, to_coordinate_system="global").to_affine_matrix(("x", "y"), ("x", "y"))
        coef = [M[0, 0], M[0, 1], M[1, 0], M[1, 1], M[0, 2], M[1, 2]]
        return g.index, g.geometry.apply(lambda p: affine_transform(p, coef)).buffer(0).values

    ids, X = to_global(xenium)
    _, C = to_global(comet)
    tree = STRtree(C)
    best, bestj = np.zeros(len(X)), -np.ones(len(X), int)
    for i, p in enumerate(X):
        for j in tree.query(p):
            inter = p.intersection(C[j]).area
            if inter > 0:
                iou = inter / p.union(C[j]).area
                if iou > best[i]:
                    best[i], bestj[i] = iou, j
    owner = defaultdict(lambda: (0.0, -1))
    for i, (v, j) in enumerate(zip(best, bestj)):
        if j >= 0 and v > owner[j][0]:
            owner[j] = (v, i)
    one = np.array([v > 0.5 and owner[j][1] == i for i, (v, j) in enumerate(zip(best, bestj))])
    return pd.DataFrame({"best_iou": best, "one_to_one": one}, index=ids)


# ── 1 · data contract ──────────────────────────────────────────────────────────
def to_multispace(sdata, core: str = "core01_hcc", keep_reference_he: bool = True):
    """Rename the core's elements to the multispace contract.

    * COMET (16 channels) → ``images['if']``, Identity to ``global`` (the reference grid);
      its DAPI channel → ``images['morphology_focus']`` (what multispace registers onto).
    * H&E → ``images['he']`` in its own coordinate system only, ready for
      ``ms.register_image``; the provided keypoint-based transform is kept on the
      same element under the coordinate system ``he_reference`` for comparison.
    * Xenium transcripts / cells / nuclei → ``points['transcripts']``,
      ``shapes['cell_boundaries']``, ``shapes['nucleus_boundaries']``.
    * COMET-native masks → ``shapes['comet_cell_masks']`` + ``tables['table_comet']``.
    * Xenium cell table → ``tables['table']``: RNA counts in ``X`` (+ ``layers['counts']``),
      the 16 COMET mean intensities in ``obsm['protein']``.
    """
    import anndata as ad
    import spatialdata as sd
    from spatialdata.models import Image2DModel, TableModel
    from spatialdata.transformations import Identity, get_transformation, set_transformation

    comet = sdata.images[f"{core}_comet"]
    he = sdata.images[f"{core}_he"]
    he_ref = get_transformation(he, to_coordinate_system="global")

    channels = [str(c) for c in comet.coords["c"].values]
    if_img = Image2DModel.parse(comet.data, dims=("c", "y", "x"), c_coords=channels,
                                transformations={"global": Identity()})
    dapi = Image2DModel.parse(comet.data[:1], dims=("c", "y", "x"), c_coords=["DAPI"],
                              transformations={"global": Identity()})
    he_img = Image2DModel.parse(he.data, dims=("c", "y", "x"), c_coords=["R", "G", "B"],
                                transformations={"he": Identity()})
    if keep_reference_he:
        set_transformation(he_img, he_ref, to_coordinate_system="he_reference")

    # Xenium table: split proteins (first 16 vars, "<channel>_mean") from genes
    t = sdata.tables[f"{core}_table"]
    prot_cols = [v for v in t.var_names if v.endswith("_mean")]
    gene_cols = [v for v in t.var_names if not v.endswith("_mean")]
    X = np.asarray(t[:, gene_cols].X)
    obs = t.obs.copy()
    obs["region"] = pd.Categorical(["cell_boundaries"] * len(obs))
    rna = ad.AnnData(X=X.astype(np.float32), obs=obs, var=pd.DataFrame(index=gene_cols))
    rna.layers["counts"] = rna.X.copy()
    rna.obsm["protein"] = pd.DataFrame(np.asarray(t[:, prot_cols].X), index=obs.index,
                                       columns=[c[:-5] for c in prot_cols])
    table = TableModel.parse(rna, region="cell_boundaries", region_key="region", instance_key="instance_id")

    tc = sdata.tables[f"{core}_comet_masks_table"].copy()
    tc.obs["region"] = pd.Categorical(["comet_cell_masks"] * tc.n_obs)
    tc.uns.pop("spatialdata_attrs", None)
    table_comet = TableModel.parse(tc, region="comet_cell_masks", region_key="region", instance_key="instance_id")

    out = sd.SpatialData(
        images={"if": if_img, "morphology_focus": dapi, "he": he_img},
        points={"transcripts": sdata.points[f"{core}_transcripts"]},
        shapes={"cell_boundaries": sdata.shapes[f"{core}_cell_boundaries"],
                "nucleus_boundaries": sdata.shapes[f"{core}_nucleus_boundaries"],
                "comet_cell_masks": sdata.shapes[f"{core}_comet_cell_masks"],
                "comet_nucleus_masks": sdata.shapes[f"{core}_comet_nucleus_masks"]},
        tables={"table": table, "table_comet": table_comet},
    )
    out.attrs["multispace"] = {
        "version": "0.1.0", "platform": "xenium", "um_per_unit": COMET_UM_PER_PX,
        "images": {"if": {"modality": "if"}, "morphology_focus": {"modality": "if"}, "he": {"modality": "he"}},
        "steps": [], "core": core,
        "note": "global = COMET pixel grid (no Xenium morphology image in this dataset); "
                "morphology_focus = COMET DAPI.",
    }
    return out


def he_dapi_correlation(sdata, coordinate_system="global", factor=4):
    """Registration self-check: correlation between H&E haematoxylin (colour deconvolution) warped onto the
    COMET grid and log COMET DAPI, at 1/``factor`` resolution. Nuclei should overlap: higher is better."""
    from skimage.color import rgb2hed
    from skimage.transform import AffineTransform, downscale_local_mean, warp
    from spatialdata.transformations import get_transformation

    dapi = downscale_local_mean(np.asarray(sdata.images["morphology_focus"].data[0]).astype(np.float32),
                                (factor, factor))
    he = sdata.images["he"]
    hem = rgb2hed(np.asarray(he.data).transpose(1, 2, 0))[..., 0]
    M = get_transformation(he, to_coordinate_system=coordinate_system).to_affine_matrix(("x", "y"), ("x", "y"))
    T = np.linalg.inv(M) @ np.diag([factor, factor, 1.0])
    w = warp(hem, AffineTransform(matrix=T), output_shape=dapi.shape, preserve_range=True, cval=np.nan)
    ok = np.isfinite(w) & (dapi > 0)
    return float(np.corrcoef(np.log1p(dapi[ok]), w[ok])[0, 1])


def registration_error(sdata, moving: str = "he", reference_cs: str = "he_reference",
                       target_cs: str = "global", n: int = 25) -> pd.DataFrame:
    """Displacement (COMET px and µm) between two transforms of the same image on an n×n grid."""
    from spatialdata.transformations import get_transformation

    el = sdata.images[moving]
    h, w = el.shape[-2:]
    A = get_transformation(el, to_coordinate_system=target_cs).to_affine_matrix(("x", "y"), ("x", "y"))
    B = get_transformation(el, to_coordinate_system=reference_cs).to_affine_matrix(("x", "y"), ("x", "y"))
    gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, n))
    P = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    d = np.linalg.norm((A @ P - B @ P)[:2], axis=0)
    return pd.DataFrame({"median_px": [np.median(d)], "max_px": [d.max()],
                         "median_um": [np.median(d) * COMET_UM_PER_PX], "max_um": [d.max() * COMET_UM_PER_PX]})


# ── 2 · pixel level (spatiomic) ──────────────────────────────────────────────
def comet_array(sdata, downscale: int = 2, drop: tuple[str, ...] = ("DAPI",), crop: dict | None = None):
    """COMET image as a channel-last float32 array (y, x, c), optionally cropped and downscaled."""
    from skimage.transform import downscale_local_mean

    img = sdata.images["if"]
    chans = [str(c) for c in img.coords["c"].values]
    keep = [i for i, c in enumerate(chans) if c not in drop]
    arr = img.data[keep]
    if crop:
        arr = arr[:, crop["y0"]:crop["y1"], crop["x0"]:crop["x1"]]
    arr = np.asarray(arr).astype(np.float32).transpose(1, 2, 0)
    if downscale > 1:
        arr = downscale_local_mean(arr, (downscale, downscale, 1)).astype(np.float32)
    return arr, [chans[i] for i in keep]


def tissue_mask(arr, sigma: float = 4):
    """Tissue mask on the smoothed sum of log intensities (triangle threshold).

    Background pixels are excluded from the SOM fit and labelled -1. Otsu is too strict here: it
    drops the dim stroma between tumour nests (55% of the image kept vs 68% with the triangle threshold).
    """
    from skimage.filters import gaussian, threshold_triangle
    from skimage.morphology import remove_small_objects

    s = gaussian(np.log1p(arr).sum(-1), sigma)
    return remove_small_objects(s > threshold_triangle(s), 500)


# Biological names of the 11 spatiomic pixel clusters fitted on core01 (25x25 SOM + Leiden, seed 0).
# Read from the mean intensity profiles (spatiomic_cluster_profiles.csv); tentative, to be checked on
# the images. Core02 pixels are labelled with the same model, so the names apply to both cores.
PIXEL_CLUSTER_NAMES = {
    0: "Dim tissue",
    1: "Lymphoid (CD45, CD45RA)",
    2: "Signal-poor",
    3: "Stroma (vimentin)",
    4: "Vessel wall (αSMA)",
    5: "All-bright (autofluorescence)",
    6: "Tumour–immune interface",
    7: "Epithelium, CK8 dim",
    8: "Epithelium, CK8+",
    9: "Epithelium, CK8+ VEGF+ IL-10+",
    10: "Epithelium, CK8+ E-cad high",
}


# Names of the 11 clusters of the joint spatiomic model (core01 + core02, each core normalised with its own
# percentiles; prepare_core02.py -> core02/joint/). Tentative, from cluster_profiles.csv. A different model
# from PIXEL_CLUSTER_NAMES: the numbers do not correspond.
JOINT_CLUSTER_NAMES = {
    0: "Immune / stroma (CD45, CD45RA)",
    1: "Vessel wall (αSMA)",
    2: "Signal-poor",
    3: "IL-10 dim",
    4: "CD206 dim",
    5: "Diffuse VEGF, IL-10, HepPar1",
    6: "All-bright (autofluorescence)",
    7: "Epithelium, E-cad+ CK8 dim",
    8: "Epithelium, CK8 low",
    9: "Epithelium, CK8+ IL-10+ VEGF+",
    10: "Epithelium, CK8 high E-cad+",
}


def pixel_cluster_label(k) -> str:
    k = int(k)
    return f"{k} · {PIXEL_CLUSTER_NAMES[k]}" if k in PIXEL_CLUSTER_NAMES else str(k)


def run_spatiomic(arr, mask=None, n_pixels=60_000, som_size=(25, 25), iterations=20, resolution=1.0,
                  neighbor_count=12, seed=0):
    """spatiomic pixel clustering. Background (outside ``mask``) gets label -1."""
    import spatiomic as so

    pix = arr[mask] if mask is not None else arr.reshape(-1, arr.shape[-1])
    sub = so.data.subsample().fit_transform(pix, method="count", count=min(n_pixels, len(pix)), seed=seed)
    clipper = so.process.clip(method="percentile", percentile_min=50.0, percentile_max=99.9, use_gpu=False)
    norm = so.process.normalize(min_value=0.0, max_value=1.0, use_gpu=False)
    sub_n = norm.fit_transform(clipper.fit_transform(sub))
    img_n = norm.transform(clipper.transform(arr))
    som = so.dimension.som(node_count=som_size, dimension_count=arr.shape[-1], distance_metric="euclidean",
                           seed=seed, use_gpu=False)
    som.fit(sub_n, iteration_count=iterations)
    qe = float(som.get_quantization_error(sub_n))
    graph = so.neighbor.knn_graph().create(som, neighbor_count=neighbor_count, use_gpu=False)
    communities, modularity = so.cluster.leiden().predict(graph, resolution=resolution, iteration_count=100,
                                                          seed=42, use_gpu=False)
    # label in row blocks: labelling the whole image at once needs ~10 GB (pixels × SOM nodes distances)
    labelled = np.vstack([som.label(img_n[r:r + 128], communities, flatten=False)
                          for r in range(0, img_n.shape[0], 128)]).astype(np.int32)
    if mask is not None:
        labelled[~mask] = -1
    return dict(img_n=img_n, labelled=labelled, som=som, communities=np.asarray(communities),
                quantization_error=qe, modularity=float(modularity), clipper=clipper, normalizer=norm)


def run_spatiomic_joint(arrs, masks, n_pixels_per_core=30_000, som_size=(25, 25), iterations=20, resolution=1.0,
                        neighbor_count=12, seed=0, per_core_norm=False):
    """One spatiomic model fitted on pixels pooled from several cores, then used to label each core.

    The same number of tissue pixels is drawn from every core, so no core dominates the SOM.

    ``per_core_norm=False``: one clipping / normalisation for all cores. A core that is brighter in every
    channel (autofluorescence, staining or exposure) then gets clusters of its own.
    ``per_core_norm=True``: each core is clipped and scaled to [0, 1] with its own percentiles before
    pooling. This removes core-level intensity offsets, and with them any true global difference.
    """
    import spatiomic as so

    def _pre():
        return (so.process.clip(method="percentile", percentile_min=50.0, percentile_max=99.9, use_gpu=False),
                so.process.normalize(min_value=0.0, max_value=1.0, use_gpu=False))

    subs = [so.data.subsample().fit_transform(a[m], method="count", count=min(n_pixels_per_core, int(m.sum())),
                                              seed=seed + i) for i, (a, m) in enumerate(zip(arrs, masks))]
    if per_core_norm:
        pres = [_pre() for _ in arrs]
        subs_n = [n.fit_transform(c.fit_transform(x)) for (c, n), x in zip(pres, subs)]
    else:
        c, n = _pre()
        n.fit(c.fit_transform(np.vstack(subs)))
        pres = [(c, n)] * len(arrs)
        subs_n = [n.transform(c.transform(x)) for x in subs]
    pooled_n = np.vstack(subs_n)
    som = so.dimension.som(node_count=som_size, dimension_count=pooled_n.shape[-1], distance_metric="euclidean",
                           seed=seed, use_gpu=False)
    som.fit(pooled_n, iteration_count=iterations)
    graph = so.neighbor.knn_graph().create(som, neighbor_count=neighbor_count, use_gpu=False)
    communities, modularity = so.cluster.leiden().predict(graph, resolution=resolution, iteration_count=100,
                                                          seed=42, use_gpu=False)
    ref = dict(som=som, communities=np.asarray(communities), quantization_error=float(som.get_quantization_error(pooled_n)),
               modularity=float(modularity), per_core_norm=per_core_norm)
    per_core = [project_spatiomic(a, m, dict(ref, clipper=c, normalizer=n)) for a, m, (c, n) in zip(arrs, masks, pres)]
    return ref, per_core


def project_spatiomic(arr, mask, ref):
    """Label another core's pixels with a fitted spatiomic model (``ref`` = ``run_spatiomic`` output).

    The clipping / normalisation fitted on the reference core, its SOM and its Leiden communities
    are reused unchanged, so cluster k means the same pixel phenotype in both cores. This is valid
    here because both cores sit on the same TMA slide (same staining and imaging run).
    """
    img_n = ref["normalizer"].transform(ref["clipper"].transform(arr))
    labelled = np.vstack([ref["som"].label(img_n[r:r + 128], ref["communities"], flatten=False)
                          for r in range(0, img_n.shape[0], 128)]).astype(np.int32)
    if mask is not None:
        labelled[~mask] = -1
    pix = img_n[mask] if mask is not None else img_n.reshape(-1, img_n.shape[-1])
    qe = float(ref["som"].get_quantization_error(pix[:: max(1, len(pix) // 60_000)]))
    return dict(img_n=img_n, labelled=labelled, quantization_error=qe)


def cluster_profiles(img_n, labelled, channels):
    ok = labelled >= 0
    df = pd.DataFrame(img_n[ok], columns=channels)
    df["cluster"] = labelled[ok]
    prof = df.groupby("cluster").mean()
    prof["n_pixels"] = df.groupby("cluster").size()
    return prof


def vicinity(labelled, order: int = 2, tile: int = 512):
    """spatiomic vicinity composition (background excluded) and its graph.

    Computed on tiles of ``tile`` px and summed (counts are additive; pairs that cross a tile
    border are ignored, <1% of pairs at tile=512). On the whole 2k × 2k image at once,
    spatiomic needs >7 GB.
    """
    import spatiomic as so

    lab = labelled.astype(np.int64)
    k = int(lab.max()) + 1
    lab[lab < 0] = k  # background as its own label, dropped below
    off = so.spatial.neighborhood_offset(neighborhood_type="queen", order=order)
    total = pd.DataFrame(0.0, index=range(k + 1), columns=range(k + 1))
    for r in range(0, lab.shape[0], tile):
        for c in range(0, lab.shape[1], tile):
            block = lab[r:r + tile, c:c + tile]
            if (block < k).sum() == 0:
                continue
            vc = so.spatial.vicinity_composition(block, neighborhood_offset=off, ignore_identities=True,
                                                 ignore_repeated=True)
            total = total.add(vc.reindex(index=range(k + 1), columns=range(k + 1), fill_value=0), fill_value=0)
    vals = total.iloc[:k, :k].to_numpy().copy()
    np.fill_diagonal(vals, 0)
    vc = pd.DataFrame(vals, index=range(k), columns=range(k))
    # spatiomic's own diagonal reset fails under pandas 3 (read-only view), so it is done above
    graph = so.spatial.vicinity_graph(vc, ignore_identities=False)
    return vc, graph, None


# ── 3 · pixels → cells ─────────────────────────────────────────────────────────
def rasterize_cells(polygons, cell_ids, shape, origin=(0.0, 0.0), scale=1.0):
    """Burn polygons (COMET px) into an integer label image of ``shape``; value = index+1."""
    from skimage.draw import polygon as draw_polygon

    mask = np.zeros(shape, dtype=np.int32)
    x0, y0 = origin
    for i, poly in enumerate(polygons, start=1):
        if poly is None or poly.is_empty:
            continue
        for g in getattr(poly, "geoms", [poly]):
            xs, ys = np.asarray(g.exterior.coords).T
            rr, cc = draw_polygon((ys - y0) / scale, (xs - x0) / scale, shape=shape)
            mask[rr, cc] = i
    return mask


def pixel_composition(cell_mask, labelled, cell_ids, k=None):
    """Fraction of each cell's tissue pixels in each spatiomic cluster (cells × k)."""
    k = int(labelled.max()) + 1 if k is None else k
    m, l = cell_mask.ravel(), labelled.ravel()
    keep = (m > 0) & (l >= 0)
    df = pd.DataFrame({"cell": m[keep], "cluster": l[keep]})
    comp = pd.crosstab(df["cell"], df["cluster"], normalize="index").reindex(columns=range(k), fill_value=0.0)
    comp.index = [cell_ids[i - 1] for i in comp.index]
    comp.columns = [f"pix_{c}" for c in comp.columns]
    return comp


def block_scale(block):
    """Centre each column, then divide the block by sqrt(total variance): every modality weighs the same."""
    b = np.asarray(block, dtype=float)
    b = b - b.mean(axis=0)
    tot = np.sqrt((b ** 2).sum() / max(len(b) - 1, 1))
    return b / tot if tot > 0 else b


def spatial_coherence(xy, labels, k: int = 10) -> float:
    """Fraction of each cell's k spatial nearest neighbours that share its label (mean over cells)."""
    from sklearn.neighbors import NearestNeighbors

    labels = np.asarray(labels)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    idx = nn.kneighbors(xy, return_distance=False)[:, 1:]
    return float((labels[idx] == labels[:, None]).mean())


def rna_protein_concordance(rna_counts: pd.DataFrame, protein: pd.DataFrame, pairs=RNA_PROTEIN_PAIRS):
    """Spearman correlation per (protein, gene) pair across cells, for pairs present in both."""
    from scipy.stats import spearmanr

    rows = []
    for p, g in pairs.items():
        if p in protein and g in rna_counts:
            r = spearmanr(rna_counts[g], protein[p]).statistic
            rows.append({"protein": p.split(" - ")[0], "gene": g, "spearman": r,
                         "gene_detected_frac": float((rna_counts[g] > 0).mean())})
    return pd.DataFrame(rows).sort_values("spearman", ascending=False).reset_index(drop=True)


def fix_unassigned(adata, key: str = "cell_type", conf_key: str = "cell_type_confidence", label: str = "Unassigned"):
    """multispace 0.1.0 labels zero-confidence cells with the first type alphabetically ('B cell').

    Relabel them explicitly until this is fixed in the package.
    """
    zero = adata.obs[conf_key].to_numpy() == 0
    ct = adata.obs[key].astype(str).to_numpy()
    ct[zero] = label
    adata.obs[key] = pd.Categorical(ct)
    return int(zero.sum())


# ── 4 · multilayer network on shared cells ─────────────────────────────────────
def knn_layer(X, k: int = 15, seed: int = 0):
    """Symmetric kNN graph (scipy sparse, binary) of the rows of X (cells)."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    import scipy.sparse as sp

    n = X.shape[0]
    rows = np.repeat(np.arange(n), k)
    A = sp.csr_matrix((np.ones(n * k), (rows, idx[:, 1:].ravel())), shape=(n, n))
    return ((A + A.T) > 0).astype(np.float32).tocsr()


def multiplex_leiden(layers: dict, resolution: float = 1.0, seed: int = 0):
    """Communities shared by all layers (multiplex modularity, Mucha et al. 2010; leidenalg).
    Each layer is a graph on the same cells; returns one label per cell."""
    import igraph as ig
    import leidenalg as la

    graphs = []
    for A in layers.values():
        r, c = A.nonzero()
        keep = r < c
        graphs.append(ig.Graph(n=A.shape[0], edges=list(zip(r[keep].tolist(), c[keep].tolist()))))
    parts = [la.RBConfigurationVertexPartition(g, resolution_parameter=resolution) for g in graphs]
    opt = la.Optimiser()
    opt.set_rng_seed(seed)
    opt.optimise_partition_multiplex(parts, n_iterations=-1)
    return np.asarray(parts[0].membership)


def layer_agreement(layers: dict):
    """Jaccard overlap of the edge sets of each pair of layers: how much the modalities agree on who is similar."""
    names = list(layers)
    out = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = layers[a], layers[b]
            inter = A.multiply(B).nnz
            out.loc[a, b] = out.loc[b, a] = inter / (A.nnz + B.nnz - inter)
    return out


def multiplex_rwr(layers: dict, seeds, restart: float = 0.7, delta: float = 0.5, tol: float = 1e-9, max_iter: int = 200):
    """Random walk with restart on a multiplex (same nodes in every layer), as in MultiXrank
    (Baptista, Gonzalez & Baudot, Commun Phys 2022). The walker follows edges within a layer, jumps to the
    same cell in another layer with probability ``delta``, and restarts at the seed cells with probability
    ``restart``. Returns one score per cell (geometric mean over layers, MultiXrank's default output)."""
    import scipy.sparse as sp

    L = len(layers)
    n = next(iter(layers.values())).shape[0]
    blocks = []
    for A in layers.values():
        d = np.asarray(A.sum(axis=0)).ravel()
        d[d == 0] = 1
        blocks.append(A @ sp.diags(1.0 / d))                       # column-stochastic per layer
    intra = sp.block_diag(blocks, format="csr")
    if L > 1:
        inter = sp.kron((np.ones((L, L)) - np.eye(L)) / (L - 1), sp.eye(n), format="csr")
        M = (1 - delta) * intra + delta * inter
    else:
        M = intra
    p0 = np.zeros(L * n)
    seeds = np.asarray(seeds)
    for l in range(L):
        p0[l * n + seeds] = 1.0
    p0 /= p0.sum()
    p = p0.copy()
    for _ in range(max_iter):
        new = (1 - restart) * (M @ p) + restart * p0
        if np.abs(new - p).sum() < tol:
            p = new
            break
        p = new
    P = p.reshape(L, n)
    return np.exp(np.log(P + 1e-300).mean(axis=0))


# One colour per cell type, shared by every cell-type figure (RNA-only and RNA + protein, both cores).
CELL_TYPE_COLORS = {"HCC tumour": "#E15759", "Hepatocyte": "#F1A9A0", "Cholangiocyte": "#EDC948",
                    "T cell": "#4E79A7", "B cell": "#1A1A1A", "Macrophage": "#B07AA1", "Dendritic cell": "#76B7B2",
                    "NK cell": "#A0CBE8", "Plasma cell": "#9C755F", "Neutrophil": "#FF9DA7",
                    "Fibroblast / stellate": "#59A14F", "Endothelial": "#F28E2B",
                    "Unknown": "#CCCCCC", "Ambiguous": "#CCCCCC", "Unresolved": "#CCCCCC"}


# ── neighbourhood enrichment (notebook section 5c) ──────────────────────────────────────────────────────
def nhood_zscore(xy, labels, seed: int = 0) -> pd.DataFrame:
    """Neighbourhood enrichment z-scores, the same settings as multispace's ``run_spatial_analysis``
    ``"neighborhood"`` section: Delaunay graph on the cell centroids, squidpy ``nhood_enrichment``
    (1,000 label permutations). Use it on cells without a SpatialData object (e.g. core02 from a table)."""
    import anndata as ad
    import squidpy as sq

    a = ad.AnnData(obs=pd.DataFrame({"label": pd.Categorical(np.asarray(labels).astype(str))}))
    a.obsm["spatial"] = np.asarray(xy, dtype=float)
    sq.gr.spatial_neighbors(a, coord_type="generic", delaunay=True)
    sq.gr.nhood_enrichment(a, cluster_key="label", seed=seed, show_progress_bar=False)
    cats = a.obs["label"].cat.categories
    return pd.DataFrame(a.uns["label_nhood_enrichment"]["zscore"], index=cats, columns=cats)


def plot_nhood(z: pd.DataFrame, ax, title: str, vmax: float = 50, order=None):
    """Heatmap of neighbourhood z-scores (red: neighbours more often than chance, blue: avoid each other)."""
    import matplotlib.pyplot as plt

    if order is not None:
        keep = [c for c in order if c in z.index]
        z = z.loc[keep, keep]
    im = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(z)), z.columns, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(len(z)), z.index, fontsize=9)
    for i in range(len(z)):
        for j in range(len(z)):
            ax.text(j, i, f"{z.iat[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(z.iat[i, j]) > 0.6 * vmax else "black")
    ax.set_title(title, fontsize=11)
    return im
