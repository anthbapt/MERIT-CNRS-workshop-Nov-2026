# Integrating heterogeneous data in spatial biology

MERIT (CNRS) workshop, November 2026 · Dr Anthony Baptista, King's College London

One hepatocellular carcinoma (HCC) core, one FFPE section, three instruments: **H&E**, **Xenium Prime 5K**
(5,001 genes, 8,290 cells) and **COMET** (DAPI + 15 proteins). In 30 minutes you load the three layers into one
SpatialData object, check the registration and the segmentation, cluster the protein pixels without segmentation,
bring the pixels back to the cells, compare joint and single-modality views, type cells from RNA and protein, and ask
which cell types sit next to which. A non-tumour core from the same tissue microarray is used for a tumour vs.
non-tumour comparison.

Two packages:

- **multispace** (cell level): one SpatialData object with a fixed data contract, image registration, cell typing,
  spatial statistics. In-house package, not yet published; it grew out of
  [MOSAIK](https://github.com/anthbapt/MOSAIK). Installed from the wheel in `wheels/`.
- **spatiomic** (pixel level): self-organising map + Leiden clustering of image pixels
  ([spatiomic.org](https://spatiomic.org)).

## What is in this repository

| File | What it is |
|---|---|
| `handson_core01_hcc.ipynb` | **The hands-on notebook**: sections 1–5, then 4b, 5b, 5c and 6 if time. Runs in about 2 min from the precomputed data |
| `demo_liver_registration.ipynb` | The live demo: registering a post-Xenium H&E onto the Xenium DAPI in code, and how it fails |
| `workshop_hcc.py` | Helpers used by the notebooks (pixel clustering, pixels → cells, metrics, multilayer network, RNA + protein cell typing, neighbourhood enrichment) |
| `handson_core01_hcc_from_raw.ipynb`, `demo_liver_registration_from_raw.ipynb` | The same two notebooks, starting from the raw data (see the last section) |
| `wheels/` | multispace 0.1.0 |
| `environment.yml`, `requirements.txt` | Pinned versions the notebooks were tested with (Python 3.12) |

## Data

All the data, precomputed and raw, are on Zenodo: [10.5281/zenodo.23867035](https://doi.org/10.5281/zenodo.23867035)

| File | Size | What it is | Needed for |
|---|---|---|---|
| `prepared.zip` | 472.5 MB | `core01_ms.zarr` (the HCC core in the multispace data contract) and the precomputed results (registration, cell typing, spatiomic, core02) | **The hands-on** |
| `figures.zip` | 23.5 MB | Figures from the slides; the hands-on displays two of them | **The hands-on** |
| `liver_prepared.zip` | 79.0 MB | Precomputed registration and 1 mm crop of the 10x liver section | The demo notebook |
| `core01_hcc.zarr.zip` | 443.2 MB | Raw HCC core: H&E, Xenium Prime 5K, COMET | Notebooks from raw |
| `core02_non_tumour.zarr.zip` | 404.1 MB | Raw non-tumour core from the same tissue microarray | Notebooks from raw |
| `10x_Liver_cancer.zip` | 10.4 GB | 10x Xenium liver cancer section with its post-Xenium H&E | Xenium Explorer, liver notebook from raw |

The HCC cores come from the UnumLocalia tissue microarray (Duchini et al., bioRxiv 2026; original deposit
[zenodo.org/records/21713660](https://zenodo.org/records/21713660)). The liver section is the 10x Genomics *Human
Liver Data, Xenium Human Multi-Tissue and Cancer Panel* (CC BY 4.0). MD5 checksums are on the Zenodo page.

## Setup (before the session, ~10 min)

You need [conda](https://conda-forge.org/download/) (or Python 3.12 with pip) and about 3 GB of free disk space.
A laptop with 8 GB of RAM is enough.

```bash
git clone https://github.com/anthbapt/MERIT-CNRS-workshop-Nov-2026
cd MERIT-CNRS-workshop-Nov-2026
conda env create -f environment.yml
conda activate merit
```

Download `prepared.zip` and `figures.zip` from Zenodo (and `liver_prepared.zip` for the demo notebook) into the
repository folder, unzip them there, and start Jupyter:

```bash
unzip prepared.zip          # -> prepared/
unzip figures.zip           # -> figures/
unzip liver_prepared.zip    # -> liver_prepared/ (optional)
jupyter lab handson_core01_hcc.ipynb
```

After setup the folder looks like this:

```
MERIT-CNRS-workshop-Nov-2026/
├── handson_core01_hcc.ipynb
├── demo_liver_registration.ipynb
├── workshop_hcc.py
├── prepared/           from prepared.zip
├── figures/            from figures.zip
└── liver_prepared/     from liver_prepared.zip (demo)
```

## Xenium Explorer (live demo)

Unzip `10x_Liver_cancer.zip` and open `10x_Liver_cancer/Xenium_V1_hLiver_cancer_section_FFPE_outs/experiment.xenium`
in [Xenium Explorer](https://www.10xgenomics.com/support/software/xenium-explorer) (v1.3 or later). The H&E
(`…_he_image.ome.tif`) and the 10x alignment file (`…_he_imagealignment.csv`) are in `10x_Liver_cancer/`.

## Notebooks that start from the raw data

`handson_core01_hcc.ipynb` and `demo_liver_registration.ipynb` load precomputed results so that they run in seconds
during the session. The two `_from_raw` versions **rebuild every result from the raw data** in a part 0, with the
same functions, then continue as the session notebooks. Each step is skipped when its output already exists, so a
second run is fast.

| Notebook | Rebuilds | First run | Writes |
|---|---|---|---|
| `handson_core01_hcc_from_raw.ipynb` | Both cores: data contract, H&E registration, cell typing, spatiomic on the whole cores, the joint spatiomic model, RNA + protein cell typing, figures | ~20 min, ~6 GB RAM | `results_from_raw/` (~1.1 GB) |
| `demo_liver_registration_from_raw.ipynb` | The whole-section registration (three scored attempts), the 1 mm crop, figures L1–L4 | ~10 min, ~6 GB RAM | `results_from_raw/liver/` (~80 MB) |

They need:

- the raw data from Zenodo in `data/`: `data/core01_hcc.zarr/` and `data/core02_non_tumour.zarr/` (from the two
  `.zarr.zip` files), and `data/10x_Liver_cancer/` (from `10x_Liver_cancer.zip`, liver notebook only);
- the build scripts they call: `prepare_core01.py`, `prepare_core02.py`, `make_figures.py`,
  `make_celltype_figures.py` (HCC), and `prepare_liver.py`, `liver_lib.py`, `make_liver_figures.py` (liver).

One step is not rebuilt: resegmentation (section 2 of the hands-on) needs the Xenium morphology image, which this
dataset does not include, and a GPU. The notebook shows the two images of the validation run from
`prepared/reseg_metabric/` (in `prepared.zip`).

## Questions

anthony.baptista@kcl.ac.uk
