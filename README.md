# GPCRmd Pipeline: Lipid Maps Precomputation

This repository contains the tools and workflows required to generate lipid density maps (*Lipid Maps*) as part of the **GPCRmd** precomputation pipeline.

---

## System and Environment Setup

These scripts are configured to run on the GPCRmd cluster. Before executing the pipeline, load the required environment modules.

### Prerequisites

* **VMD (Visual Molecular Dynamics):** Required for handling molecular structures and trajectory data.
* **Miniconda3:** Provides the Python environment and package dependencies.

### Create the Virtual Environment

```bash
conda env create -f gpcrmd_env.yml
```

### Configure Paths and Database Information

This scripts was design taking into account the GPCRmd cluster. Replace paths or variables for yours:

```python
# Set paths and files
files_path = ""
mediaroot = ""

# Load database information from the compl_info.json file
db_dict = json_dict(files_path + "Precomputed/compl_info.json")
dynids = {"dyn" + a for a in args.dynids} if args.dynids else db_dict.keys()
```

---

## Running the Pipeline

```bash
python lipid_insertion.py --whole_residue --threads 3 > lipids.log 2>&1 &
```

### Parameters

* `--whole_residue`: Computes lipid density maps using the entire lipid residue rather than individual atoms.
* `--threads 3`: Allocates 3 CPU threads for parallel execution.

### Output Redirection

```bash
> lipids.log 2>&1 &
```

* Redirects standard output (`stdout`) and standard error (`stderr`) to `lipids.log`.
* Runs the process in the background.

```
```
