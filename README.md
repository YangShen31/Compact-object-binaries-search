# CASSI
### Setup
Download LAMOST MRS DR9 Parameter Catalog from [www.lamost.org/dr9/v2.0/catalogue](https://www.lamost.org/dr9/v2.0/catalogue) and place it into created directory 'data/'. 

### Step 1: Primary sample selection
To select the primary sample (main-sequence, metal-poor stars), run `1_Sample_Selection.ipynb`.

### Step 2: Spectral analysis
To perform the spectral analysis and extract lithium absorption EW, run the script `2.1_spectral_analysis.py`. For a closer look to one specific observation, use notebook `2.2_Gaussian_fit.ipynb`. 

### Step 3: Candidates selection
To select candidates with high lithium absorption, run `3.1_Li_high_Candidates.ipynb`. Then, to further select candidates with high variability in radial velocities and relevant RUWE value, run `3.2_RV_chi2_selection.ipynb`. 

## Further exploration
### 1. Lightcurve plotting
`4.1_lc_plotting.ipynb` provides codes to plot TESS and ASAS-SN's lightcurves. 

### 2. Radial Velocities fitting
`3.2_lc_plotting.ipynb`
