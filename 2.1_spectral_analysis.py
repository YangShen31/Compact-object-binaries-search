# %%
import sys
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.stats import norm
from matplotlib.pyplot import figure

df = pd.read_csv('Data/target_LAMOST_RV.csv')
# %%
def Gaussian_fit(file_name):
	filepath = '/Users/mac/Desktop/cassi_folder/spectra_file/' + file_name
	hdu_list = fits.open(filepath, memmap=True)
	data = Table(hdu_list[2].data) #red spectrum
	SNR = hdu_list[1].header['SNR']
	if SNR < 10:
		raise ValueError("Signal To Noise must be bigger than 10")
	obsid = hdu_list[0].header['OBSID']
	star_name = hdu_list[0].header['DESIG']
	LAMOST_vr = df.loc[df['designation']== star_name]["rv_br1"].values[0]

	# %%

	# %%
	# Extract wavelength, flux, and uncertainties
	wv = data["WAVELENGTH"][0]
	fx = data["FLUX"][0]
	errr = np.sqrt(1/data["IVAR"][0])

	# %% [markdown]
	# ## Step 1: Determine the radial velocity of the star

	# %%
	wv_0_Ha = 6562.8
	c = 299792.458 #unit: km/s

	#find the possible wavelength of the H-alpha to restrict the wavelength range
	LAMOST_Ha = LAMOST_vr * wv_0_Ha / c + wv_0_Ha
	Min = LAMOST_Ha - 5
	Max = LAMOST_Ha + 5
	within_range = (wv >= Min) & (wv <= Max)
	wv_filtered = wv[within_range]
	fx_filtered = fx[within_range]
	min_fx = np.min(fx_filtered)
	min_indices = np.where((fx_filtered == min_fx) & (fx_filtered != 0))[0]
	wv_min = wv_filtered[min_indices]

	#calculate the radial velocity
	def radial_velocity(wv_0, wv): 
		a = c * (wv - wv_0)/wv_0
		return a

	v_r = radial_velocity(wv_0_Ha, wv_min)
	# %%

	# %% [markdown]
	# ## Step 2: Find the wavelength of lithium in the emission

	# %%
	wv_0_Li = 6707.835
	wv_Li = wv_0_Li *(1 + v_r/c)
	wv_Li

	# %% [markdown]
	# ## Step 3: Find the line of best fit near lithium line

	# %%
	@custom_model
	def sum_of_gaussians(x, amplitude=1., mean=-1., sigma=1.):
		try:
			return (amplitude * np.exp(-0.5 * ((x - mean) / sigma)**2) + 1)
		except Exception as e:
			print(f"Gaussian fit failed: {e}")
			return 1e6

	#filter out the region for linear best fit line
	wv_Li_min = wv_Li - 20
	wv_Li_max = wv_Li + 20
	wv_range = (wv >= wv_Li_min) & (wv <= wv_Li_max)
	wv_filtered = wv[wv_range]
	fx_filtered = fx[wv_range]
	errr_filtered = errr[wv_range]

	#fit with account to the uncertainty
	def linear_model(x, m, c):
		return m * x + c
	popt, pcov = curve_fit(linear_model, wv_filtered, fx_filtered, sigma=errr_filtered, absolute_sigma=True)
	m, c = popt
	line_of_best_fit = m * wv_filtered + c

	#omit the outlier
	residuals = fx_filtered - line_of_best_fit
	residuals_std = np.std(residuals)
	threshold = 3 * residuals_std

	# Filter out points with large residuals
	mask = np.abs(residuals) < threshold
	wv_removed = wv_filtered[mask]
	fx_removed = fx_filtered[mask]
	errr_filtered = errr_filtered[mask]

	popt, pcov = curve_fit(linear_model, wv_removed - np.mean(wv), fx_removed, sigma=errr_filtered, absolute_sigma=True)
	m, c = popt
	line_of_best_fit = m * (wv - np.mean(wv)) + c

	# %% [markdown]
	# ## Step 4: Reduce the spectra with respect to the line of best fit

	# %%
	fx_normalized_filtered = (fx/line_of_best_fit)[wv_range]
	errr_normalized_filtered = (errr/line_of_best_fit)[wv_range]

	# %% [markdown]
	# ## Step 5: Fit the Gaussian model to this region

	# %%
	sigma_Li = wv_Li/7500/2.3548
	m_init = sum_of_gaussians(amplitude=-0.05,mean=wv_Li,sigma=sigma_Li)

	#constraints
	m_init.amplitude.fixed = False
	m_init.mean.fixed = True
	m_init.sigma.fixed = True
	m_init.amplitude.min = -1
	m_init.amplitude.max = 0

	# 4. Fit and catch fitting errors
	fit = LevMarLSQFitter()
	fx_final = fx_normalized_filtered / np.median(fx_normalized_filtered)
	errr_final = errr_normalized_filtered / np.median(fx_normalized_filtered)

	m = fit(m_init, wv_filtered, fx_final, maxiter=500, weights=1/errr_final)
	cov = fit.fit_info['param_cov']
	error = cov**0.5

	# Equivalent width
	W = m.amplitude.value * m.sigma.value * np.sqrt(2) * np.pi
	W_error = -error[0][0] / m.amplitude.value * W if m.amplitude.value != 0 else np.nan

	# chi²
	y_pred = m(wv_filtered)
	chi2 = np.sum(((fx_final - y_pred) / errr_final)**2)
	reduced_chi2 = chi2 / (len(wv_filtered) - 1)

	# conditions = (W < -0.082) & (W/W_error > 3)

	# if conditions:
	# 	fig1 = plt.figure()
	# 	plt.rcParams.update({'font.family':'times'})
	# 	plt.plot(wv, fx)
	# 	plt.xlabel('wavelength ($\AA$)', size=13)
	# 	plt.ylabel('flux', size=13)
	# 	plt.title(f"Radial Velocity: {v_r}")

	# 	plt.savefig("Data/plots/" + f"{star_name}_whole_plots.png",
	# 				transparent=False, dpi=900, bbox_inches='tight')

	# 	plt.close(fig1)   # <-- CLOSE FIRST FIGURE


	# 	# ---- Second plot ----
	# 	fig2 = plt.figure(figsize=(10, 6))
	# 	plt.plot(wv_filtered, fx_final, lw=4)
	# 	plt.plot(wv, m(wv), color='r', lw=4)
	# 	plt.xticks(np.arange(6697, 6713, step=3), fontsize=30)
	# 	plt.yticks(np.arange(0, 1.20, step=0.1), fontsize=30)
	# 	plt.xlim(wv_Li_min + 15, wv_Li_max - 15)
	# 	plt.ylim(np.min(fx_normalized_filtered) - 0.1,
	# 			np.max(fx_normalized_filtered) + 0.1)
	# 	plt.xlabel('wavelength ($\AA$)', size=23)
	# 	plt.ylabel('normalized flux', size=23)
	# 	plt.vlines(x=wv_Li, ymax=1.10, ymin=0.7, color="purple")

	# 	plt.savefig("Data/plots/" + f"{star_name}_Li_fitting.png")

	# 	plt.close(fig2)

	return [star_name, file_name, v_r[0], W, W_error, reduced_chi2]

# %% [markdown]
# ## Select the database to analyze

# %%
# Choice: LAMOST MRS/LRS DR9, GALAH, SDSS optical


# %%
results = []       
# number of times the fit went wrong
Fit = 0

# number of times the signal to noise is less than 10
SN = 0

directory = '/Users/mac/Desktop/cassi_folder/spectra_file/'

for filename in sorted(os.listdir(directory)):
    try:
        result = Gaussian_fit(filename)
        results.append(result)
    except Exception as e:
        if str(e) == "unsupported operand type(s) for ** or pow(): 'NoneType' and 'float'":
            Fit += 1
        if str(e) == "Signal To Noise must be bigger than 10":
            SN += 1
        print(f"Error processing file {filename}: {e}")


# %%
df = pd.DataFrame(results, columns=['designation', 'obsid' ,'radial velocity', 'Equivalent width', 'Equivalent width uncertainty', 'reduced chi2'])
df.to_csv('Data/Results/targets.csv')


