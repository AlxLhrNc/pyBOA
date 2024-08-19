# pyBOA front detection

Detection algorithm for oceanographic data (specifically chlorophyl / temperature but can be used for others)

**Original pseudo-code:**
Belkin, I.M., O’Reilly, J.E., 2009. An algorithm for oceanic front detection in chlorophyll and SST satellite imagery.
Journal of Marine Systems, Special Issue on Observational Studies of Oceanic Fronts 78, 319–326_ [DOI](https://doi.org/10.1016/j.jmarsys.2008.11.018).


**Transcription of the work from:**
Lin et al. (2019) - Matlab, _Lin, L., Liu, D., Luo, C., Xie, L., 2019. Double fronts in the Yellow Sea in summertime identified using sea surface
temperature data of multi-scale ultra-high resolution analysis. Continental Shelf Research 175, 76–86._ [DOI](https://doi.org/10.1016/j.csr.2019.02.004).
Ben Galuardi, _boaR - R package_ [DOI](https://rdrr.io/github/galuardi/boaR/man/boaR-package.html)


**Additions:**
Generalized contextual filter, rolling percentile selection, morphological thinning for single lines.

**What to get:**
The sample netcdf file, the stnd_alone file, and pyBOA.py.

**Important**
This works as an extension of the xarray packages and was built under python 3.9
Currently under liscence GNU General Public License, see [Zenodo](https://zenodo.org/records/8135921). See proper citation and attribution in cff file and zenodo.json.
