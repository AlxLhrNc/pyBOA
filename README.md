# boa_front_detection

Detection algorithm for oceanographic data (specifically chlorophyll but can be used for others)

**Original pseudo-code:**
_Belkin, I.M., O’Reilly, J.E., 2009. An algorithm for oceanic front detection in chlorophyll and SST satellite imagery.
Journal of Marine Systems, Special Issue on Observational Studies of Oceanic Fronts 78, 319–326_ (https://doi.org/10.1016/j.jmarsys.2008.11.018).


**Transcription of the work from:**
Lin et al. (2019) - Matlab, _Lin, L., Liu, D., Luo, C., Xie, L., 2019. Double fronts in the Yellow Sea in summertime identified using sea surface
temperature data of multi-scale ultra-high resolution analysis. Continental Shelf Research 175, 76–86._ (https://doi.org/10.1016/j.csr.2019.02.004).
Ben Galuardi, _boaR - R package_ (https://rdrr.io/github/galuardi/boaR/man/boaR-package.html)


**Additions:**
Generalized contextual filter, rolling percentile selection, morphological thinning for single lines.

**What to get:**
The sample netcdf file, the stnd_alone file, and pyBOA.py.
