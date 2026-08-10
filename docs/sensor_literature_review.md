# Fiber Photometry Sensor Literature Review

This appendix documents the sensor registry used by pyBer preprocessing. It is
practical rather than encyclopedic: the table favors sensors that have been used
or can plausibly be used with bulk fiber photometry, and it records uncertainty
when the literature reports assay-specific values rather than one universal
photometry number.

## Operational Rules Used By pyBer

- A 405 nm channel is often a useful control for green cpGFP-based sensors, but
  it is not automatically a true isobestic point for every reporter or optical
  setup.
- Red sensors, peptide sensors, and newer GPCR sensors often require empirical
  validation of the control wavelength.
- Kinetics measured in cultured cells, slices, two-photon imaging, and in vivo
  fiber photometry are not interchangeable. pyBer therefore uses conservative
  photometry-oriented sampling and low-pass recommendations.
- The expected fluorescence direction matters. Most sensors brighten on ligand
  binding or Ca2+ increase, while sDarken is a darkening 5-HT sensor.
- Sensor metadata is analysis metadata. pyBer stores the selected sensor and
  the trace-check result in the export sidecar and embedded HDF5 metadata.

## Calcium Sensors

GCaMP sensors remain the default activity reporters for many fiber photometry
experiments. GCaMP3 is legacy and slow. GCaMP6f, GCaMP6m, and GCaMP6s provide
the classic fast, medium, and sensitive tradeoff described by Chen et al. 2013.
jGCaMP7 improves performance across sensitivity and kinetics, while jGCaMP8
pushes faster rise kinetics and improved single-event reporting. Red calcium
sensors such as jRGECO1a are useful for multiplexing, but the green 405 control
assumption should not be copied blindly.

Key sources:

- GCaMP3: Tian et al. 2009, Nature Methods, https://www.nature.com/articles/nmeth.1398
- GCaMP6: Chen et al. 2013, Nature, https://pmc.ncbi.nlm.nih.gov/articles/PMC3777791/
- jGCaMP7: Dana et al. 2019, Nature Methods, https://www.nature.com/articles/s41592-019-0435-6
- jGCaMP8: Zhang et al. 2023, Nature, https://www.nature.com/articles/s41586-023-05828-9
- jRGECO1a: Dana et al. 2016, eLife, https://elifesciences.org/articles/12727

## Monoamine Sensors

Dopamine, serotonin, and norepinephrine sensors are usually faster than calcium
photometry readouts and can show subsecond events. dLight and GRAB-DA are the
main dopamine families. GRAB-5HT brightens on serotonin, while sDarken darkens.
GRAB-NE and nLight report norepinephrine, with newer variants improving
sensitivity and selectivity.

Key sources:

- dLight1: Patriarchi et al. 2018, Science, https://pmc.ncbi.nlm.nih.gov/articles/PMC6287765/
- Expanded dLight palette: https://pmc.ncbi.nlm.nih.gov/articles/PMC8169200/
- GRAB-DA: Sun et al. 2018, Cell, https://www.cell.com/cell/fulltext/S0092-8674(18)30845-6
- GRAB-DA2: Sun et al. 2020, Nature Methods, https://pmc.ncbi.nlm.nih.gov/articles/PMC7648260/
- GRAB-5HT1.0: Wan et al. 2021, Neuron, https://pmc.ncbi.nlm.nih.gov/articles/PMC8544647/
- Improved GRAB-5HT green and red sensors: https://pmc.ncbi.nlm.nih.gov/articles/PMC11377854/
- sDarken: Kubitschke et al. 2022, Nature Communications, https://www.nature.com/articles/s41467-022-35200-w
- GRAB-NE: Feng et al. 2019, Neuron, https://pmc.ncbi.nlm.nih.gov/articles/PMC6533151/
- GRAB-NE2: Feng et al. 2024, Neuron, https://pubmed.ncbi.nlm.nih.gov/38547869/
- nLight: Kagiampaki et al. 2023, Nature Methods, https://pmc.ncbi.nlm.nih.gov/articles/PMC7615053/

## Acetylcholine Sensors

GRAB-ACh and iAChSnFR are common acetylcholine choices. GRAB-ACh3.0 has
subsecond kinetics in reported stimulation assays and reduced downstream
coupling. iAChSnFR is a binding-protein sensor with rapid fluorescence changes.

Key sources:

- GRAB-ACh3.0: Jing et al. 2020, Nature Biotechnology, https://pmc.ncbi.nlm.nih.gov/articles/PMC7606762/
- iAChSnFR: Borden et al. 2020, https://authors.library.caltech.edu/records/vezvm-wdv47
- Red-shifted GRAB-ACh sensors: https://pmc.ncbi.nlm.nih.gov/articles/PMC11703214/

## Fast Transmitter Sensors

Glutamate sensors such as SF-iGluSnFR and iGluSnFR3 can require the highest
bandwidth in the registry. GABA sensors are typically slower and more
assay-dependent than glutamate sensors. pyBer recommends higher sampling and
low-pass settings for iGluSnFR3 than for GABA or peptide sensors.

Key sources:

- SF-iGluSnFR: Marvin et al. 2018, Nature Methods, https://pubmed.ncbi.nlm.nih.gov/30377363/
- iGluSnFR3: Aggarwal et al. 2023, Nature Methods, https://www.nature.com/articles/s41592-023-01863-6
- iGABASnFR: Marvin et al. 2019, Nature Methods, https://www.nature.com/articles/s41592-019-0471-2
- iGABASnFR2 resources: https://www.janelia.org/open-science/igabasnfr2

## Purine, Histamine, Lipid, and Peptide Sensors

These sensors are powerful but often slower and more context dependent. pyBer
uses lower default low-pass values for endocannabinoid, orexin, opioid,
oxytocin, and GRAB neuropeptide sensors. For adenosine, ATP, and histamine, the
registry uses moderate settings and warns users to inspect controls.

Key sources:

- GRAB-eCB2.0: Dong et al. 2021, Nature Biotechnology, https://pubmed.ncbi.nlm.nih.gov/34764491/
- OxLight1: Duffet et al. 2022, Nature Methods, https://pmc.ncbi.nlm.nih.gov/articles/PMC8831244/
- Opioid peptide sensors including kLight: Massengill et al. 2024, Nature Neuroscience, https://www.nature.com/articles/s41593-024-01697-1
- NOPLight: https://pmc.ncbi.nlm.nih.gov/articles/PMC11199706/
- GRABAdo and in vivo adenosine work: https://www.pnas.org/doi/10.1073/pnas.2212387120
- iAdo: https://www.nature.com/articles/s41467-025-59530-7
- GRAB-ATP1.0: https://www.yulonglilab.org/pdfs/A%20sensitive%20GRAB%20sensor%20for%20detecting%20extracellular%20ATP%20in%20vitro%20and%20in%20vivo.pdf
- GRAB-HA: https://pubmed.ncbi.nlm.nih.gov/36924772/
- OT1.0: https://pmc.ncbi.nlm.nih.gov/articles/PMC11182738/
- GRAB neuropeptide toolkit: https://pmc.ncbi.nlm.nih.gov/articles/PMC11205257/
- GRAB-NPY1.0: https://www.frontiersin.org/journals/cellular-neuroscience/articles/10.3389/fncel.2023.1221147/full

## Current pyBer Registry Coverage

The current registry includes: GCaMP3, GCaMP6f, GCaMP6m, GCaMP6s, jGCaMP7f,
jGCaMP7s, jGCaMP8f, jGCaMP8m, jGCaMP8s, jRGECO1a, dLight1.1, dLight1.2,
dLight1.3b, RdLight1, GRAB-DA1m, GRAB-DA1h, GRAB-DA2m, GRAB-DA2h,
GRAB-5HT1.0, gGRAB-5HT3.0, rGRAB-5HT3.0, sDarken, GRAB-NE1m, GRAB-NE2m,
nLightG, GRAB-ACh3.0, iAChSnFR, SF-iGluSnFR, iGluSnFR3.v857, iGABASnFR,
iGABASnFR2, GRAB-eCB2.0, OxLight1, kLight, NOPLight, GRAB-Ado1.0, iAdo,
GRAB-ATP1.0, GRAB-HA, OT1.0, GRAB-SST, GRAB-CRF, GRAB-CCK, GRAB-NPY1.0,
GRAB-NTS, and GRAB-VIP.
