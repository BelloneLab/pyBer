# sensor_registry.py
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class SensorInfo:
    """Curated sensor facts used by pyBer preprocessing."""

    sensor_id: str
    name: str
    family: str
    target: str
    color: str
    direction: str
    excitation_nm: str
    emission_nm: str
    isobestic_nm: str
    rise: str
    decay: str
    affinity: str
    dynamic_range: str
    recommended_fs_hz: float
    recommended_lowpass_hz: float
    notes: str
    paper_url: str
    source: str
    kinetics_context: str = ""


SENSOR_UNKNOWN = "unspecified"


def _s(
    sensor_id: str,
    name: str,
    family: str,
    target: str,
    color: str,
    direction: str,
    excitation_nm: str,
    emission_nm: str,
    isobestic_nm: str,
    rise: str,
    decay: str,
    affinity: str,
    dynamic_range: str,
    recommended_fs_hz: float,
    recommended_lowpass_hz: float,
    notes: str,
    paper_url: str,
    source: str,
) -> SensorInfo:
    return SensorInfo(
        sensor_id=sensor_id,
        name=name,
        family=family,
        target=target,
        color=color,
        direction=direction,
        excitation_nm=excitation_nm,
        emission_nm=emission_nm,
        isobestic_nm=isobestic_nm,
        rise=rise,
        decay=decay,
        affinity=affinity,
        dynamic_range=dynamic_range,
        recommended_fs_hz=float(recommended_fs_hz),
        recommended_lowpass_hz=float(recommended_lowpass_hz),
        notes=notes,
        paper_url=paper_url,
        source=source,
    )


# Values are deliberately conservative and photometry-oriented. Many papers
# report cell-line, slice, imaging and in vivo kinetics separately; when no
# single photometry value is standardized, the row says so instead of inventing
# false precision.
SENSORS: List[SensorInfo] = [
    _s(
        SENSOR_UNKNOWN,
        "Unspecified sensor",
        "Unspecified",
        "unknown",
        "unknown",
        "increase",
        "465/470",
        "green channel",
        "405 control if valid",
        "unknown",
        "unknown",
        "unknown",
        "unknown",
        50.0,
        10.0,
        "Use generic preprocessing. Select a real sensor to enable sensor-aware checks.",
        "https://www.cell.com/neuron/fulltext/S0896-6273(23)00890-5",
        "Fiber photometry primer",
    ),
    _s(
        "gcamp3",
        "GCaMP3",
        "Calcium",
        "Ca2+",
        "green",
        "increase",
        "470/488",
        "~510",
        "~405 to 410",
        "slow, hundreds of ms",
        "seconds",
        "Ca2+ affinity in sub-uM range, context dependent",
        "improved over GCaMP2",
        20.0,
        4.0,
        "Legacy calcium sensor. Use for older data, but expect slow kinetics and lower SNR than GCaMP6 and later.",
        "https://www.nature.com/articles/nmeth.1398",
        "Tian et al. 2009 and early GCaMP literature",
    ),
    _s(
        "gcamp6f",
        "GCaMP6f",
        "Calcium",
        "Ca2+",
        "green",
        "increase",
        "470/488",
        "~510",
        "~405 to 410",
        "fast for calcium, tens of ms to peak in imaging assays",
        "~0.4 to 0.8 s in many neuronal assays",
        "high-affinity Ca2+ indicator",
        "large single-AP response, lower sensitivity than 6s",
        40.0,
        8.0,
        "Good when timing matters. Fiber photometry calcium transients should be positive and slower than monoamine sensors.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC3777791/",
        "Chen et al. 2013",
    ),
    _s(
        "gcamp6m",
        "GCaMP6m",
        "Calcium",
        "Ca2+",
        "green",
        "increase",
        "470/488",
        "~510",
        "~405 to 410",
        "intermediate",
        "~1 s class",
        "high-affinity Ca2+ indicator",
        "balanced sensitivity and kinetics",
        30.0,
        6.0,
        "Balanced calcium sensor. Use moderate low-pass and avoid overinterpreting subsecond timing.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC3777791/",
        "Chen et al. 2013",
    ),
    _s(
        "gcamp6s",
        "GCaMP6s",
        "Calcium",
        "Ca2+",
        "green",
        "increase",
        "470/488",
        "~510",
        "~405 to 410",
        "slow",
        "seconds",
        "high-affinity Ca2+ indicator",
        "very sensitive, slow decay",
        20.0,
        4.0,
        "Sensitive but slow. Prefer lower low-pass and do not infer precise event timing from sharp peaks.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC3777791/",
        "Chen et al. 2013",
    ),
    _s(
        "jgcamp7f",
        "jGCaMP7f",
        "Calcium",
        "Ca2+",
        "green",
        "increase",
        "470/488",
        "~510",
        "~405 to 410",
        "fast calcium",
        "subsecond",
        "engineered affinity variant",
        "high performance in neuronal populations",
        40.0,
        8.0,
        "Fast jGCaMP7 variant. Keep enough sampling to preserve fast calcium events.",
        "https://www.nature.com/articles/s41592-019-0435-6",
        "Dana et al. 2019",
    ),
    _s(
        "jgcamp7s",
        "jGCaMP7s",
        "Calcium",
        "Ca2+",
        "green",
        "increase",
        "470/488",
        "~510",
        "~405 to 410",
        "slow-sensitive calcium",
        "seconds",
        "engineered affinity variant",
        "very sensitive",
        20.0,
        4.0,
        "Sensitive jGCaMP7 variant. Use lower low-pass and treat sustained dynamics as plausible.",
        "https://www.nature.com/articles/s41592-019-0435-6",
        "Dana et al. 2019",
    ),
    _s(
        "jgcamp8f",
        "jGCaMP8f",
        "Calcium",
        "Ca2+",
        "green",
        "increase",
        "470/488",
        "~510",
        "~405 to 410",
        "very fast, millisecond-class rise in imaging assays",
        "faster than jGCaMP7f",
        "engineered affinity variant",
        "fast, less sensitive than 8s",
        60.0,
        12.0,
        "Fastest common green calcium option. Preserve sampling rate when raw data permit.",
        "https://www.nature.com/articles/s41586-023-05828-9",
        "Zhang et al. 2023",
    ),
    _s(
        "jgcamp8m",
        "jGCaMP8m",
        "Calcium",
        "Ca2+",
        "green",
        "increase",
        "470/488",
        "~510",
        "~405 to 410",
        "very fast",
        "medium decay",
        "engineered affinity variant",
        "balanced 8-series response",
        50.0,
        10.0,
        "Middle jGCaMP8 variant. Good compromise between event timing and sensitivity.",
        "https://www.nature.com/articles/s41586-023-05828-9",
        "Zhang et al. 2023",
    ),
    _s(
        "jgcamp8s",
        "jGCaMP8s",
        "Calcium",
        "Ca2+",
        "green",
        "increase",
        "470/488",
        "~510",
        "~405 to 410",
        "very fast rise",
        "slow decay",
        "engineered affinity variant",
        "very sensitive",
        40.0,
        8.0,
        "Sensitive jGCaMP8 variant. Rise is fast, but decay remains slower than 8f.",
        "https://www.nature.com/articles/s41586-023-05828-9",
        "Zhang et al. 2023",
    ),
    _s(
        "jrgeco1a",
        "jRGECO1a",
        "Calcium",
        "Ca2+",
        "red",
        "increase",
        "560/590",
        "~600",
        "not a standard 405 control",
        "fast red calcium, assay dependent",
        "~0.4 s half-decay reported in neuronal imaging",
        "Ca2+ affinity, context dependent",
        "red calcium response",
        40.0,
        8.0,
        "Red calcium sensor. Do not assume a green 405 isobestic correction is valid.",
        "https://elifesciences.org/articles/12727",
        "Dana et al. 2016",
    ),
    _s(
        "dlight11",
        "dLight1.1",
        "Dopamine",
        "dopamine",
        "green",
        "increase",
        "465/488",
        "~516",
        "~405 to 410 commonly used",
        "subsecond",
        "subsecond to seconds, clearance dependent",
        "sub-uM DA affinity family",
        "large DA response",
        60.0,
        12.0,
        "DA sensor. Fast transients are plausible; preserve sampling and use reference correction carefully.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC6287765/",
        "Patriarchi et al. 2018",
    ),
    _s(
        "dlight12",
        "dLight1.2",
        "Dopamine",
        "dopamine",
        "green",
        "increase",
        "465/488",
        "~516",
        "~405 to 410 commonly used",
        "subsecond",
        "subsecond to seconds, clearance dependent",
        "dLight affinity-tuned variant",
        "larger response than early dLight variants in many assays",
        60.0,
        12.0,
        "DA sensor with fast behavioral transients. Keep higher low-pass than slow calcium sensors.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC6287765/",
        "Patriarchi et al. 2018",
    ),
    _s(
        "dlight13b",
        "dLight1.3b",
        "Dopamine",
        "dopamine",
        "green",
        "increase",
        "465/488",
        "~516",
        "~405 to 410 commonly used",
        "subsecond",
        "subsecond to seconds, clearance dependent",
        "expanded dLight palette",
        "optimized DA response",
        60.0,
        12.0,
        "Frequently used DA variant. Sensor response shape depends strongly on DA clearance and region.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC8169200/",
        "Patriarchi and Tian labs, expanded dLight palette",
    ),
    _s(
        "grabda1m",
        "GRAB-DA1m",
        "Dopamine",
        "dopamine",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410 commonly used",
        "subsecond",
        "subsecond",
        "medium-affinity DA variant",
        "large DA response",
        60.0,
        12.0,
        "Medium-affinity GRAB dopamine sensor. Good for faster or larger DA changes.",
        "https://www.cell.com/cell/fulltext/S0092-8674(18)30845-6",
        "Sun et al. 2018",
    ),
    _s(
        "grabda1h",
        "GRAB-DA1h",
        "Dopamine",
        "dopamine",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410 commonly used",
        "subsecond",
        "subsecond",
        "high-affinity DA variant",
        "sensitive DA response",
        60.0,
        12.0,
        "High-affinity dopamine sensor. Watch for saturation in regions with large DA release.",
        "https://www.cell.com/cell/fulltext/S0092-8674(18)30845-6",
        "Sun et al. 2018",
    ),
    _s(
        "grabda2m",
        "GRAB-DA2m",
        "Dopamine",
        "dopamine",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410 commonly used",
        "subsecond",
        "subsecond",
        "next-generation medium-affinity DA variant",
        "improved sensitivity and SNR",
        60.0,
        12.0,
        "Next-generation DA sensor. Strong fast positive transients are expected.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC7648260/",
        "Sun et al. 2020",
    ),
    _s(
        "grabda2h",
        "GRAB-DA2h",
        "Dopamine",
        "dopamine",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410 commonly used",
        "subsecond",
        "subsecond",
        "next-generation high-affinity DA variant",
        "improved sensitivity and SNR",
        60.0,
        12.0,
        "High-affinity next-generation DA sensor. Watch for saturation and movement-coupled reference bleed.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC7648260/",
        "Sun et al. 2020",
    ),
    _s(
        "grab5ht10",
        "GRAB-5HT1.0",
        "Serotonin",
        "serotonin",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410 commonly used",
        "tau_on ~70 ms in reported assays",
        "subsecond to seconds, context dependent",
        "EC50 ~22 nM reported",
        "sensitive 5-HT response",
        50.0,
        10.0,
        "Serotonin sensor. Positive transients are expected, but region and transporter kinetics shape decay.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC8544647/",
        "Wan et al. 2021",
    ),
    _s(
        "grab5ht30",
        "gGRAB-5HT3.0",
        "Serotonin",
        "serotonin",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410 commonly used",
        "fast, assay dependent",
        "subsecond to seconds, context dependent",
        "improved 5-HT affinity variants",
        "large fluorescence increase reported",
        50.0,
        10.0,
        "Improved green 5-HT sensor. Use sensor-aware polarity and saturation checks.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC11377854/",
        "Li lab improved GRAB-5HT sensors, 2024",
    ),
    _s(
        "rgrab5ht30",
        "rGRAB-5HT3.0",
        "Serotonin",
        "serotonin",
        "red",
        "increase",
        "560/590",
        "~600",
        "not a standard 405 control",
        "fast, assay dependent",
        "subsecond to seconds",
        "red 5-HT variant",
        "red fluorescence increase",
        50.0,
        10.0,
        "Red 5-HT sensor. Use red-channel controls, not a default green 405 assumption.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC11377854/",
        "Li lab improved GRAB-5HT sensors, 2024",
    ),
    _s(
        "sdarken",
        "sDarken",
        "Serotonin",
        "serotonin",
        "green",
        "decrease",
        "465/488",
        "~510",
        "not standardized",
        "faster than GRAB-5HT in review comparisons",
        "paper dependent",
        "affinity variants available, around 100 nM class for main version",
        "fluorescence decreases on 5-HT binding",
        50.0,
        10.0,
        "Darkening serotonin sensor. Expected biological events are downward unless user inverts polarity for display.",
        "https://www.nature.com/articles/s41467-022-35200-w",
        "Kubitschke et al. 2022",
    ),
    _s(
        "grabne1m",
        "GRAB-NE1m",
        "Norepinephrine",
        "norepinephrine",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410 commonly used",
        "subsecond",
        "subsecond to seconds",
        "nanomolar to micromolar family range",
        "up to ~230% peak response reported for family",
        50.0,
        10.0,
        "Original GRAB norepinephrine family. Check DA cross-sensitivity depending on region.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC6533151/",
        "Feng et al. 2019",
    ),
    _s(
        "grabne2m",
        "GRAB-NE2m",
        "Norepinephrine",
        "norepinephrine",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410 commonly used",
        "subsecond",
        "subsecond to seconds",
        "next-generation medium-affinity NE variant",
        "improved response and sensitivity",
        50.0,
        10.0,
        "Next-generation NE sensor. Positive transients can be fast and event-boundary locked.",
        "https://pubmed.ncbi.nlm.nih.gov/38547869/",
        "Feng et al. 2024",
    ),
    _s(
        "nlightg",
        "nLightG",
        "Norepinephrine",
        "norepinephrine",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410 likely usable, verify setup",
        "fast, improved over GRAB-NE in comparisons",
        "clearance dependent",
        "sensitive NE indicator",
        "improved sensitivity and selectivity",
        60.0,
        12.0,
        "Green NE sensor. Faster kinetics than GRAB-NE in reported comparisons.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC7615053/",
        "Kagiampaki et al. 2023",
    ),
    _s(
        "grabach30",
        "GRAB-ACh3.0",
        "Acetylcholine",
        "acetylcholine",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410 commonly used",
        "tau_on ~0.09 s in reported stimulation assay",
        "tau_off ~0.91 s in reported stimulation assay",
        "physiologically relevant ACh affinity",
        "optimized ACh response",
        50.0,
        10.0,
        "Optimized green ACh sensor. Subsecond rises are plausible, but decays are slower.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC7606762/",
        "Jing et al. 2020",
    ),
    _s(
        "iachsnfr",
        "iAChSnFR",
        "Acetylcholine",
        "acetylcholine",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify variant",
        "rapid rise",
        "rapid decay",
        "binding-protein ACh sensor",
        "large fluorescence changes",
        60.0,
        12.0,
        "Binding-protein ACh sensor with fast kinetics. Preserve subsecond dynamics.",
        "https://authors.library.caltech.edu/records/vezvm-wdv47",
        "Borden et al. 2020",
    ),
    _s(
        "iglusnfr",
        "SF-iGluSnFR",
        "Glutamate",
        "glutamate",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify setup",
        "fast synaptic reporter",
        "fast, variant dependent",
        "sub-uM to mM variants",
        "~5x class for original iGluSnFR family",
        80.0,
        20.0,
        "Glutamate can be very fast. Avoid heavy smoothing and keep high sampling when possible.",
        "https://pubmed.ncbi.nlm.nih.gov/30377363/",
        "Marvin et al. 2018",
    ),
    _s(
        "iglusnfr3",
        "iGluSnFR3.v857",
        "Glutamate",
        "glutamate",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify setup",
        "improved activation kinetics",
        "fast, variant dependent",
        "glutamate affinity variant",
        "improved synaptic response",
        100.0,
        25.0,
        "Current high-performance glutamate family. Keep high bandwidth for synaptic signals.",
        "https://www.nature.com/articles/s41592-023-01863-6",
        "Aggarwal et al. 2023",
    ),
    _s(
        "igabasnfr",
        "iGABASnFR",
        "GABA",
        "GABA",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify setup",
        "atypical, slower than glutamate",
        "paper dependent",
        "~240 uM reported for original sensor",
        "GABA response",
        50.0,
        10.0,
        "Original GABA sensor. Kinetics are not a simple single exponential in all assays.",
        "https://www.nature.com/articles/s41592-019-0471-2",
        "Marvin et al. 2019",
    ),
    _s(
        "igabasnfr2",
        "iGABASnFR2",
        "GABA",
        "GABA",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify setup",
        "faster than iGABASnFR1",
        "single-exponential in reported comparison",
        "EC50 ~6.4 uM reported in secondary summary",
        "4x sensitivity improvement reported",
        60.0,
        12.0,
        "Improved GABA sensor. Preserve moderate bandwidth but expect slower kinetics than glutamate.",
        "https://www.janelia.org/open-science/igabasnfr2",
        "Janelia iGABASnFR2 resources and 2025 preprint",
    ),
    _s(
        "grabecb20",
        "GRAB-eCB2.0",
        "Endocannabinoid",
        "2-AG, AEA",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify setup",
        "seconds",
        "seconds",
        "low-uM eCB response",
        "robust physiological response",
        20.0,
        4.0,
        "Endocannabinoid sensor. Events are slow relative to monoamine sensors.",
        "https://pubmed.ncbi.nlm.nih.gov/34764491/",
        "Dong et al. 2021",
    ),
    _s(
        "oxlight1",
        "OxLight1",
        "Orexin",
        "orexin A/B",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify setup",
        "rapid in vivo kinetics for neuropeptide sensor",
        "seconds, behavior and clearance dependent",
        "orexin receptor based",
        "sensitive orexin response",
        20.0,
        4.0,
        "Orexin neuropeptide sensor. Expect slow behavior-scale transients, not spike-like events.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC8831244/",
        "Duffet et al. 2022",
    ),
    _s(
        "klight",
        "kLight",
        "Opioid",
        "dynorphin, kappa-opioid ligands",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify setup",
        "peptide sensor kinetics, paper dependent",
        "seconds, ligand dependent",
        "opioid peptide receptor based",
        "ligand-dependent fluorescence response",
        20.0,
        4.0,
        "Opioid peptide sensor. Treat responses as slow neuromodulator or peptide dynamics.",
        "https://www.nature.com/articles/s41593-024-01697-1",
        "Massengill et al. 2024",
    ),
    _s(
        "noplight",
        "NOPLight",
        "Opioid",
        "nociceptin/orphanin FQ",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify setup",
        "seconds, ligand and assay dependent",
        "seconds",
        "NOP receptor based",
        "ligand-dependent fluorescence response",
        20.0,
        4.0,
        "N/OFQ opioid peptide sensor. In vivo fiber photometry validates slow endogenous peptide dynamics.",
        "https://www.nature.com/articles/s41467-024-49712-0",
        "Zhou et al. NOPLight, 2024",
    ),
    _s(
        "grabado10",
        "GRAB-Ado1.0",
        "Adenosine",
        "adenosine",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify setup",
        "~100 ms class in reported sensor description",
        "seconds, extracellular clearance dependent",
        "EC50 ~60 nM reported in early GRABAdo work",
        "sensitive adenosine response",
        40.0,
        8.0,
        "Adenosine sensor. Expect neuromodulatory timescale transients and verify purine selectivity controls.",
        "https://www.pnas.org/doi/10.1073/pnas.2212387120",
        "GRABAdo in vivo adenosine studies",
    ),
    _s(
        "iado",
        "iAdo",
        "Adenosine",
        "adenosine",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify setup",
        "high-performance, assay dependent",
        "seconds, clearance dependent",
        "adenosine deaminase based",
        "high-performance extracellular adenosine response",
        40.0,
        8.0,
        "Newer adenosine sensor. Use sensor-aware low-pass but inspect validation because field standards are still evolving.",
        "https://www.nature.com/articles/s41467-025-59530-7",
        "High-performance adenosine sensor, 2025",
    ),
    _s(
        "grabatp10",
        "GRAB-ATP1.0",
        "ATP",
        "ATP",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify setup",
        "subsecond to seconds, assay dependent",
        "seconds",
        "extracellular ATP sensor",
        "robust ATP fluorescence response",
        40.0,
        8.0,
        "Extracellular ATP sensor. Expect slower purinergic dynamics than glutamate and check pharmacology controls.",
        "https://www.yulonglilab.org/pdfs/A%20sensitive%20GRAB%20sensor%20for%20detecting%20extracellular%20ATP%20in%20vitro%20and%20in%20vivo.pdf",
        "GRABATP1.0 in vitro and in vivo report",
    ),
    _s(
        "grabha",
        "GRAB-HA",
        "Histamine",
        "histamine",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify setup",
        "high temporal resolution in slice and in vivo assays",
        "seconds, sleep-wake state dependent",
        "histamine receptor based",
        "robust extracellular histamine response",
        40.0,
        8.0,
        "Histamine sensor. Dynamics can be state dependent, so avoid overly aggressive baseline windows across sleep-wake transitions.",
        "https://pubmed.ncbi.nlm.nih.gov/36924772/",
        "GRABHA sensors, Neuron 2023",
    ),
    _s(
        "ot10",
        "OT1.0",
        "Oxytocin",
        "oxytocin",
        "green",
        "increase",
        "400/465/488 depending on setup",
        "~510",
        "not standardized",
        "temporal oxytocin release sensor, assay dependent",
        "seconds",
        "oxytocin receptor based",
        "detects oxytocin release in compartments",
        20.0,
        4.0,
        "Oxytocin neuropeptide sensor. Treat responses as slow volume-transmission signals.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC11182738/",
        "OT1.0 oxytocin sensor, 2023",
    ),
    _s(
        "grabnp_sst",
        "GRAB-SST",
        "Neuropeptide",
        "somatostatin",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify setup",
        "neuropeptide sensor, seconds class",
        "seconds",
        "SST receptor based",
        "selective GRAB neuropeptide response",
        20.0,
        4.0,
        "Part of the GRAB neuropeptide toolkit. Use slow settings and validate region-specific peptide identity.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC11205257/",
        "GRAB neuropeptide toolkit, Science 2023",
    ),
    _s(
        "grabnp_crf",
        "GRAB-CRF",
        "Neuropeptide",
        "corticotropin-releasing factor",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify setup",
        "neuropeptide sensor, seconds class",
        "seconds",
        "CRF receptor based",
        "selective GRAB neuropeptide response",
        20.0,
        4.0,
        "CRF neuropeptide sensor. Expect slow stress-related dynamics rather than sharp calcium-like events.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC11205257/",
        "GRAB neuropeptide toolkit, Science 2023",
    ),
    _s(
        "grabnp_cck",
        "GRAB-CCK",
        "Neuropeptide",
        "cholecystokinin",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify setup",
        "neuropeptide sensor, seconds class",
        "seconds",
        "CCK receptor based",
        "selective GRAB neuropeptide response",
        20.0,
        4.0,
        "CCK neuropeptide sensor. Use low-pass settings appropriate for peptide release.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC11205257/",
        "GRAB neuropeptide toolkit, Science 2023",
    ),
    _s(
        "grabnp_npy",
        "GRAB-NPY1.0",
        "Neuropeptide",
        "neuropeptide Y",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify setup",
        "neuropeptide sensor, seconds class",
        "seconds",
        "NPY receptor based",
        "dose-dependent NPY response",
        20.0,
        4.0,
        "NPY sensor. Endogenous release has been shown in neuronal preparations; in vivo interpretation needs controls.",
        "https://www.frontiersin.org/journals/cellular-neuroscience/articles/10.3389/fncel.2023.1221147/full",
        "GRAB-NPY1.0 characterization, 2023",
    ),
    _s(
        "grabnp_nts",
        "GRAB-NTS",
        "Neuropeptide",
        "neurotensin",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify setup",
        "neuropeptide sensor, seconds class",
        "seconds",
        "NTS receptor based",
        "selective GRAB neuropeptide response",
        20.0,
        4.0,
        "Neurotensin sensor from the GRAB neuropeptide toolkit. Prefer slow transient settings.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC11205257/",
        "GRAB neuropeptide toolkit, Science 2023",
    ),
    _s(
        "grabnp_vip",
        "GRAB-VIP",
        "Neuropeptide",
        "vasoactive intestinal peptide",
        "green",
        "increase",
        "465/488",
        "~510",
        "~405 to 410, verify setup",
        "neuropeptide sensor, seconds class",
        "seconds",
        "VIP receptor based",
        "selective GRAB neuropeptide response",
        20.0,
        4.0,
        "VIP neuropeptide sensor from the GRAB toolkit. Do not use high-bandwidth calcium assumptions.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC11205257/",
        "GRAB neuropeptide toolkit, Science 2023",
    ),
    _s(
        "rdlight1",
        "RdLight1",
        "Dopamine",
        "dopamine",
        "red",
        "increase",
        "560/590",
        "~600",
        "not a standard 405 control",
        "subsecond, color-variant dependent",
        "subsecond to seconds",
        "dLight red palette",
        "red dopamine response",
        50.0,
        10.0,
        "Red dopamine sensor for multiplexing. Do not assume green 405 correction.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC8169200/",
        "Patriarchi lab expanded dLight palette",
    ),
]


_KINETIC_OVERRIDES: Dict[str, Dict[str, str]] = {
    "gcamp3": {
        "rise": "t1/2 rise 83 +/- 2 ms",
        "decay": "t1/2 decay 610 +/- 32 ms",
        "kinetics_context": (
            "Cultured hippocampal slice AP response from Tian et al. GCaMP3 work; "
            "bulk photometry events can be broader."
        ),
    },
    "gcamp6f": {
        "rise": "tau_rise 50-100 ms",
        "decay": "tau_decay 200-300 ms",
        "kinetics_context": (
            "Simultaneous spikes/imaging estimates for 1-5 AP GCaMP6f events; "
            "population photometry and brain temperature can broaden decay."
        ),
    },
    "gcamp6m": {
        "rise": "intermediate 6f/6s, assay dependent",
        "decay": "t1/2 decay ~0.73 s at 37 C",
        "kinetics_context": (
            "GCaMP6m is the middle 6-series variant. A dopamine-neuron imaging benchmark "
            "reported median half-life near 0.725 s at 37 C, but rise values vary by preparation."
        ),
    },
    "gcamp6s": {
        "rise": "tau_rise 150-200 ms",
        "decay": "tau_decay ~750 ms",
        "kinetics_context": (
            "Slow-sensitive GCaMP6 estimate from simultaneous spikes/imaging reports for "
            "multi-AP events; fiber photometry can be slower."
        ),
    },
    "jgcamp7f": {
        "rise": "1AP half-rise ~25-27 ms",
        "decay": "1AP half-decay ~180 ms; 10AP ~520 ms",
        "kinetics_context": (
            "Janelia/Dana jGCaMP7 measurements. Exact decay depends on 1AP versus train "
            "and cultured screen versus in vivo single-cell assay."
        ),
    },
    "jgcamp7s": {
        "rise": "1AP half-rise ~56 ms",
        "decay": "10AP half-decay ~1.69 s",
        "kinetics_context": (
            "Dana et al. jGCaMP7s measurements. This is a high-SNR slow variant, so "
            "photometry peaks should not be treated as millisecond timing markers."
        ),
    },
    "jgcamp8f": {
        "rise": "half-rise 7.1 +/- 0.74 ms",
        "decay": "half-decay 67.4 +/- 11.2 ms",
        "kinetics_context": (
            "Janelia field-stimulated cultured neuron screen, 1 AP at 80 Hz context; "
            "in vivo fiber photometry is release and population filtered."
        ),
    },
    "jgcamp8m": {
        "rise": "half-rise 7.1 +/- 0.61 ms",
        "decay": "half-decay 118.3 +/- 13.2 ms",
        "kinetics_context": (
            "Janelia field-stimulated cultured neuron screen, 1 AP at 80 Hz context; "
            "8m trades fast rise for slower decay and better sensitivity than 8f."
        ),
    },
    "jgcamp8s": {
        "rise": "half-rise 10.1 +/- 0.86 ms",
        "decay": "half-decay 306.7 +/- 32.2 ms",
        "kinetics_context": (
            "Janelia field-stimulated cultured neuron screen, 1 AP at 80 Hz context; "
            "8s is sensitive but its decay is still calcium-limited."
        ),
    },
    "jrgeco1a": {
        "rise": "2AP half-rise similar to GCaMP6f",
        "decay": "2AP half-decay similar to GCaMP6s",
        "kinetics_context": (
            "Dana et al. report jRGECO1a rise/decay comparable to GCaMP6f/GCaMP6s, "
            "not one universal fiber-photometry tau."
        ),
    },
    "dlight11": {
        "rise": "on ~10 ms",
        "decay": "off ~100 ms",
        "kinetics_context": (
            "dLight1 family in vitro kinetics. In vivo photometry peaks are additionally "
            "shaped by dopamine release, uptake, and optical averaging."
        ),
    },
    "dlight12": {
        "rise": "on ~10 ms",
        "decay": "off ~100 ms",
        "kinetics_context": (
            "dLight1.1/1.2 family kinetics from Patriarchi et al.; treat 10/100 ms "
            "as sensor speed, not guaranteed in vivo event width."
        ),
    },
    "dlight13b": {
        "rise": "peak ~10 ms single-stim; ~45 ms 2-stim",
        "decay": "tau ~120 ms single; 182 +/- 44 ms 2-stim",
        "kinetics_context": (
            "dLight1.3b hotspot line-scan measurements. Bulk fiber photometry may be "
            "slower because it averages release sites and uptake."
        ),
    },
    "rdlight1": {
        "rise": "tau_on 126 +/- 15 ms",
        "decay": "tau_off 320 +/- 42 ms",
        "kinetics_context": (
            "Expanded dLight palette paper measured RdLight1 rise and decay by fitting "
            "the average response; red-channel controls must be validated separately."
        ),
    },
    "grabda1m": {
        "rise": "tau_on ~60 ms",
        "decay": "tau_off ~0.7 s",
        "kinetics_context": (
            "First-generation GRAB-DA medium-affinity values from rapid DA application "
            "and slice/cell characterization."
        ),
    },
    "grabda1h": {
        "rise": "tau_on ~130 ms",
        "decay": "tau_off ~2.5 s",
        "kinetics_context": (
            "First-generation high-affinity GRAB-DA is slower off than GRAB-DA1m, "
            "consistent with tighter ligand binding."
        ),
    },
    "grabda2m": {
        "rise": "tau_on ~80 ms",
        "decay": "tau_off ~0.6-3 s",
        "kinetics_context": (
            "Dual-color GRAB-DA comparison reports roughly 80 ms on kinetics and "
            "0.6-3 s off kinetics across affinity variants."
        ),
    },
    "grabda2h": {
        "rise": "tau_on ~80 ms",
        "decay": "tau_off ~0.6-3 s",
        "kinetics_context": (
            "Dual-color GRAB-DA comparison reports roughly 80 ms on kinetics and "
            "0.6-3 s off kinetics across affinity variants; high-affinity traces can linger."
        ),
    },
    "grab5ht10": {
        "rise": "tau_on ~0.2 s",
        "decay": "tau_off ~3.1 s",
        "kinetics_context": (
            "GRAB-5HT1.0 response to 5-HT application; stimulation-evoked tissue "
            "decay also reflects transporter clearance."
        ),
    },
    "grab5ht30": {
        "rise": "rise ~0.25 s",
        "decay": "decay ~1.39 s",
        "kinetics_context": (
            "GRAB5HT3.0 hippocampal photometry report gives a shorter rise than decay; "
            "exact time course still depends on region and SERT."
        ),
    },
    "rgrab5ht30": {
        "rise": "subsecond, variant dependent",
        "decay": "seconds to tens of seconds in vivo",
        "kinetics_context": (
            "Improved red GRAB-5HT variants have assay-specific kinetics; red in vivo "
            "washout and endogenous release can be much slower than sensor binding."
        ),
    },
    "sdarken": {
        "rise": "<1 s response in patch-clamp fluorometry",
        "decay": "tau_off 1.24 s in in vivo fit",
        "kinetics_context": (
            "sDarken is a darkening 5-HT sensor. The 1.24 s value is an in vivo "
            "movement-stop fluorescence decay fit, not a universal ligand-off constant."
        ),
    },
    "grabne1m": {
        "rise": "subsecond, exact tau assay dependent",
        "decay": "subsecond to seconds, exact tau assay dependent",
        "kinetics_context": (
            "Original GRAB-NE reports high temporal resolution, but comparable absolute "
            "on/off constants are not consistently standardized across registry sources."
        ),
    },
    "grabne2m": {
        "rise": "tau_on 0.12 s",
        "decay": "tau_off 1.72 s",
        "kinetics_context": (
            "GRAB-NE2m next-generation characterization. GRAB-NE2h reports similar "
            "on kinetics and tau_off around 1.93 s."
        ),
    },
    "nlightg": {
        "rise": "about 8x faster than GRAB-NE",
        "decay": "about 3x faster than GRAB-NE",
        "kinetics_context": (
            "nLightG paper reports relative speed improvement over GRAB-NE; absolute "
            "in vivo photometry kinetics remain dominated by NE clearance."
        ),
    },
    "grabach30": {
        "rise": "tau_on ~0.09 s",
        "decay": "tau_off ~0.91 s",
        "kinetics_context": (
            "GRAB-ACh3.0 sensor characterization. Endogenous stimulation protocols can "
            "show slower decay if acetylcholine clearance is limiting."
        ),
    },
    "iachsnfr": {
        "rise": "tau_on 280 +/- 32 ms",
        "decay": "tau_off 762 +/- 75 ms",
        "kinetics_context": (
            "iAChSnFR activation and inactivation constants from the original sensor "
            "characterization/preprint assay; in vivo waveforms remain release dependent."
        ),
    },
    "iglusnfr": {
        "rise": "fast synaptic reporter, variant specific",
        "decay": "10-100x slower than glutamate lifetime",
        "kinetics_context": (
            "SF-iGluSnFR has faster fluorescence than calcium indicators, but synaptic "
            "optical decays vary strongly with expression, localization, and uptake."
        ),
    },
    "iglusnfr3": {
        "rise": "1AP rise 18.9 +/- 0.5 ms",
        "decay": "decay tau <30 ms benchmark",
        "kinetics_context": (
            "iGluSnFR3.v857 cultured-neuron field stimulation reports faster 1AP rise "
            "than WT; later benchmarks put iGluSnFR3 decay near 29 ms."
        ),
    },
    "igabasnfr": {
        "rise": "slower than glutamate, assay dependent",
        "decay": "paper and expression dependent",
        "kinetics_context": (
            "Original iGABASnFR kinetics are harder to summarize with one tau because "
            "tissue GABA clearance and sensor expression strongly shape decay."
        ),
    },
    "igabasnfr2": {
        "rise": "rise 72 +/- 8 ms",
        "decay": "tau_off ~73 ms class",
        "kinetics_context": (
            "iGABASnFR2 screening/eLife report; alternate 10-AP tests report about "
            "38 +/- 10 ms rise, so pyBer treats it as moderate-bandwidth."
        ),
    },
    "grabecb20": {
        "rise": "tau_rise ~1.0 s",
        "decay": "tau_decay ~6.3 s",
        "kinetics_context": (
            "GRAB-eCB2.0 in vivo/slice seizure model fit. Endocannabinoid events are "
            "seconds scale compared with monoamine sensors."
        ),
    },
    "oxlight1": {
        "rise": "subsecond activation, assay dependent",
        "decay": "seconds, release/clearance dependent",
        "kinetics_context": (
            "OxLight1 supports fast orexin detection for a neuropeptide sensor, but the "
            "published in vivo photometry traces should be interpreted on behavior timescales."
        ),
    },
    "klight": {
        "rise": "seconds, ligand and variant dependent",
        "decay": "tens of seconds to minutes",
        "kinetics_context": (
            "kLight variants differ in off kinetics; published dynorphin measurements "
            "reflect slow peptide diffusion and clearance as much as sensor binding."
        ),
    },
    "noplight": {
        "rise": "tau_on 595 +/- 69 ms",
        "decay": "tau_off 30-60 s in tissue",
        "kinetics_context": (
            "NOPLight activation measured with direct N/OFQ application; tissue off "
            "kinetics were 30.9 +/- 4.5 to 53.1 +/- 6.6 s across concentrations."
        ),
    },
    "grabado10": {
        "rise": "in vivo Ado rise ~30 s reported",
        "decay": "seconds to tens of seconds, ENT dependent",
        "kinetics_context": (
            "GRAB-Ado reports slow extracellular adenosine dynamics in vivo; this row "
            "uses biological release/clearance times because a universal sensor tau is not standardized."
        ),
    },
    "iado": {
        "rise": "seconds, cell-type and event dependent",
        "decay": "seconds to tens of seconds, cell-type dependent",
        "kinetics_context": (
            "HypnoS/iAdo reports intracellular adenosine; seizure and sleep-wake kinetics "
            "depend on cell type and metabolic pathway."
        ),
    },
    "grabatp10": {
        "rise": "tau_on ~28 ms",
        "decay": "tau_off ~283 ms",
        "kinetics_context": (
            "GRAB-ATP1.0 rapid response kinetics from the Neuron sensor characterization; "
            "extracellular ATP release itself can be slower in tissue."
        ),
    },
    "grabha": {
        "rise": "rise 0.3-0.6 s",
        "decay": "decay 1.4-2.3 s",
        "kinetics_context": (
            "GRAB-HA histamine sensor characterization reports this rise/decay range; "
            "sleep-wake photometry can be slower due to state dynamics."
        ),
    },
    "ot10": {
        "rise": "tau_on 480 +/- 84 ms",
        "decay": "seconds, compartment dependent",
        "kinetics_context": (
            "OT1.0 has a reported subsecond on constant in slice assays. Decay depends "
            "on compartment, stimulus train, and peptide clearance."
        ),
    },
    "grabnp_sst": {
        "rise": "tau_on 300-400 ms class",
        "decay": "tau_off 3-12 s class",
        "kinetics_context": (
            "GRAB neuropeptide toolkit reports general on/off ranges for peptide sensors; "
            "individual ligand/receptor rows should be validated experimentally."
        ),
    },
    "grabnp_crf": {
        "rise": "tau_on 300-400 ms class",
        "decay": "tau_off 3-12 s class",
        "kinetics_context": (
            "GRAB neuropeptide toolkit reports general on/off ranges for peptide sensors; "
            "CRF release measurements can be behavior and stress-state dependent."
        ),
    },
    "grabnp_cck": {
        "rise": "tau_on 300-400 ms class",
        "decay": "tau_off 3-12 s class",
        "kinetics_context": (
            "GRAB neuropeptide toolkit reports general on/off ranges for peptide sensors; "
            "CCK release should be interpreted on slow neuromodulatory timescales."
        ),
    },
    "grabnp_npy": {
        "rise": "tau_on 300-400 ms class",
        "decay": "tau_off 3-12 s class",
        "kinetics_context": (
            "GRAB neuropeptide toolkit kinetics are the best comparable prior; GRAB-NPY1.0 "
            "papers emphasize activation and endogenous detection more than one universal tau."
        ),
    },
    "grabnp_nts": {
        "rise": "tau_on 300-400 ms class",
        "decay": "tau_off 3-12 s class",
        "kinetics_context": (
            "GRAB neuropeptide toolkit reports general on/off ranges for peptide sensors; "
            "NTS timing is expected to be seconds scale in bulk photometry."
        ),
    },
    "grabnp_vip": {
        "rise": "tau_on 300-400 ms class",
        "decay": "tau_off 3-12 s class",
        "kinetics_context": (
            "GRAB neuropeptide toolkit reports general on/off ranges for peptide sensors; "
            "VIP timing should not be analyzed with calcium-style sharp-event assumptions."
        ),
    },
}

for _idx, _sensor in enumerate(SENSORS):
    _override = _KINETIC_OVERRIDES.get(_sensor.sensor_id)
    if _override:
        SENSORS[_idx] = replace(_sensor, **_override)


SENSOR_BY_ID: Dict[str, SensorInfo] = {s.sensor_id: s for s in SENSORS}


def all_sensors() -> List[SensorInfo]:
    return list(SENSORS)


def get_sensor(sensor_id: str) -> SensorInfo:
    return SENSOR_BY_ID.get(str(sensor_id or "").strip(), SENSOR_BY_ID[SENSOR_UNKNOWN])


def sensor_options() -> List[str]:
    return [s.name for s in SENSORS]


def sensor_name(sensor_id: str) -> str:
    return get_sensor(sensor_id).name


def assess_sensor_trace(sensor_id: str, time: np.ndarray, signal: np.ndarray) -> Dict[str, object]:
    """Return a compact sensor-vs-trace sanity check.

    This is intentionally descriptive, not a classifier. It checks polarity,
    high-frequency burden, and whether events look faster than the selected
    sensor family can plausibly report.
    """
    sensor = get_sensor(sensor_id)
    t = np.asarray(time, float)
    y = np.asarray(signal, float)
    n = min(t.size, y.size)
    if n < 20:
        return {"status": "unknown", "message": "Too few samples for sensor check.", "metrics": {}}
    t = t[:n]
    y = y[:n]
    m = np.isfinite(t) & np.isfinite(y)
    if np.sum(m) < 20:
        return {"status": "unknown", "message": "Trace has too few finite samples for sensor check.", "metrics": {}}
    t = t[m]
    y = y[m]
    dt = float(np.nanmedian(np.diff(t))) if t.size > 2 else float("nan")
    fs = 1.0 / dt if np.isfinite(dt) and dt > 0 else float("nan")

    med = float(np.nanmedian(y))
    centered = y - med
    mad = 1.4826 * float(np.nanmedian(np.abs(centered)))
    if not np.isfinite(mad) or mad <= 1e-12:
        mad = float(np.nanstd(centered))
    if not np.isfinite(mad) or mad <= 1e-12:
        mad = 1.0
    z = centered / mad
    pos_tail = float(np.nanquantile(z, 0.99))
    neg_tail = float(-np.nanquantile(z, 0.01))
    hf = np.diff(z)
    hf_mad = 1.4826 * float(np.nanmedian(np.abs(hf - np.nanmedian(hf)))) if hf.size else 0.0
    expected_down = sensor.direction.lower().startswith("decrease")
    polarity_ok = (neg_tail >= 0.75 * pos_tail) if expected_down else (pos_tail >= 0.75 * neg_tail)

    messages: List[str] = []
    status = "ok"
    if sensor.sensor_id != SENSOR_UNKNOWN and not polarity_ok:
        status = "warn"
        expected = "downward" if expected_down else "upward"
        observed = "upward" if pos_tail > neg_tail else "downward"
        messages.append(f"{sensor.name} is expected to report {expected} events, but the raw trace is dominated by {observed} excursions.")
    if np.isfinite(fs) and fs < max(2.0, 0.5 * sensor.recommended_fs_hz):
        status = "warn"
        messages.append(f"Sampling is {fs:.1f} Hz; {sensor.name} is usually better recorded near {sensor.recommended_fs_hz:.0f} Hz or above.")
    if hf_mad > 3.0 and sensor.family in {"Calcium", "Endocannabinoid", "Orexin", "Opioid"}:
        status = "warn"
        messages.append(f"High-frequency jitter is large for a slow {sensor.family} sensor; inspect motion artifacts and low-pass settings.")
    if not messages:
        messages.append(f"Trace polarity and sampling are broadly consistent with {sensor.name}.")

    return {
        "status": status,
        "message": " ".join(messages),
        "metrics": {
            "fs_hz": fs,
            "positive_tail_z": pos_tail,
            "negative_tail_z": neg_tail,
            "hf_mad_z": hf_mad,
            "expected_direction": sensor.direction,
        },
    }
