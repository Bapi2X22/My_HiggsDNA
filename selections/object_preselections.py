import awkward as ak
import numpy as np

def object_preselections(self, 
    photons: ak.Array,
    jets: ak.Array,
    electrons: ak.Array,
    muons: ak.Array,
    year="2018",
):
    """
    Apply object-level preselection only.
    Returns selected photons, bjets, and leptons.
    No event-level requirements applied here.
    """

    # ------------------------
    # Year-dependent lepton pT cuts
    # ------------------------
    if year.startswith("2016"):
        ele_pt_cut, mu_pt_cut = 27, 26
    elif year == "2017":
        ele_pt_cut, mu_pt_cut = 33, 29
    elif year == "2018":
        ele_pt_cut, mu_pt_cut = 33, 26
    elif year == "2024":
        ele_pt_cut, mu_pt_cut = 30, 24
    else:
        raise ValueError(f"Unknown year {year}")

    # ========================
    # ELECTRONS
    # ========================
    good_electrons = (
        (electrons.pt > ele_pt_cut) &
        (np.abs(electrons.eta) < 2.5) &
        ~((np.abs(electrons.eta) > 1.44) & (np.abs(electrons.eta) < 1.57)) &
        (electrons.mvaIso_WP80) &
        (electrons.pfRelIso03_all < 0.15)
    )

    selected_electrons = electrons[good_electrons]

    # ========================
    # MUONS
    # ========================
    good_muons = (
        (muons.pt > mu_pt_cut) &
        (np.abs(muons.eta) < 2.4) &
        (muons.pfRelIso03_all < 0.15)
    )

    selected_muons = muons[good_muons]

    # Combine leptons (object level only)
    selected_leptons = ak.concatenate(
        [selected_electrons, selected_muons],
        axis=1
    )

    # ========================
    # BJETS
    # ========================
    good_jets = (
        (jets.pt > 20) &
        (np.abs(jets.eta) < 2.4) &
        (jets.btagUParTAK4B > 0.1272)
    )

    selected_bjets = jets[good_jets]

    # ========================
    # PHOTONS
    # ========================
    abs_eta = np.abs(photons.eta)

    valid_eta = (
        (abs_eta <= 2.5) &
        ~((abs_eta >= 1.442) & (abs_eta <= 1.566))
    )

    is_barrel = abs_eta < 1.442
    is_endcap = (abs_eta > 1.566) & (abs_eta < 2.5)

    barrel_cut = is_barrel & (photons.mvaID > -0.02)
    endcap_cut = is_endcap & (photons.mvaID > -0.26)

    good_photons = (
        (photons.pt > 10) &
        valid_eta &
        (barrel_cut | endcap_cut) &
        (~photons.pixelSeed)
    )

    selected_photons = photons[good_photons]

    return selected_photons, selected_bjets, selected_leptons, selected_electrons, selected_muons
