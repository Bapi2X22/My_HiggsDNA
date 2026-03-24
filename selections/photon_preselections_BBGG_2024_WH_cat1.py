import awkward as ak
import numpy as np

def delta_r_manual(obj1, obj2):
    deta = obj1.eta[:, None] - obj2.eta
    dphi = np.abs(obj1.phi[:, None] - obj2.phi)
    dphi = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return np.sqrt(deta**2 + dphi**2)

def photon_preselections_BBGG_2024_WH_cat1(self,
    photons: ak.Array,
    events: ak.Array,
    electron_veto=True,
    revert_electron_veto=False,
    year="2024",
    IsFlag=False):
    """
    Apply full preselection on leptons, jets, and photons.
    Finally return only photons from events that pass all criteria.
    """

    print("Number of events before preselection:", len(events))

    print("Year:", year)

    # ------------------------
    # Lepton selection
    # ------------------------
    if year.startswith("2016"):
        ele_pt_cut, mu_pt_cut = 27, 26
    elif year == "2017":
        ele_pt_cut, mu_pt_cut = 33, 29
    elif year == "2018":
        # ele_pt_cut, mu_pt_cut = 33, 26
        ele_pt_cut, mu_pt_cut = 33, 26

    elif year == "2022":  # Just for testing
        ele_pt_cut, mu_pt_cut = 33, 26
    elif year == "2022EE":  # Just for testing
        ele_pt_cut, mu_pt_cut = 33, 26
    elif year == "2024":  # Just for testing
        ele_pt_cut, mu_pt_cut = 30, 24

    else:
        raise ValueError(f"Unknown year {year}")
    
    #-------------------------
    # b-tagging working point
    #-------------------------

    if year == "2016APV":
        wp_medium = 0.2598
    elif year == "2016":
        wp_medium = 0.2489
    elif year == "2017":
        wp_medium = 0.304
    elif year == "2018":
        wp_medium = 0.2783
    elif year == "2022":
        wp_medium = 0.2783
    elif year == "2022EE":
        wp_medium = 0.2783
    elif year == "2024":
        wp_medium = 0.2783


    else:
        raise ValueError(f"Unknown year {year}")


    electrons = events.Electron

    good_electrons = (
        (electrons.pt > ele_pt_cut) &
        (np.abs(electrons.eta) < 2.5) &  # keep within tracker acceptance
        ~((np.abs(electrons.eta) > 1.44) & (np.abs(electrons.eta) < 1.57)) # remove transition
        # &(electrons.mvaIso_WP80)    # tight MVA ID
        # &(electrons.pfRelIso03_all < 0.15)  # isolation cut
    )

    good_muons = (
        (events.Muon.pt > mu_pt_cut)
        & (np.abs(events.Muon.eta) < 2.4)
        # & (events.Muon.pfRelIso03_all < 0.15)
    )

    one_ele = ak.num(events.Electron[good_electrons]) == 1
    one_mu = ak.num(events.Muon[good_muons]) == 1
    lepton_channel_mask = one_ele | one_mu

    selected_electrons = events.Electron[good_electrons]
    print("selected_electrons", len(selected_electrons[ak.num(selected_electrons.pt)>0]))
    selected_muons = events.Muon[good_muons]
    print("selected_muons", len(selected_muons[ak.num(selected_muons.pt)>0]))
    selected_leptons = ak.concatenate([selected_electrons, selected_muons], axis=1)
    print("selected_leptons", len(selected_leptons[ak.num(selected_leptons.pt)>0]))
    print("selected Electrons", len(selected_leptons[ak.num(selected_leptons[abs(selected_leptons.pdgId)==11])>0]))
    print("selected Muons", len(selected_leptons[ak.num(selected_leptons[abs(selected_leptons.pdgId)==13])>0]))

    # ------------------------
    # Jet selection
    # ------------------------
    good_jets = (
        (events.Jet.pt > 20)
        & (np.abs(events.Jet.eta) < 2.4)
        # & (events.Jet.btagDeepFlavB > wp_medium)
        # & (events.Jet.btagCSVV2 > wp_medium)
    )
    selected_bjets = events.Jet[good_jets] 
    print("selected_b_jets: ", selected_bjets)
    at_least_two_bjets = ak.num(selected_bjets) >= 2

    abs_eta = np.abs(photons.eta)

    # Barrel–endcap transition exclusion (1.442 ≤ |η| ≤ 1.566)
    valid_eta = (abs_eta <= 2.5) & ~((abs_eta >= 1.442) & (abs_eta <= 1.566))

    # Barrel vs Endcap ID cuts
    is_barrel = abs_eta < 1.442
    is_endcap = (abs_eta > 1.566) & (abs_eta < 2.5)

    # Apply region-specific MVA thresholds
    barrel_cut = is_barrel & (photons.mvaID > -0.02)
    endcap_cut = is_endcap & (photons.mvaID > -0.26)

    # Combine everything
    good_photons = (
        (photons.pt > 10)
        & valid_eta
        # & (barrel_cut | endcap_cut)
        # & (~photons.pixelSeed)
    )
    selected_photons = photons[good_photons]
    # at_least_two_photons = ak.num(selected_photons) >= 2

    # mask_npho = ak.num(selected_photons) >= 2
    # selected_photons = selected_photons[mask_npho]

    # # sort by pT (descending)
    # selected_photons = selected_photons[
    #     ak.argsort(selected_photons.pt, axis=1, ascending=False)
    # ]

    # # leading / subleading pT cuts
    # mask_pho_pt = (
    #     (selected_photons.pt[:, 0] > 20.0) &
    #     (selected_photons.pt[:, 1] > 15.0)
    # )

    # selected_photons = selected_photons[mask_pho_pt]

    # at_least_two_photons = ak.num(selected_photons) >= 2

    sorted_photons = selected_photons[
        ak.argsort(selected_photons.pt, axis=1, ascending=False)
    ]

    leading_pt = ak.firsts(sorted_photons.pt)
    subleading_pt = ak.firsts(sorted_photons.pt[:, 1:])

    at_least_two_photons = (
        (leading_pt > 20.0) &
        (subleading_pt > 15.0)
    )

    # dr = delta_r_manual(selected_leptons, selected_photons)
    # dr_mask = ak.all(ak.all(dr > 0.4, axis=-1), axis=-1)

    # # ΔR between electrons and photons
    # dr_electrons = delta_r_manual(selected_electrons, selected_photons)

    # # ΔR between muons and photons
    # dr_muons = delta_r_manual(selected_muons, selected_photons)


    # event_mask = lepton_channel_mask & at_least_two_bjets & at_least_two_photons & dr_mask
    event_mask = lepton_channel_mask & at_least_two_bjets & at_least_two_photons

    empty_photons = ak.Array([[]] * len(events))
    empty_bjets = ak.Array([[]] * len(events))
    empty_electrons = ak.Array([[]] * len(events))
    empty_muons = ak.Array([[]] * len(events))

    filtered_photons = ak.where(event_mask, selected_photons, empty_photons)
    filtered_jets = ak.where(event_mask, selected_bjets, empty_bjets)
    filtered_electrons = ak.where(event_mask, selected_electrons, empty_electrons)
    filtered_muons = ak.where(event_mask, selected_muons, empty_muons)

    # events_sel = events[event_mask]

    # filtered_photons   = events_sel.Photon[good_photons[event_mask]]
    # filtered_bjets     = events_sel.Jet[good_jets[event_mask]]
    # filtered_electrons = events_sel.Electron[good_electrons[event_mask]]
    # filtered_muons     = events_sel.Muon[good_muons[event_mask]]


    return filtered_photons, filtered_jets, filtered_electrons, filtered_muons