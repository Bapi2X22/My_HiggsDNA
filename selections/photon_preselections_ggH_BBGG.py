import awkward as ak
import numpy as np

def delta_r_manual(obj1, obj2):
    deta = obj1.eta[:, None] - obj2.eta
    dphi = np.abs(obj1.phi[:, None] - obj2.phi)
    dphi = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return np.sqrt(deta**2 + dphi**2)


def photon_preselections_ggH_BBGG(self,
    photons: ak.Array,
    events: ak.Array,
    electron_veto=True,
    revert_electron_veto=False,
    year="2018",
    IsFlag=False):
    """
    Apply full preselection on leptons, jets, and photons.
    Finally return only photons from events that pass all criteria.
    """

    print("Number of events before preselection:", len(events))
    print("Year:", year)

    #-------------------------
    # b-tagging working point
    #-------------------------
    if year == "2022":
        wp_medium = 0.2783
    elif year == "2022EE":
        wp_medium = 0.2783
    else:
        raise ValueError(f"Unknown year {year}")

    gen = events.GenPart

    bquarks = gen[abs(gen.pdgId) == 5]
    mother_idx = bquarks.genPartIdxMother
    from_a_mask = gen[mother_idx].pdgId == 35
    bquarks_from_a = bquarks[from_a_mask]

    # ------------------------
    # Jet selection
    # ------------------------
    good_jets = (
        (events.Jet.pt > 20.0)
        & (np.abs(events.Jet.eta) < 2.4)
        & (events.Jet.btagDeepFlavB > wp_medium)
    )
    selected_bjets = events.Jet[good_jets]
    at_least_two_bjets = ak.num(selected_bjets) >= 2

    # ------------------------
    # Photon selection
    # ------------------------
    abs_eta = np.abs(photons.eta)

    # Barrel–endcap transition exclusion (1.442 ≤ |η| ≤ 1.566)
    valid_eta = (abs_eta <= 2.5) & ~((abs_eta >= 1.442) & (abs_eta <= 1.566))

    # Barrel vs Endcap ID cuts
    is_barrel = abs_eta < 1.442
    is_endcap = (abs_eta > 1.566) & (abs_eta < 2.5)

    # Apply region-specific MVA thresholds
    barrel_cut = is_barrel & (photons.mvaID > -0.02)
    endcap_cut = is_endcap & (photons.mvaID > -0.26)

    # Base photon selection
    good_photons = (
        (photons.pt > 10)
        & valid_eta
        & (barrel_cut | endcap_cut)
        & (~photons.pixelSeed)
    )
    selected_photons = photons[good_photons]

    # Sort photons by pT in descending order
    selected_photons = selected_photons[ak.argsort(selected_photons.pt, ascending=False)]

    # ------------------------
    # Lead / Sublead cuts
    # ------------------------
    # Define masks for events that have at least 2 photons
    has_two_photons = ak.num(selected_photons) >= 2

    # Safe indexing: fill with -inf if photon is missing
    lead_pt = ak.fill_none(ak.firsts(selected_photons.pt), -999)
    sublead_pt = ak.fill_none(ak.pad_none(selected_photons.pt, 2)[:, 1], -999)

    lead_cut = lead_pt > 30
    sublead_cut = sublead_pt > 18

    # Event must have 2 photons and satisfy both pT cuts
    photon_event_mask = has_two_photons & lead_cut & sublead_cut
    # photon_event_mask = has_two_photons & lead_cut

    # ------------------------
    # Combine jet + photon criteria
    # ------------------------
    event_mask = at_least_two_bjets & photon_event_mask
    # event_mask = photon_event_mask

    # Prepare empty arrays for masked-out events
    empty_photons = ak.Array([[]] * len(events))
    empty_bjets = ak.Array([[]] * len(events))
    empty_bquarks = ak.Array([[]] * len(events))

    # Keep only events passing full selection
    filtered_photons = ak.where(event_mask, selected_photons, empty_photons)
    filtered_jets = ak.where(event_mask, selected_bjets, empty_bjets)
    gen_bquark = ak.where(event_mask, bquarks_from_a, empty_bquarks)

    gen_bquark = gen_bquark[ak.num(gen_bquark.pt)>0]

    print(f"Events passing full preselection: {ak.sum(event_mask)}")

    return filtered_photons, filtered_jets, gen_bquark
