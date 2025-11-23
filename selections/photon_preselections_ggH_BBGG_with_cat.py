import awkward as ak
import numpy as np

def delta_r_manual(obj1, obj2):
    deta = obj1.eta[:, None] - obj2.eta
    dphi = np.abs(obj1.phi[:, None] - obj2.phi)
    dphi = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return np.sqrt(deta**2 + dphi**2)


def photon_preselections_ggH_BBGG_with_cat(self,
    photons: ak.Array,
    events: ak.Array,
    electron_veto=True,
    revert_electron_veto=False,
    year="2022",
    IsFlag=False):

    print("Number of events before preselection:", len(events))
    print("Year:", year)

    #-------------------------
    # b-tagging working point
    #-------------------------
    if year in ["2022", "2022EE"]:
        wp_medium = 0.2783
    else:
        raise ValueError(f"Unknown year {year}")

    # ------------------------
    # Jet selection
    # ------------------------
    good_jets_mask = (
        (events.Jet.pt > -1.0)
        & (np.abs(events.Jet.eta) < 2.4)
    )

    good_jets = events.Jet[good_jets_mask]

    jet_mask = ak.num(good_jets) > 0

    # DeepFlavB medium WP b-jets
    deepflav_mask = good_jets_mask & (events.Jet.btagDeepFlavB > wp_medium)
    deepflav_bjets = events.Jet[deepflav_mask]

    # UParTAK4B b-jets
    upar_mask = good_jets_mask & (events.Jet.btagUParTAK4probbb > 0.38)
    upar_bjets = events.Jet[upar_mask]

    # -------------------------------
    # Orthogonal Category Definitions
    # -------------------------------
    cat1 = ak.num(deepflav_bjets) >= 2

    cat2_base = ak.num(upar_bjets) >= 1
    cat2 = cat2_base & (~cat1)

    cat3_base = ak.num(good_jets) == 1
    cat3 = cat3_base & (~cat1) & (~cat2)

    # ------------------------
    # Photon selection
    # ------------------------
    abs_eta = np.abs(photons.eta)
    valid_eta = (abs_eta <= 2.5) & ~((abs_eta >= 1.442) & (abs_eta <= 1.566))

    is_barrel = abs_eta < 1.442
    is_endcap = (abs_eta > 1.566) & (abs_eta < 2.5)

    barrel_cut = is_barrel & (photons.mvaID > -0.02)
    endcap_cut = is_endcap & (photons.mvaID > -0.26)

    good_photons_mask = (
        (photons.pt > 10)
        & valid_eta
        & (barrel_cut | endcap_cut)
        & (~photons.pixelSeed)
    )

    selected_photons = photons[good_photons_mask]
    selected_photons = selected_photons[ak.argsort(selected_photons.pt, ascending=False)]

    # lead + sublead logic
    has_two_photons = ak.num(selected_photons) >= 2
    lead_pt = ak.fill_none(ak.firsts(selected_photons.pt), -999)
    sublead_pt = ak.fill_none(ak.pad_none(selected_photons.pt, 2)[:, 1], -999)

    lead_cut = lead_pt > 30
    sublead_cut = sublead_pt > 18

    photon_event_mask = has_two_photons & lead_cut & sublead_cut
    # photon_event_mask = has_two_photons & lead_cut

    # -----------------------------
    # Final event masks per category
    # -----------------------------
    mask1 = cat1 & photon_event_mask
    mask2 = cat2 & photon_event_mask
    mask3 = cat3 & photon_event_mask

    # mask1 = jet_mask & photon_event_mask
    # mask2 = jet_mask & photon_event_mask
    # mask3 = jet_mask & photon_event_mask

    # empty arrays used for masking
    empty = ak.Array([[]] * len(events))

    photons_cat1 = ak.where(mask1, selected_photons, empty)
    photons_cat2 = ak.where(mask2, selected_photons, empty)
    photons_cat3 = ak.where(mask3, selected_photons, empty)

    jets_cat1 = ak.where(mask1, deepflav_bjets, empty)
    jets_cat2 = ak.where(mask2, upar_bjets, empty)
    jets_cat3 = ak.where(mask3, good_jets, empty)

    # jets_cat1 = ak.where(mask1, good_jets, empty)
    # jets_cat2 = ak.where(mask2, good_jets, empty)
    # jets_cat3 = ak.where(mask3, good_jets, empty)

    print(f"Cat1 events (≥2 DeepFlavB bjets): {ak.sum(mask1)}")
    print(f"Cat2 events (≥1 btagUParTAK4probbb bjets, excl Cat1): {ak.sum(mask2)}")
    print(f"Cat3 events (exactly 1 good jet, excl Cat1/2): {ak.sum(mask3)}")

    return {
        "cat1": (photons_cat1, jets_cat1),
        "cat2": (photons_cat2, jets_cat2),
        "cat3": (photons_cat3, jets_cat3),
    }
