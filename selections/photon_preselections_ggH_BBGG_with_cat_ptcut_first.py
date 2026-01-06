import awkward as ak
import numpy as np

def delta_r_manual(obj1, obj2):
    deta = obj1.eta[:, None] - obj2.eta
    dphi = np.abs(obj1.phi[:, None] - obj2.phi)
    dphi = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return np.sqrt(deta**2 + dphi**2)


def photon_preselections_ggH_BBGG_with_cat(
    self,
    photons: ak.Array,
    events: ak.Array,
    electron_veto=True,
    revert_electron_veto=False,
    year="2022",
    IsFlag=False):

    print("Number of events before preselection:", len(events))
    print("Year:", year)

    # -------------------------
    # b-tagging working point
    # -------------------------
    if year in ["2022", "2022EE", "2024"]:
        wp_medium = 0.2783
    else:
        raise ValueError(f"Unknown year {year}")

    # ------------------------
    # Jet selection
    # ------------------------
    good_jets_mask = (
        (events.Jet.pt > 20.0)
        & (np.abs(events.Jet.eta) < 2.4)
    )
    good_jets = events.Jet[good_jets_mask]

    # b-tag working points
    deepflav_mask = good_jets_mask & (events.Jet.btagDeepFlavB > wp_medium)
    deepflav_bjets = events.Jet[deepflav_mask]

    upar_mask = good_jets_mask & (events.Jet.btagUParTAK4probbb > 0.38)
    upar_bjets = events.Jet[upar_mask]

    # -------------------------------
    # Orthogonal Categories
    # -------------------------------
    cat1 = ak.num(deepflav_bjets) >= 2
    cat2 = (ak.num(upar_bjets) >= 1) & (~cat1)
    cat3 = (ak.num(good_jets) == 1) & (~cat1) & (~cat2)

    # =====================================================
    # ROOT-style photon selection (NO pre-filtering!)
    # =====================================================

    # Sort photons
    sorted_idx = ak.argsort(photons.pt, ascending=False)
    pho_sorted = photons[sorted_idx]

    # Pad so iL/iS always exist
    pho_padded = ak.pad_none(pho_sorted, 2)
    iL = pho_padded[:, 0]
    iS = pho_padded[:, 1]

    # Fill None → safe values
    lead_pt = ak.fill_none(iL.pt, -999)
    sub_pt  = ak.fill_none(iS.pt, -999)

    lead_eta = ak.fill_none(iL.eta, 0)
    sub_eta  = ak.fill_none(iS.eta, 0)

    lead_mva = ak.fill_none(iL.mvaID, -999)
    sub_mva  = ak.fill_none(iS.mvaID, -999)

    lead_pix = ak.fill_none(iL.pixelSeed, 1)
    sub_pix  = ak.fill_none(iS.pixelSeed, 1)

    # photon multiplicity
    has_two_photons = ak.num(pho_sorted) >= 2

    # pT cuts
    pt_cut = (lead_pt > 30) & (sub_pt > 18)

    # eta acceptance
    abs_eta_L = np.abs(lead_eta)
    abs_eta_S = np.abs(sub_eta)

    eta_cut_L = ((abs_eta_L < 1.442) |
                 ((abs_eta_L > 1.556) & (abs_eta_L < 2.5)))
    eta_cut_S = ((abs_eta_S < 1.442) |
                 ((abs_eta_S > 1.556) & (abs_eta_S < 2.5)))

    # MVAID cuts
    mva_cut_L = ak.where(abs_eta_L < 1.442, lead_mva > -0.02, lead_mva > -0.26)
    mva_cut_S = ak.where(abs_eta_S < 1.442, sub_mva > -0.02, sub_mva > -0.26)

    # pixel seed
    pixel_cut = (lead_pix == 0) & (sub_pix == 0)

    # Final ROOT photon mask
    photon_event_mask = (
        has_two_photons &
        pt_cut &
        eta_cut_L & eta_cut_S &
        mva_cut_L & mva_cut_S &
        pixel_cut
    )

    # =====================================================
    # Final category masks including photon selection
    # =====================================================
    mask1 = cat1 & photon_event_mask
    mask2 = cat2 & photon_event_mask
    mask3 = cat3 & photon_event_mask

    print(f"Cat1 events: {ak.sum(mask1)}")
    print(f"Cat2 events: {ak.sum(mask2)}")
    print(f"Cat3 events: {ak.sum(mask3)}")

    # =====================================================
    # Final Output: EXACTLY TWO PHOTONS PER EVENT
    # =====================================================
    two_photons = pho_sorted[:, :2]   # lead/sublead only

    empty = ak.Array([[]] * len(events))

    photons_cat1 = ak.where(mask1, two_photons, empty)
    photons_cat2 = ak.where(mask2, two_photons, empty)
    photons_cat3 = ak.where(mask3, two_photons, empty)

    jets_cat1 = ak.where(mask1, deepflav_bjets, empty)
    jets_cat2 = ak.where(mask2, upar_bjets, empty)
    jets_cat3 = ak.where(mask3, good_jets, empty)

    return {
        "cat1": (photons_cat1, jets_cat1),
        "cat2": (photons_cat2, jets_cat2),
        "cat3": (photons_cat3, jets_cat3),
    }

