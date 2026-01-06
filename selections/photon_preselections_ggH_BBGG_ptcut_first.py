import awkward as ak
import numpy as np

def delta_r_manual(obj1, obj2):
    deta = obj1.eta[:, None] - obj2.eta
    dphi = np.abs(obj1.phi[:, None] - obj2.phi)
    dphi = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return np.sqrt(deta**2 + dphi**2)


# def photon_preselections_ggH_BBGG_ptcut_first(
#     self,
#     photons,
#     events,
#     electron_veto=True,
#     revert_electron_veto=False,
#     year="2022",
#     IsFlag=False
# ):
#     """
#     AWKWARD VERSION IDENTICAL TO ROOT CUT LOGIC
#     """

#     print("Number of events before preselection:", len(events))
#     print("Year:", year)

#     # ----------------------------------------------------
#     # 1. b-tagging working point
#     # ----------------------------------------------------
#     if year in ["2022", "2022EE"]:
#         wp_medium = 0.2783
#     else:
#         raise ValueError(f"Unknown year {year}")

#     # ----------------------------------------------------
#     # 2. Gen b-quarks from A → bb
#     # ----------------------------------------------------
#     gen = events.GenPart
#     bquark_mask = (abs(gen.pdgId) == 5)
#     bquarks = gen[bquark_mask]

#     mother_idx = bquarks.genPartIdxMother
#     mothers = gen[mother_idx]
#     from_a_mask = (mothers.pdgId == 35)

#     bquarks_from_a = bquarks[from_a_mask]

#     # ----------------------------------------------------
#     # 3. Jet selection
#     # ----------------------------------------------------
#     good_jets = (
#         (events.Jet.pt > 20.0)
#         & (np.abs(events.Jet.eta) < 2.4)
#         & (events.Jet.btagDeepFlavB > wp_medium)
#     )

#     selected_bjets = events.Jet[good_jets]
#     at_least_two_bjets = ak.num(selected_bjets) >= 2

#     # ----------------------------------------------------
#     # 4. ROOT-like PHOTON SELECTION
#     # *** IMPORTANT: DO NOT pre-filter photons ***
#     # ----------------------------------------------------

#     # 4.1 Sort ALL photons by descending pT (ROOT: Argsort + Reverse)
#     sorted_idx = ak.argsort(photons.pt, ascending=False)
#     pho_sorted = photons[sorted_idx]

#     # 4.2 Ensure safe indexing (pad to ≥2 photons)
#     pho_padded = ak.pad_none(pho_sorted, 2)

#     iL = pho_padded[:, 0]   # lead
#     iS = pho_padded[:, 1]   # sublead

#     # 4.3 Require ≥2 photons (ROOT: nPhoton >= 2)
#     has_two_photons = ak.num(pho_sorted) >= 2

#     # ----------------------------------------------------
#     # 4.4 pT cuts (ROOT)
#     # ----------------------------------------------------
#     pt_cut = (iL.pt > 30) & (iS.pt > 18)

#     # ----------------------------------------------------
#     # 4.5 eta acceptance (ROOT EXACT)
#     # |η| < 1.442 OR (1.556 < |η| < 2.5)
#     # ----------------------------------------------------
#     abs_eta_L = np.abs(iL.eta)
#     abs_eta_S = np.abs(iS.eta)

#     eta_cut_L = ((abs_eta_L < 1.442) |
#                  ((abs_eta_L > 1.556) & (abs_eta_L < 2.5)))
#     eta_cut_S = ((abs_eta_S < 1.442) |
#                  ((abs_eta_S > 1.556) & (abs_eta_S < 2.5)))

#     # ----------------------------------------------------
#     # 4.6 MVAID cuts (same ternary logic as ROOT)
#     # ----------------------------------------------------
#     mva_cut_L = ak.where(abs_eta_L < 1.442,
#                          iL.mvaID > -0.02,
#                          iL.mvaID > -0.26)

#     mva_cut_S = ak.where(abs_eta_S < 1.442,
#                          iS.mvaID > -0.02,
#                          iS.mvaID > -0.26)

#     # ----------------------------------------------------
#     # 4.7 Pixel seed veto (ROOT: == 0)
#     # ----------------------------------------------------
#     pixel_cut = (iL.pixelSeed == 0) & (iS.pixelSeed == 0)

#     # ----------------------------------------------------
#     # 4.8 Final PHOTON event mask identical to ROOT
#     # ----------------------------------------------------
#     photon_event_mask = (
#         has_two_photons &
#         pt_cut &
#         eta_cut_L & eta_cut_S &
#         mva_cut_L & mva_cut_S &
#         pixel_cut
#     )

#     # ----------------------------------------------------
#     # 5. Combine jet & photon criteria
#     # ----------------------------------------------------
#     event_mask = at_least_two_bjets & photon_event_mask

#     print("Events passing full preselection:", ak.sum(event_mask))

#     # ----------------------------------------------------
#     # 6. Return filtered objects just like ROOT
#     # ----------------------------------------------------
#     empty_photons = ak.Array([[]] * len(events))
#     empty_jets = ak.Array([[]] * len(events))
#     empty_bquarks = ak.Array([[]] * len(events))

#     filtered_photons = ak.where(event_mask, pho_sorted, empty_photons)
#     filtered_jets = ak.where(event_mask, selected_bjets, empty_jets)
#     filtered_bquarks = ak.where(event_mask, bquarks_from_a, empty_bquarks)

#     # Keep only bquarks that exist
#     filtered_bquarks = filtered_bquarks[ak.num(filtered_bquarks.pt) > 0]

#     return filtered_photons, filtered_jets, filtered_bquarks





def photon_preselections_ggH_BBGG_ptcut_first(
    self,
    photons: ak.Array,
    events: ak.Array,
    electron_veto=True,
    revert_electron_veto=False,
    year="2022",
    IsFlag=False):
    """
    HiggsDNA-style, ROOT-compatible photon preselection.
    Lead/sublead cuts are applied BEFORE the good_photons ID cuts.
    Output always contains exactly 0 or 2 photons per event.
    """

    print("Number of events before preselection:", len(events))
    print("Year:", year)

    #-------------------------
    # b-tagging working point
    #-------------------------
    if year in ["2022", "2022EE", "2024"]:
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

    # =====================================================
    # Sort ALL photons — no cuts yet
    # =====================================================
    sorted_idx = ak.argsort(photons.pt, ascending=False)
    pho_sorted = photons[sorted_idx]

    # pad so iL and iS always exist (as None)
    pho_padded = ak.pad_none(pho_sorted, 2)

    iL = pho_padded[:, 0]
    iS = pho_padded[:, 1]

    # =====================================================
    # Lead/Sublead pt cuts (ROOT)
    # =====================================================
    lead_pt     = ak.fill_none(iL.pt, -999)
    sublead_pt  = ak.fill_none(iS.pt, -999)

    has_two_photons = ak.num(pho_sorted) >= 2
    lead_cut     = lead_pt > 30
    sublead_cut  = sublead_pt > 18

    pt_mask = has_two_photons & lead_cut & sublead_cut

    # =====================================================
    # Now apply ID/eta/pixel cuts to ONLY iL and iS
    # =====================================================
    abs_eta_L = np.abs(ak.fill_none(iL.eta, 0))
    abs_eta_S = np.abs(ak.fill_none(iS.eta, 0))

    # eta acceptance
    eta_cut_L = ((abs_eta_L < 1.442) |
                 ((abs_eta_L > 1.556) & (abs_eta_L < 2.5)))
    eta_cut_S = ((abs_eta_S < 1.442) |
                 ((abs_eta_S > 1.556) & (abs_eta_S < 2.5)))

    # MVAID
    mva_L = ak.fill_none(iL.mvaID, -999)
    mva_S = ak.fill_none(iS.mvaID, -999)

    mva_cut_L = ak.where(abs_eta_L < 1.442, mva_L > -0.02, mva_L > -0.26)
    mva_cut_S = ak.where(abs_eta_S < 1.442, mva_S > -0.02, mva_S > -0.26)

    # pixel seed veto
    pix_L = ak.fill_none(iL.pixelSeed, 1)
    pix_S = ak.fill_none(iS.pixelSeed, 1)

    pixel_cut = (pix_L == 0) & (pix_S == 0)

    # Master photon event mask
    photon_event_mask = pt_mask & eta_cut_L & eta_cut_S & mva_cut_L & mva_cut_S & pixel_cut

    # =====================================================
    # Combine jet + photon criteria
    # =====================================================
    event_mask = at_least_two_bjets & photon_event_mask

    print("Events passing full preselection:", ak.sum(event_mask))

    # =====================================================
    # FINAL OUTPUT — EXACTLY 2 photons or []
    # =====================================================
    empty_photons = ak.Array([[]] * len(events))
    empty_jets    = ak.Array([[]] * len(events))
    empty_bquarks = ak.Array([[]] * len(events))

    # only keep first two photons
    two_photons = pho_sorted[:, :2]

    filtered_photons = ak.where(event_mask, two_photons, empty_photons)
    filtered_jets    = ak.where(event_mask, selected_bjets, empty_jets)
    gen_bquark       = ak.where(event_mask, bquarks_from_a, empty_bquarks)

    # keep only events where b-quarks exist
    gen_bquark = gen_bquark[ak.num(gen_bquark.pt) > 0]

    return filtered_photons, filtered_jets, gen_bquark


