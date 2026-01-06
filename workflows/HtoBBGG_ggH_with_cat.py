from higgs_dna.workflows.base import HggBaseProcessor
from higgs_dna.tools.chained_quantile import ChainedQuantileRegression
from higgs_dna.tools.SC_eta import add_photon_SC_eta
from higgs_dna.tools.EELeak_region import veto_EEleak_flag
from higgs_dna.tools.EcalBadCalibCrystal_events import remove_EcalBadCalibCrystal_events
from higgs_dna.tools.gen_helpers import get_fiducial_flag, get_genJets, get_higgs_gen_attributes
from higgs_dna.selections.photon_preselections_ggH_BBGG_with_cat_ptcut_first import photon_preselections_ggH_BBGG_with_cat
from higgs_dna.selections.lepton_selections import select_electrons, select_muons
from higgs_dna.selections.jet_selections import select_jets, jetvetomap
from higgs_dna.selections.lumi_selections import select_lumis
from higgs_dna.utils.dumping_utils import (
    diphoton_ak_array,
    dump_ak_array,
    diphoton_list_to_pandas,
    dump_pandas,
    get_obj_syst_dict,
)
from higgs_dna.utils.misc_utils import choose_jet

from higgs_dna.systematics import object_systematics as available_object_systematics
from higgs_dna.systematics import object_corrections as available_object_corrections
from higgs_dna.systematics import weight_systematics as available_weight_systematics
from higgs_dna.systematics import weight_corrections as available_weight_corrections

import functools
import operator
import os
import warnings
from typing import Any, Dict, List, Optional
import awkward
import numpy
import sys
import vector
from coffea import processor
from coffea.analysis_tools import Weights
from copy import deepcopy

import logging

logger = logging.getLogger(__name__)

vector.register_awkward()


class HtoBBGG_ggH_with_cat(HggBaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def choose_nth_object_variable(self, variable, n, fill_value):
        """
        this helper function is used to create flat collection from a jagged collection,
        parameters:
        * variable: (awkward array) selected variable from the object
        * n: (int) nth object to be selected
        * fill_value: (float) value with which to fill the padded none.
        """
        variable = variable[
            awkward.local_index(variable) == n
        ]
        variable = awkward.pad_none(
            variable, 1
        )
        variable = awkward.flatten(
            awkward.fill_none(variable, fill_value)
        )
        return variable

    def process(self, events: awkward.Array) -> Dict[Any, Any]:
        dataset_name = events.metadata["dataset"]

        # data or monte carlo?
        self.data_kind = "mc" if hasattr(events, "GenPart") else "data"

        n_events = len(events)
        print(f"Dataset {dataset_name}: {n_events} events")

        # here we start recording possible coffea accumulators
        # most likely histograms, could be counters, arrays, ...
        histos_etc = {}
        histos_etc[dataset_name] = {}
        if self.data_kind == "mc":
            histos_etc[dataset_name]["nTot"] = int(
                awkward.num(events.genWeight, axis=0)
            )
            histos_etc[dataset_name]["nPos"] = int(awkward.sum(events.genWeight > 0))
            histos_etc[dataset_name]["nNeg"] = int(awkward.sum(events.genWeight < 0))
            histos_etc[dataset_name]["nEff"] = int(
                histos_etc[dataset_name]["nPos"] - histos_etc[dataset_name]["nNeg"]
            )
            histos_etc[dataset_name]["genWeightSum"] = float(
                awkward.sum(events.genWeight)
            )
        else:
            histos_etc[dataset_name]["nTot"] = int(len(events))
            histos_etc[dataset_name]["nPos"] = int(histos_etc[dataset_name]["nTot"])
            histos_etc[dataset_name]["nNeg"] = int(0)
            histos_etc[dataset_name]["nEff"] = int(histos_etc[dataset_name]["nTot"])
            histos_etc[dataset_name]["genWeightSum"] = float(len(events))

        # lumi mask
        if self.data_kind == "data":
            try:
                lumimask = select_lumis(self.year[dataset_name][0], events, logger)
                events = events[lumimask]
            except:
                logger.info(
                    f"[ lumimask ] Skip now! Unable to find year info of {dataset_name}"
                )

        metadata = {}

        if self.data_kind == "mc":
            # Add sum of gen weights before selection for normalisation in postprocessing
            metadata["sum_genw_presel"] = str(awkward.sum(events.genWeight))
        else:
            metadata["sum_genw_presel"] = "Data"

        # apply filters and triggers
        events = self.apply_filters_and_triggers(events)
        n_events_2 = len(events)
        print(f"after trigger: Dataset {dataset_name}: {n_events_2} events")

        # remove events affected by EcalBadCalibCrystal
        if self.data_kind == "data":
            events = remove_EcalBadCalibCrystal_events(events)

        try:
            correction_names = self.corrections[dataset_name]
        except KeyError:
            correction_names = []
        try:
            systematic_names = self.systematics[dataset_name]
        except KeyError:
            systematic_names = []

        for correction_name in correction_names:
            if correction_name in available_object_corrections.keys():
                logger.info(
                    f"Applying correction {correction_name} to dataset {dataset_name}"
                )
                varying_function = available_object_corrections[correction_name]
                events = varying_function(
                    events=events, year=self.year[dataset_name][0]
                )
            elif correction_name in available_weight_corrections:
                # event weight corrections will be applied after photon preselection / application of further taggers
                continue
            else:
                # may want to throw an error instead, needs to be discussed
                warnings.warn(f"Could not process correction {correction_name}.")
                continue

        #-----------------------------------------------------Store the gen information-------------------------------------------------------------
        if self.data_kind == "mc":

            gen = events.GenPart
            Gen_photons = gen[(gen.pdgId == 22) & (gen.status == 1)]
            mother_idx = Gen_photons.genPartIdxMother
            from_a_mask = gen[mother_idx].pdgId == 35
            photons_from_a = Gen_photons[from_a_mask]

            sorted_gen_photons = photons_from_a[awkward.argsort(photons_from_a.pt, axis=1, ascending=False)]

            leading_gen_photon_pt = self.choose_nth_object_variable(sorted_gen_photons.pt, 0, -999.0)
            subleading_gen_photon_pt = self.choose_nth_object_variable(sorted_gen_photons.pt, 1, -999.0)
            leading_gen_photon_eta = self.choose_nth_object_variable(sorted_gen_photons.eta, 0, -999.0)
            subleading_gen_photon_eta = self.choose_nth_object_variable(sorted_gen_photons.eta, 1, -999.0)
            leading_gen_photon_phi = self.choose_nth_object_variable(sorted_gen_photons.phi, 0, -999.0)
            subleading_gen_photon_phi = self.choose_nth_object_variable(sorted_gen_photons.phi, 1, -999.0)


        original_photons = events.Photon
        nPhotons = len(events.Photon)
        print("Number of photons: ", nPhotons)
        # NOTE: jet jerc systematics are added in the correction functions and handled later
        original_jets = events.Jet

        # systematic object variations
        for systematic_name in systematic_names:
            if systematic_name in available_object_systematics.keys():
                systematic_dct = available_object_systematics[systematic_name]
                if systematic_dct["object"] == "Photon":
                    logger.info(
                        f"Adding systematic {systematic_name} to photons collection of dataset {dataset_name}"
                    )
                    original_photons.add_systematic(
                        # passing the arguments here explicitly since I want to pass the events to the varying function. If there is a more elegant / flexible way, just change it!
                        name=systematic_name,
                        kind=systematic_dct["args"]["kind"],
                        what=systematic_dct["args"]["what"],
                        varying_function=functools.partial(
                            systematic_dct["args"]["varying_function"],
                            events=events,
                            year=self.year[dataset_name][0],
                        )
                        # name=systematic_name, **systematic_dct["args"]
                    )
                # to be implemented for other objects here
            elif systematic_name in available_weight_systematics:
                # event weight systematics will be applied after photon preselection / application of further taggers
                continue
            else:
                # may want to throw an error instead, needs to be discussed
                warnings.warn(
                    f"Could not process systematic variation {systematic_name}."
                )
                continue

        # Applying systematic variations
        photons_dct = {}
        photons_dct["nominal"] = original_photons
        logger.debug(original_photons.systematics.fields)
        for systematic in original_photons.systematics.fields:
            for variation in original_photons.systematics[systematic].fields:
                # deepcopy to allow for independent calculations on photon variables with CQR
                photons_dct[f"{systematic}_{variation}"] = deepcopy(
                    original_photons.systematics[systematic][variation]
                )

        # NOTE: jet jerc systematics are added in the corrections, now extract those variations and create the dictionary
        jerc_syst_list, jets_dct = get_obj_syst_dict(original_jets, ["pt", "mass"])
        # object systematics dictionary
        logger.debug(f"[ jerc systematics ] {jerc_syst_list}")

        # Build the flattened array of all possible variations
        variations_combined = []
        variations_combined.append(original_photons.systematics.fields)
        # NOTE: jet jerc systematics are not added with add_systematics
        variations_combined.append(jerc_syst_list)
        # Flatten
        variations_flattened = sum(variations_combined, [])  # Begin with empty list and keep concatenating
        # Attach _down and _up
        variations = [item + suffix for item in variations_flattened for suffix in ['_down', '_up']]
        # Add nominal to the list
        variations.append('nominal')
        logger.debug(f"[systematics variations] {variations}")

        for variation in variations:
            photons, jets = photons_dct["nominal"], events.Jet
            if variation == "nominal":
                pass  # Do nothing since we already get the unvaried, but nominally corrected objets above
            elif variation in [*photons_dct]:  # [*dict] gets the keys of the dict since Python >= 3.5
                photons = photons_dct[variation]
            elif variation in [*jets_dct]:
                jets = jets_dct[variation]
            do_variation = variation  # We can also simplify this a bit but for now it works

            # photon preselection
            Result = photon_preselections_ggH_BBGG_with_cat(self, photons, events, year=self.year[dataset_name][0])

            pho1, bjet1 = Result['cat1']
            pho2, bjet2 = Result['cat2']
            pho3, bjet3 = Result['cat3']

            # pho1 = photons
            # pho2 = photons
            # pho3 = photons
            # bjet1 = jets
            # bjet2 = jets
            # bjet3 = jets

            # leps = awkward.concatenate([events.Electron, events.Muon], axis=1)
            # photons, b_jets, leps = photons, jets, leps
            Nphotons = awkward.num(photons.pt)

            # sort photons in each event descending in pt
            # make descending-pt combinations of photons
            photons = pho1[awkward.argsort(pho1.pt, ascending=False)]
            photons["charge"] = awkward.zeros_like(
                photons.pt
            )  # added this because charge is not a property of photons in nanoAOD v11. We just assume every photon has charge zero...
            diphotons = awkward.combinations(
                photons, 2, fields=["pho_lead", "pho_sublead"]
            )
            print(diphotons.pho_lead.pt)

            # Store b-Jets information             
            # b_jets = b_jets[awkward.argsort(b_jets.pt, ascending=False)]
            b_jets = bjet1[awkward.argsort(bjet1.btagDeepFlavB, ascending=False)][:, :2]
            b_jets = b_jets[awkward.argsort(b_jets.pt, ascending=False)]
            b_jets["charge"] = awkward.zeros_like(
                b_jets.pt
            )  # added this because charge is not a property of photons in nanoAOD v11. We just assume every photon has charge zero...
            diJets = awkward.combinations(
                b_jets, 2, fields=["bjet_lead", "bjet_sublead"]
            )

            bbgg = awkward.cartesian({"gg": diphotons, "bb": diJets})
            bbgg = awkward.zip({
                "pho_lead": bbgg.gg.pho_lead,
                "pho_sublead": bbgg.gg.pho_sublead,
                "bjet_lead": bbgg.bb.bjet_lead,
                "bjet_sublead": bbgg.bb.bjet_sublead,
            })

            bjet2_sorted = bjet2[awkward.argsort(bjet2.btagUParTAK4probbb, ascending=False)]

            # Pick only the best (highest UParTAK4B) jet
            best_bjet2 = awkward.firsts(bjet2_sorted)

            # Build diphotons from Cat2 photons
            photons2 = pho2[awkward.argsort(pho2.pt, ascending=False)]
            diphotons2 = awkward.combinations(
                photons2, 2, fields=["pho_lead", "pho_sublead"]
            )

            # Create BBG structure
            bgg2 = awkward.zip({
                "pho_lead": diphotons2.pho_lead,
                "pho_sublead": diphotons2.pho_sublead,
                "bjet_best": best_bjet2,
            })

            # Build diphotons from Cat2 photons
            photons3 = pho3[awkward.argsort(pho3.pt, ascending=False)]
            diphotons3 = awkward.combinations(
                photons3, 2, fields=["pho_lead", "pho_sublead"]
            )

            bjet3_sorted = bjet3[awkward.argsort(bjet3.btagDeepFlavB, ascending=False)]

            # Pick only the best (highest UParTAK4B) jet
            best_bjet3 = awkward.firsts(bjet3_sorted)

            # Create BBG structure
            bgg3 = awkward.zip({
                "pho_lead": diphotons3.pho_lead,
                "pho_sublead": diphotons3.pho_sublead,
                "bjet": best_bjet3,
            })

            NDiphotons = len(diphotons)
            print(f"Dataset {dataset_name}: {NDiphotons} Diphotons")
            # now turn the diphotons into candidates with four momenta and such
            diphoton_4mom = bbgg["pho_lead"] + bbgg["pho_sublead"]
            dijet_4mom = bbgg["bjet_lead"] + bbgg["bjet_sublead"]
            bbgg_4mom = bbgg["pho_lead"] + bbgg["pho_sublead"] + bbgg["bjet_lead"] + bbgg["bjet_sublead"]
            bbgg["dipho_pt"] = diphoton_4mom.pt
            # bbgg["pt"] = diphoton_4mom.pt
            bbgg["dipho_eta"] = diphoton_4mom.eta
            bbgg["dipho_phi"] = diphoton_4mom.phi
            bbgg["dipho_mass"] = diphoton_4mom.mass
            bbgg["dipho_charge"] = diphoton_4mom.charge
            bbgg["dijet_pt"] = dijet_4mom.pt
            bbgg["dijet_eta"] = dijet_4mom.eta
            bbgg["dijet_phi"] = dijet_4mom.phi
            bbgg["dijet_mass"] = dijet_4mom.mass
            bbgg["mass"] = bbgg_4mom.mass
            bbgg["Nphotons"] = Nphotons

            diphoton_4mom2 = bgg2["pho_lead"] + bgg2["pho_sublead"]
            bgg2["dipho_pt"] = diphoton_4mom2.pt
            # bbgg["pt"] = diphoton_4mom.pt
            bgg2["dipho_eta"] = diphoton_4mom2.eta
            bgg2["dipho_phi"] = diphoton_4mom2.phi
            bgg2["dipho_mass"] = diphoton_4mom2.mass
            bgg2["dipho_charge"] = diphoton_4mom2.charge
            bgg2["Nphotons"] = Nphotons

            diphoton_4mom3 = bgg3["pho_lead"] + bgg3["pho_sublead"]
            bgg3["dipho_pt"] = diphoton_4mom3.pt
            # bbgg["pt"] = diphoton_4mom.pt
            bgg3["dipho_eta"] = diphoton_4mom3.eta
            bgg3["dipho_phi"] = diphoton_4mom3.phi
            bgg3["dipho_mass"] = diphoton_4mom3.mass
            bgg3["dipho_charge"] = diphoton_4mom3.charge
            bgg3["Nphotons"] = Nphotons

            diphotons = awkward.with_name(diphotons, "PtEtaPhiMCandidate")

            NDiphotons_2 = len(diphotons)
            print(f"Pos2: Dataset {dataset_name}: {NDiphotons_2} Diphotons")


            # diphotons = diphotons[fid_det_passed]
            NDiphotons_3 = len(diphotons)
            print(f"Pos3: Dataset {dataset_name}: {NDiphotons_3} Diphotons")

            NDiphotons_4 = len(diphotons)
            print(f"Pos4: Dataset {dataset_name}: {NDiphotons_4} Diphotons")
            # workflow specific processing
            events, process_extra = self.process_extra(events)
            histos_etc.update(process_extra)

            jets = awkward.with_name(jets, "PtEtaPhiMCandidate")

            electrons = awkward.zip(
                {
                    "pt": events.Electron.pt,
                    "eta": events.Electron.eta,
                    "phi": events.Electron.phi,
                    "mass": events.Electron.mass,
                    "charge": events.Electron.charge,
                    "cutBased": events.Electron.cutBased,
                    "mvaIso_WP90": events.Electron.mvaIso_WP90,
                    "mvaIso_WP80": events.Electron.mvaIso_WP80,
                }
            )
            electrons = awkward.with_name(electrons, "PtEtaPhiMCandidate")

            # Special cut for base workflow to replicate iso cut for electrons also for muons
            events['Muon'] = events.Muon[events.Muon.pfRelIso03_all < 0.2]

            muons = events['Muon']
            muons = awkward.with_name(muons, "PtEtaPhiMCandidate")

#-------------------------------------------------------------------------------------------------------------------------------------
            for tagger in self.taggers:
                (
                    bbgg["_".join([tagger.name, str(tagger.priority)])],
                    tagger_extra,
                ) = tagger(
                    events, bbgg
                )  # creates new column in bbgg - tagger priority, or 0, also return list of histrograms here?
                histos_etc.update(tagger_extra)

            # if there are taggers to run, arbitrate by them first
            # Deal with order of tagger priorities
            # Turn from diphoton jagged array to whether or not an event was selected
            if len(self.taggers):
                counts = awkward.num(bbgg.pt, axis=1)
                flat_tags = numpy.stack(
                    (
                        awkward.flatten(
                            bbgg[
                                "_".join([tagger.name, str(tagger.priority)])
                            ]
                        )
                        for tagger in self.taggers
                    ),
                    axis=1,
                )
                tags = awkward.from_regular(
                    awkward.unflatten(flat_tags, counts), axis=2
                )
                winner = awkward.min(tags[tags != 0], axis=2)
                bbgg["best_tag"] = winner

                # lowest priority is most important (ascending sort)
                # leave in order of diphoton pT in case of ties (stable sort)
                sorted = awkward.argsort(bbgg.best_tag, stable=True)
                bbgg = bbgg[sorted]

            bbgg = awkward.firsts(bbgg)
            # set bbgg as part of the event record
            events[f"bbgg_{do_variation}"] = bbgg
            # annotate bbgg with event information
            bbgg["event"] = events.event
            bbgg["lumi"] = events.luminosityBlock
            bbgg["run"] = events.run
            # nPV just for validation of pileup reweighting
            bbgg["nPV"] = events.PV.npvs
            bbgg["fixedGridRhoAll"] = events.Rho.fixedGridRhoAll
            # annotate bbgg with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
            if self.data_kind == "mc":
                bbgg["genWeight"] = events.genWeight
                bbgg["dZ"] = events.GenVtx.z - events.PV.z
                # Necessary for differential xsec measurements in final fits ("truth" variables)
                bbgg["HTXS_Higgs_pt"] = events.HTXS.Higgs_pt
                bbgg["HTXS_Higgs_y"] = events.HTXS.Higgs_y
                bbgg["HTXS_njets30"] = events.HTXS.njets30  # Need to clarify if this variable is suitable, does it fulfill abs(eta_j) < 2.5? Probably not
                # Preparation for HTXS measurements later, start with stage 0 to disentangle VH into WH and ZH for final fits
                bbgg["HTXS_stage_0"] = events.HTXS.stage_0
            # Fill zeros for data because there is no GenVtx for data, obviously
            else:
                bbgg["dZ"] = awkward.zeros_like(events.PV.z)

            # drop events without a preselected diphoton candidate
            # drop events without a tag, if there are tags
            if len(self.taggers):
                selection_mask = ~(
                    awkward.is_none(bbgg)
                    | awkward.is_none(bbgg.best_tag)
                )
                bbgg = bbgg[selection_mask]
            else:
                selection_mask = ~awkward.is_none(bbgg)
                bbgg = bbgg[selection_mask]

            # return if there is no surviving events
            if len(bbgg) == 0:
                logger.debug("No surviving events in this run, return now!")
                return histos_etc
            if self.data_kind == "mc":
                # initiate Weight container here, after selection, since event selection cannot easily be applied to weight container afterwards
                event_weights = Weights(size=len(events[selection_mask]))

                # corrections to event weights:
                for correction_name in correction_names:
                    if correction_name in available_weight_corrections:
                        logger.info(
                            f"Adding correction {correction_name} to weight collection of dataset {dataset_name}"
                        )
                        varying_function = available_weight_corrections[
                            correction_name
                        ]
                        event_weights = varying_function(
                            events=events[selection_mask],
                            photons=events[f"bbgg_{do_variation}"][
                                selection_mask
                            ],
                            weights=event_weights,
                            dataset_name=dataset_name,
                            year=self.year[dataset_name][0],
                        )

                # systematic variations of event weights go to nominal output dataframe:
                if do_variation == "nominal":
                    for systematic_name in systematic_names:
                        if systematic_name in available_weight_systematics:
                            logger.info(
                                f"Adding systematic {systematic_name} to weight collection of dataset {dataset_name}"
                            )
                            if systematic_name == "LHEScale":
                                if hasattr(events, "LHEScaleWeight"):
                                    bbgg["nweight_LHEScale"] = awkward.num(
                                        events.LHEScaleWeight[selection_mask],
                                        axis=1,
                                    )
                                    bbgg[
                                        "weight_LHEScale"
                                    ] = events.LHEScaleWeight[selection_mask]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            elif systematic_name == "LHEPdf":
                                if hasattr(events, "LHEPdfWeight"):
                                    # two AlphaS weights are removed
                                    bbgg["nweight_LHEPdf"] = (
                                        awkward.num(
                                            events.LHEPdfWeight[selection_mask],
                                            axis=1,
                                        )
                                        - 2
                                    )
                                    bbgg[
                                        "weight_LHEPdf"
                                    ] = events.LHEPdfWeight[selection_mask][
                                        :, :-2
                                    ]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            else:
                                varying_function = available_weight_systematics[
                                    systematic_name
                                ]
                                event_weights = varying_function(
                                    events=events[selection_mask],
                                    photons=events[f"bbgg_{do_variation}"][
                                        selection_mask
                                    ],
                                    weights=event_weights,
                                    dataset_name=dataset_name,
                                    year=self.year[dataset_name][0],
                                )

                bbgg["weight_central"] = event_weights.weight()
                # Store variations with respect to central weight
                if do_variation == "nominal":
                    if len(event_weights.variations):
                        logger.info(
                            "Adding systematic weight variations to nominal output file."
                        )
                    for modifier in event_weights.variations:
                        bbgg["weight_" + modifier] = event_weights.weight(
                            modifier=modifier
                        )

                # Multiply weight by genWeight for normalisation in post-processing chain
                event_weights._weight = (
                    events["genWeight"][selection_mask]
                    * bbgg["weight_central"]
                )
                bbgg["weight"] = event_weights.weight()

            # Add weight variables (=1) for data for consistent datasets
            else:
                bbgg["weight_central"] = awkward.ones_like(
                    bbgg["event"]
                )
                bbgg["weight"] = awkward.ones_like(bbgg["event"])
#-------------------------------------------------------------------------------------------------------------------------------------

            for tagger in self.taggers:
                (
                    bgg2["_".join([tagger.name, str(tagger.priority)])],
                    tagger_extra,
                ) = tagger(
                    events, bgg2
                )  # creates new column in bgg2 - tagger priority, or 0, also return list of histrograms here?
                histos_etc.update(tagger_extra)

            # if there are taggers to run, arbitrate by them first
            # Deal with order of tagger priorities
            # Turn from diphoton jagged array to whether or not an event was selected
            if len(self.taggers):
                counts = awkward.num(bgg2.pt, axis=1)
                flat_tags = numpy.stack(
                    (
                        awkward.flatten(
                            bgg2[
                                "_".join([tagger.name, str(tagger.priority)])
                            ]
                        )
                        for tagger in self.taggers
                    ),
                    axis=1,
                )
                tags = awkward.from_regular(
                    awkward.unflatten(flat_tags, counts), axis=2
                )
                winner = awkward.min(tags[tags != 0], axis=2)
                bgg2["best_tag"] = winner

                # lowest priority is most important (ascending sort)
                # leave in order of diphoton pT in case of ties (stable sort)
                sorted = awkward.argsort(bgg2.best_tag, stable=True)
                bgg2 = bgg2[sorted]

            bgg2 = awkward.firsts(bgg2)
            # set bgg2 as part of the event record
            events[f"bgg2_{do_variation}"] = bgg2
            # annotate bgg2 with event information
            bgg2["event"] = events.event
            bgg2["lumi"] = events.luminosityBlock
            bgg2["run"] = events.run
            # nPV just for validation of pileup reweighting
            bgg2["nPV"] = events.PV.npvs
            bgg2["fixedGridRhoAll"] = events.Rho.fixedGridRhoAll
            # annotate bgg2 with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
            if self.data_kind == "mc":
                bgg2["genWeight"] = events.genWeight
                bgg2["dZ"] = events.GenVtx.z - events.PV.z
                # Necessary for differential xsec measurements in final fits ("truth" variables)
                bgg2["HTXS_Higgs_pt"] = events.HTXS.Higgs_pt
                bgg2["HTXS_Higgs_y"] = events.HTXS.Higgs_y
                bgg2["HTXS_njets30"] = events.HTXS.njets30  # Need to clarify if this variable is suitable, does it fulfill abs(eta_j) < 2.5? Probably not
                # Preparation for HTXS measurements later, start with stage 0 to disentangle VH into WH and ZH for final fits
                bgg2["HTXS_stage_0"] = events.HTXS.stage_0
            # Fill zeros for data because there is no GenVtx for data, obviously
            else:
                bgg2["dZ"] = awkward.zeros_like(events.PV.z)

            # drop events without a preselected diphoton candidate
            # drop events without a tag, if there are tags
            if len(self.taggers):
                selection_mask = ~(
                    awkward.is_none(bgg2)
                    | awkward.is_none(bgg2.best_tag)
                )
                bgg2 = bgg2[selection_mask]
            else:
                selection_mask = ~awkward.is_none(bgg2)
                bgg2 = bgg2[selection_mask]

            # return if there is no surviving events
            if len(bgg2) == 0:
                logger.debug("No surviving events in this run, return now!")
                return histos_etc
            if self.data_kind == "mc":
                # initiate Weight container here, after selection, since event selection cannot easily be applied to weight container afterwards
                event_weights = Weights(size=len(events[selection_mask]))

                # corrections to event weights:
                for correction_name in correction_names:
                    if correction_name in available_weight_corrections:
                        logger.info(
                            f"Adding correction {correction_name} to weight collection of dataset {dataset_name}"
                        )
                        varying_function = available_weight_corrections[
                            correction_name
                        ]
                        event_weights = varying_function(
                            events=events[selection_mask],
                            photons=events[f"bgg2_{do_variation}"][
                                selection_mask
                            ],
                            weights=event_weights,
                            dataset_name=dataset_name,
                            year=self.year[dataset_name][0],
                        )

                # systematic variations of event weights go to nominal output dataframe:
                if do_variation == "nominal":
                    for systematic_name in systematic_names:
                        if systematic_name in available_weight_systematics:
                            logger.info(
                                f"Adding systematic {systematic_name} to weight collection of dataset {dataset_name}"
                            )
                            if systematic_name == "LHEScale":
                                if hasattr(events, "LHEScaleWeight"):
                                    bgg2["nweight_LHEScale"] = awkward.num(
                                        events.LHEScaleWeight[selection_mask],
                                        axis=1,
                                    )
                                    bgg2[
                                        "weight_LHEScale"
                                    ] = events.LHEScaleWeight[selection_mask]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            elif systematic_name == "LHEPdf":
                                if hasattr(events, "LHEPdfWeight"):
                                    # two AlphaS weights are removed
                                    bgg2["nweight_LHEPdf"] = (
                                        awkward.num(
                                            events.LHEPdfWeight[selection_mask],
                                            axis=1,
                                        )
                                        - 2
                                    )
                                    bgg2[
                                        "weight_LHEPdf"
                                    ] = events.LHEPdfWeight[selection_mask][
                                        :, :-2
                                    ]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            else:
                                varying_function = available_weight_systematics[
                                    systematic_name
                                ]
                                event_weights = varying_function(
                                    events=events[selection_mask],
                                    photons=events[f"bgg2_{do_variation}"][
                                        selection_mask
                                    ],
                                    weights=event_weights,
                                    dataset_name=dataset_name,
                                    year=self.year[dataset_name][0],
                                )

                bgg2["weight_central"] = event_weights.weight()
                # Store variations with respect to central weight
                if do_variation == "nominal":
                    if len(event_weights.variations):
                        logger.info(
                            "Adding systematic weight variations to nominal output file."
                        )
                    for modifier in event_weights.variations:
                        bgg2["weight_" + modifier] = event_weights.weight(
                            modifier=modifier
                        )

                # Multiply weight by genWeight for normalisation in post-processing chain
                event_weights._weight = (
                    events["genWeight"][selection_mask]
                    * bgg2["weight_central"]
                )
                bgg2["weight"] = event_weights.weight()

            # Add weight variables (=1) for data for consistent datasets
            else:
                bgg2["weight_central"] = awkward.ones_like(
                    bgg2["event"]
                )
                bgg2["weight"] = awkward.ones_like(bgg2["event"])
#-------------------------------------------------------------------------------------------------------------------------------------

            for tagger in self.taggers:
                (
                    bgg3["_".join([tagger.name, str(tagger.priority)])],
                    tagger_extra,
                ) = tagger(
                    events, bgg3
                )  # creates new column in bgg3 - tagger priority, or 0, also return list of histrograms here?
                histos_etc.update(tagger_extra)

            # if there are taggers to run, arbitrate by them first
            # Deal with order of tagger priorities
            # Turn from diphoton jagged array to whether or not an event was selected
            if len(self.taggers):
                counts = awkward.num(bgg3.pt, axis=1)
                flat_tags = numpy.stack(
                    (
                        awkward.flatten(
                            bgg3[
                                "_".join([tagger.name, str(tagger.priority)])
                            ]
                        )
                        for tagger in self.taggers
                    ),
                    axis=1,
                )
                tags = awkward.from_regular(
                    awkward.unflatten(flat_tags, counts), axis=2
                )
                winner = awkward.min(tags[tags != 0], axis=2)
                bgg3["best_tag"] = winner

                # lowest priority is most important (ascending sort)
                # leave in order of diphoton pT in case of ties (stable sort)
                sorted = awkward.argsort(bgg3.best_tag, stable=True)
                bgg3 = bgg3[sorted]

            bgg3 = awkward.firsts(bgg3)
            # set bgg3 as part of the event record
            events[f"bgg3_{do_variation}"] = bgg3
            # annotate bgg3 with event information
            bgg3["event"] = events.event
            bgg3["lumi"] = events.luminosityBlock
            bgg3["run"] = events.run
            # nPV just for validation of pileup reweighting
            bgg3["nPV"] = events.PV.npvs
            bgg3["fixedGridRhoAll"] = events.Rho.fixedGridRhoAll
            # annotate bgg3 with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
            if self.data_kind == "mc":
                bgg3["genWeight"] = events.genWeight
                bgg3["dZ"] = events.GenVtx.z - events.PV.z
                # Necessary for differential xsec measurements in final fits ("truth" variables)
                bgg3["HTXS_Higgs_pt"] = events.HTXS.Higgs_pt
                bgg3["HTXS_Higgs_y"] = events.HTXS.Higgs_y
                bgg3["HTXS_njets30"] = events.HTXS.njets30  # Need to clarify if this variable is suitable, does it fulfill abs(eta_j) < 2.5? Probably not
                # Preparation for HTXS measurements later, start with stage 0 to disentangle VH into WH and ZH for final fits
                bgg3["HTXS_stage_0"] = events.HTXS.stage_0
            # Fill zeros for data because there is no GenVtx for data, obviously
            else:
                bgg3["dZ"] = awkward.zeros_like(events.PV.z)

            # drop events without a preselected diphoton candidate
            # drop events without a tag, if there are tags
            if len(self.taggers):
                selection_mask = ~(
                    awkward.is_none(bgg3)
                    | awkward.is_none(bgg3.best_tag)
                )
                bgg3 = bgg3[selection_mask]
            else:
                selection_mask = ~awkward.is_none(bgg3)
                bgg3 = bgg3[selection_mask]

            # return if there is no surviving events
            if len(bgg3) == 0:
                logger.debug("No surviving events in this run, return now!")
                return histos_etc
            if self.data_kind == "mc":
                # initiate Weight container here, after selection, since event selection cannot easily be applied to weight container afterwards
                event_weights = Weights(size=len(events[selection_mask]))

                # corrections to event weights:
                for correction_name in correction_names:
                    if correction_name in available_weight_corrections:
                        logger.info(
                            f"Adding correction {correction_name} to weight collection of dataset {dataset_name}"
                        )
                        varying_function = available_weight_corrections[
                            correction_name
                        ]
                        event_weights = varying_function(
                            events=events[selection_mask],
                            photons=events[f"bgg3_{do_variation}"][
                                selection_mask
                            ],
                            weights=event_weights,
                            dataset_name=dataset_name,
                            year=self.year[dataset_name][0],
                        )

                # systematic variations of event weights go to nominal output dataframe:
                if do_variation == "nominal":
                    for systematic_name in systematic_names:
                        if systematic_name in available_weight_systematics:
                            logger.info(
                                f"Adding systematic {systematic_name} to weight collection of dataset {dataset_name}"
                            )
                            if systematic_name == "LHEScale":
                                if hasattr(events, "LHEScaleWeight"):
                                    bgg3["nweight_LHEScale"] = awkward.num(
                                        events.LHEScaleWeight[selection_mask],
                                        axis=1,
                                    )
                                    bgg3[
                                        "weight_LHEScale"
                                    ] = events.LHEScaleWeight[selection_mask]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            elif systematic_name == "LHEPdf":
                                if hasattr(events, "LHEPdfWeight"):
                                    # two AlphaS weights are removed
                                    bgg3["nweight_LHEPdf"] = (
                                        awkward.num(
                                            events.LHEPdfWeight[selection_mask],
                                            axis=1,
                                        )
                                        - 2
                                    )
                                    bgg3[
                                        "weight_LHEPdf"
                                    ] = events.LHEPdfWeight[selection_mask][
                                        :, :-2
                                    ]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            else:
                                varying_function = available_weight_systematics[
                                    systematic_name
                                ]
                                event_weights = varying_function(
                                    events=events[selection_mask],
                                    photons=events[f"bgg3_{do_variation}"][
                                        selection_mask
                                    ],
                                    weights=event_weights,
                                    dataset_name=dataset_name,
                                    year=self.year[dataset_name][0],
                                )

                bgg3["weight_central"] = event_weights.weight()
                # Store variations with respect to central weight
                if do_variation == "nominal":
                    if len(event_weights.variations):
                        logger.info(
                            "Adding systematic weight variations to nominal output file."
                        )
                    for modifier in event_weights.variations:
                        bgg3["weight_" + modifier] = event_weights.weight(
                            modifier=modifier
                        )

                # Multiply weight by genWeight for normalisation in post-processing chain
                event_weights._weight = (
                    events["genWeight"][selection_mask]
                    * bgg3["weight_central"]
                )
                bgg3["weight"] = event_weights.weight()

            # Add weight variables (=1) for data for consistent datasets
            else:
                bgg3["weight_central"] = awkward.ones_like(
                    bgg3["event"]
                )
                bgg3["weight"] = awkward.ones_like(bgg3["event"])

#------------------------------------store bbgg objects info--------------------------------------------------------------------------

        if self.output_location is not None:
            if self.output_format == "root":
                df_bbgg = diphoton_list_to_pandas(self, bbgg)  # same converter works fine
            else:
                akarr_bbgg = diphoton_ak_array(self, bbgg)
                akarr_bbgg = akarr_bbgg[[f for f in akarr_bbgg.fields if "lead_fixedGridRhoAll" not in f]]

            # keep original filename (no suffix)
            fname_bbgg = (
                events.behavior["__events_factory__"]._partition_key.replace("/", "_")
                + ".%s" % self.output_format
            )

            # create subdirectory structure
            subdirs_bbgg = []
            if "dataset" in events.metadata:
                subdirs_bbgg.append(events.metadata["dataset"])
            subdirs_bbgg.append(do_variation)
            # subdirs_bbgg.append("bbgg")   # store dijet files here

            if self.output_format == "root":
                dump_pandas(self, df_bbgg, fname_bbgg, self.output_location, subdirs_bbgg)
            else:
                dump_ak_array(self, akarr_bbgg, fname_bbgg, self.output_location, metadata, subdirs_bbgg)

        

        if self.output_location is not None:
            if self.output_format == "root":
                df_bgg2 = diphoton_list_to_pandas(self, bgg2)  # same converter works fine
            else:
                akarr_bgg2 = diphoton_ak_array(self, bgg2)
                akarr_bgg2 = akarr_bgg2[[f for f in akarr_bgg2.fields if "lead_fixedGridRhoAll" not in f]]

            # keep original filename (no suffix)
            fname_bgg2 = (
                events.behavior["__events_factory__"]._partition_key.replace("/", "_")
                + ".%s" % self.output_format
            )

            # create subdirectory structure
            subdirs_bgg2 = []
            if "dataset" in events.metadata:
                subdirs_bgg2.append(events.metadata["dataset"])
            subdirs_bgg2.append(do_variation)
            subdirs_bgg2.append("bgg2")
            # subdirs_bgg2.append("bgg2")   # store dijet files here

            if self.output_format == "root":
                dump_pandas(self, df_bgg2, fname_bgg2, self.output_location, subdirs_bgg2)
            else:
                dump_ak_array(self, akarr_bgg2, fname_bgg2, self.output_location, metadata, subdirs_bgg2)


        if self.output_location is not None:
            if self.output_format == "root":
                df_bgg3 = diphoton_list_to_pandas(self, bgg3)  # same converter works fine
            else:
                akarr_bgg3 = diphoton_ak_array(self, bgg3)
                akarr_bgg3 = akarr_bgg3[[f for f in akarr_bgg3.fields if "lead_fixedGridRhoAll" not in f]]

            # keep original filename (no suffix)
            fname_bgg3 = (
                events.behavior["__events_factory__"]._partition_key.replace("/", "_")
                + ".%s" % self.output_format
            )

            # create subdirectory structure
            subdirs_bgg3 = []
            if "dataset" in events.metadata:
                subdirs_bgg3.append(events.metadata["dataset"])
            subdirs_bgg3.append(do_variation)
            subdirs_bgg3.append("bgg3")
            # subdirs_bgg3.append("bgg3")   # store dijet files here

            if self.output_format == "root":
                dump_pandas(self, df_bgg3, fname_bgg3, self.output_location, subdirs_bgg3)
            else:
                dump_ak_array(self, akarr_bgg3, fname_bgg3, self.output_location, metadata, subdirs_bgg3)

        return histos_etc

    def process_extra(self, events: awkward.Array) -> awkward.Array:
        return events, {}

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass