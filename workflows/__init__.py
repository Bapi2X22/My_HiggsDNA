from higgs_dna.workflows.base import HggBaseProcessor
from higgs_dna.workflows.fiducial import HggFiducialProcessor
from higgs_dna.workflows.dystudies import TagAndProbeProcessor
from higgs_dna.workflows.HHbbgg import HHbbggProcessor
from higgs_dna.workflows.particleLevel import ParticleLevelProcessor
from higgs_dna.workflows.top import TopProcessor
from higgs_dna.workflows.Zmmy import ZmmyProcessor, ZmmyHist, ZmmyZptHist
from higgs_dna.workflows.hpc_processor import HplusCharmProcessor
from higgs_dna.workflows.zee_processor import ZeeProcessor
from higgs_dna.workflows.lowmass import LowMassProcessor
from higgs_dna.workflows.btagging import BTaggingEfficienciesProcessor
from higgs_dna.workflows.stxs import STXSProcessor
from higgs_dna.workflows.diphoton_training import DiphoTrainingProcessor
from higgs_dna.workflows.My_first_processor import My_first_processor  # Example custom workflow
from higgs_dna.workflows.My_first_processor_beta import My_first_processor_beta
from higgs_dna.workflows.HtoBBGG_bkg import HtoBBGG_bkg
from higgs_dna.workflows.HtoBBGG_ggH import HtoBBGG_ggH
from higgs_dna.workflows.HtoBBGG_ggH_with_cat import HtoBBGG_ggH_with_cat
from higgs_dna.workflows.My_first_processor_2024_WH_cat1 import My_first_processor_2024_WH_cat1
from higgs_dna.workflows.My_first_processor_WH_Cat1 import My_first_processor_WH_Cat1
from higgs_dna.workflows.My_first_processor_WH_Cat1_HDNA_presel import My_first_processor_WH_Cat1_HDNA_presel
from higgs_dna.workflows.HtoAAtobbgg_BKG_2024 import HtoAAtobbgg_BKG_2024
from higgs_dna.workflows.HtoAAtobbgg_BKG_2024_without_presel import HtoAAtobbgg_BKG_2024_without_presel


from higgs_dna.workflows.taggers import taggers

workflows = {}

workflows["base"] = HggBaseProcessor
workflows["fiducial"] = HggFiducialProcessor
workflows["tagandprobe"] = TagAndProbeProcessor
workflows["HHbbgg"] = HHbbggProcessor
workflows["particleLevel"] = ParticleLevelProcessor
workflows["top"] = TopProcessor
workflows["zmmy"] = ZmmyProcessor
workflows["zmmyHist"] = ZmmyHist
workflows["zmmyZptHist"] = ZmmyZptHist
workflows["hpc"] = HplusCharmProcessor
workflows["zee"] = ZeeProcessor
workflows["lowmass"] = LowMassProcessor
workflows["BTagging"] = BTaggingEfficienciesProcessor
workflows["stxs"] = STXSProcessor
workflows["diphotonID"] = DiphoTrainingProcessor
workflows["My_first_processor"] = My_first_processor  # Example custom workflow
workflows["My_first_processor_beta"] = My_first_processor_beta
workflows["HtoBBGG_bkg"] = HtoBBGG_bkg
workflows["HtoBBGG_ggH"] = HtoBBGG_ggH
workflows["HtoBBGG_ggH_with_cat"] = HtoBBGG_ggH_with_cat
workflows["My_first_processor_2024_WH_cat1"] = My_first_processor_2024_WH_cat1
workflows["My_first_processor_WH_Cat1"] = My_first_processor_WH_Cat1
workflows["My_first_processor_WH_Cat1_HDNA_presel"] = My_first_processor_WH_Cat1_HDNA_presel
workflows["HtoAAtobbgg_BKG_2024"] = HtoAAtobbgg_BKG_2024
workflows["HtoAAtobbgg_BKG_2024_without_presel"] = HtoAAtobbgg_BKG_2024_without_presel


__all__ = ["workflows", "taggers"]
