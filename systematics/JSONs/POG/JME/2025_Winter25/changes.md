## Changes: 2025-10-27 (Winter25Prompt25 JERC and JVM JSONs)

Merge Request: [!1](https://gitlab.cern.ch/cms-analysis-corrections/JME/Run3-25Prompt-Winter25-NanoAODv15/-/merge_requests/1)

In this MR we provide the 2025 JSON files for AK4 and AK8 (PUPPI) jets, as well as the JSON file for jet veto map application. These cover the eras 2025 C-F.

In more detail:

- The MC truth JECs (`L1FastJet`, `L2Relative`, `L3Absolute`) are based on the `Winter25Prompt25_V2_MC` tag. They were presented in [JERC, May 13](https://indico.cern.ch/event/1545816/#36-l2rel-for-winter25) for AK4PFPuppi jets and in [JERC, Sep. 02](https://indico.cern.ch/event/1580247/#42-ak8-puppi-ak4-chs-mctruth-w) for AK8PFPuppi jets.
- The residual JECs (`L2L3Residual`) are based on the `Winter25Prompt25_V2_DATA` tag and are run-dependent, in the same format as 2024 and 2023 JECs. They were presented in [JERC, Sep. 16](https://indico.cern.ch/event/1586754/#sc-43-1-semi-virtual-combinati). Dedicated L2L3Residual JECs are derived for eras 2025C, 2025D, and 2025E. For 2025F, JECs are cloned from 2025E, since they were found to give an acceptable closure. Dedicated L2L3Residual JECs for eras 2025F and 2025G will be provided in a future release.
- The JEC uncertainties are cloned from `Summer24`, and now have the `Winter25Prompt25_V2_MC` tag. 
- The JER SFs and PtResolution are cloned from `Summer23BPixPrompt23_RunD_JRV1_MC`.
- The Jet Veto Map (JVM) was presented in [JERC, Sep. 16](https://indico.cern.ch/event/1586754/#sc-43-1-semi-virtual-combinati). As usual, the `jetvetomap` histogram is recommended for analyzers.

The related PRs in JECDatabase are: [#209](https://github.com/cms-jet/JECDatabase/pull/209), [#211](https://github.com/cms-jet/JECDatabase/pull/211)
