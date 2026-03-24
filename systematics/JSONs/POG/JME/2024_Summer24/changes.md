## Changes: 2025-12-02 (Update to Summer24Prompt24_V2 JECs)

Merge Request: [!1](https://gitlab.cern.ch/cms-analysis-corrections/JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/-/merge_requests/1)

In this MR, we update the 2024 JEC version from `Summer24Prompt24_V1` to `Summer24Prompt24_V2`. 

The new `V2` release adds JECs for AK8 PUPPI jets, in addition to the existing AK4 PUPPI JECs. The MC truth JECs for AK8 PUPPI jets were presented in [JERC, Nov. 25](https://indico.cern.ch/event/1612934/#50-mc-truth-jecs-for-ak8-puppi). As usual, the L2L3Residuals, JER files, and uncertainties for AK8 PUPPI are inherited from the corresponding AK4 PUPPI files.

For AK4 PUPPI jets, the `Summer24Prompt24_V2` JECs are **unchanged** with respect to `Summer24Prompt24_V1`.

The corresponding PR in JECDatabase is: [#216](https://github.com/cms-jet/JECDatabase/pull/216)
