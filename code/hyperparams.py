from instructions import *

hyperparams = {
    "AdviceHelpfullnessDataLoader": {
        "trunc_bool": True,
        "inst": instructions["AdviceHelpfullnessDataLoader"][2],
        "include_tags": False,
        "grad_accum": 4,
        "lr": 2e-5,
        "epoch": 17
    },
    "AnswerGrammaticalityDataLoader": {
        "trunc_bool": True,
        "inst": instructions["AnswerGrammaticalityDataLoader"][2],
        "include_tags": True,
        "grad_accum": 8,
        "lr": 2e-5,
        "epoch": 15
    },
    "AnswerValidityDataLoader": {
        "trunc_bool": True,
        "inst": instructions["AnswerValidityDataLoader"][2],
        "include_tags": True,
        "grad_accum": 32,
        "lr": 2e-4,
        "epoch": 10
    },
    "QuestionAnswerabilityDataLoader": {
        "trunc_bool": True,
        "inst": instructions["QuestionAnswerabilityDataLoader"][2],
        "include_tags": False,
        "grad_accum": 4,
        "lr": 2e-4,
        "epoch": 20
    },
    "CommonsenseReasoningDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CommonsenseReasoningDataLoader"][1],
        "include_tags": False,
        "grad_accum": 16,
        "lr": 2e-4,
        "epoch": 18
    },
    "BestCounterNarrativeDataLoader": {
        "trunc_bool": True,
        "inst": instructions["BestCounterNarrativeDataLoader"][1],
        "include_tags": True,
        "grad_accum": 4,
        "lr": 3e-4,
        "epoch": 3
    },
    "CHOCounterNarrativeDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CHOCounterNarrativeDataLoader"][1],
        "include_tags": True,
        "grad_accum": 32,
        "lr": 2e-4,
        "epoch": 15
    },
    "CounterNarrativeGrammaticalityDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterNarrativeGrammaticalityDataLoader"][0],
        "include_tags": True,
        "grad_accum": 32,
        "lr": 2e-4,
        "epoch": 12
    },
    "CounterNarrativeSpecificityDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterNarrativeSpecificityDataLoader"][2],
        "include_tags": False,
        "grad_accum": 8,
        "lr": 2e-4,
        "epoch": 12
    },
    "CounterNarrativeSuitabilityDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterNarrativeSuitabilityDataLoader"][1],
        "include_tags": True,
        "grad_accum": 128,
        "lr": 3e-4,
        "epoch": 13
    },
    "CounterNarrativeInformativenessDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterNarrativeInformativenessDataLoader"][2],
        "include_tags": False,
        "grad_accum": 32,
        "lr": 2e-4,
        "epoch": 15
    },
    "CounterNarrativeOffensivenessDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterNarrativeOffensivenessDataLoader"][1],
        "include_tags": True,
        "grad_accum": 64,
        "lr": 3e-4,
        "epoch": 5
    },
    "CounterNarrativeStanceDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterNarrativeStanceDataLoader"][2],
        "include_tags": False,
        "grad_accum": 4,
        "lr": 2e-5,
        "epoch": 13
    },
    "HateSpeechOffensivenessDataLoader": {
        "trunc_bool": True,
        "inst": instructions["HateSpeechOffensivenessDataLoader"][0],
        "include_tags": True,
        "grad_accum": 128,
        "lr": 3e-4,
        "epoch": 2
    },
    "CounterfactualStoryRewritingCounterfactualDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterfactualStoryRewritingCounterfactualDataLoader"][1],
        "include_tags": True,
        "grad_accum": 64,
        "lr": 2e-4,
        "epoch": 7
    },
    "CounterfactualStoryRewritingEndingDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterfactualStoryRewritingEndingDataLoader"][0],
        "include_tags": True,
        "grad_accum": 16,
        "lr": 3e-4,
        "epoch": 2
    },
    "CounterfactualStoryRewritingPlotDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterfactualStoryRewritingPlotDataLoader"][3],
        "include_tags": True,
        "grad_accum": 32,
        "lr": 2e-4,
        "epoch": 15
    },
    "CounterfactualStoryRewritingPremiseDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterfactualStoryRewritingPremiseDataLoader"][2],
        "include_tags": False,
        "grad_accum": 8,
        "lr": 3e-4,
        "epoch": 4
    },
    "CounterfactualStoryRewritingSecondDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterfactualStoryRewritingSecondDataLoader"][2],
        "include_tags": False,
        "grad_accum": 8,
        "lr": 2e-5,
        "epoch": 18
    },
    "DefeasibleInferenceAttenuatorEffectivenessDataLoader": {
        "trunc_bool": True,
        "inst": instructions["DefeasibleInferenceAttenuatorEffectivenessDataLoader"][0],
        "include_tags": False,
        "grad_accum": 32,
        "lr": 2e-4,
        "epoch": 19
    },
    "DefeasibleInferenceIntensifierEffectivenessDataLoader": {
        "trunc_bool": True,
        "inst": instructions["DefeasibleInferenceIntensifierEffectivenessDataLoader"][1],
        "include_tags": False,
        "grad_accum": 128,
        "lr": 3e-4,
        "epoch": 15
    },
    "HellaSwagDataLoader": {
        "trunc_bool": True,
        "inst": instructions["HellaSwagDataLoader"][2],
        "include_tags": False,
        "grad_accum": 64,
        "lr": 2e-5,
        "epoch": 17
    },

    "multitask_AdviceHelpfullnessDataLoader": {
        "trunc_bool": True,
        "inst": instructions["AdviceHelpfullnessDataLoader"][2],
        "include_tags": False,
        "grad_accum": 8,
        "lr": 2e-5,
        "epoch": 1
    },
    "multitask_AnswerGrammaticalityDataLoader": {
        "trunc_bool": True,
        "inst": instructions["AnswerGrammaticalityDataLoader"][2],
        "include_tags": True,
        "grad_accum": 8,
        "lr": 2e-5,
        "epoch": 8
    },
    "multitask_AnswerValidityDataLoader": {
        "trunc_bool": True,
        "inst": instructions["AnswerValidityDataLoader"][2],
        "include_tags": True,
        "grad_accum": 8,
        "lr": 2e-5,
        "epoch": 18
    },
    "multitask_QuestionAnswerabilityDataLoader": {
        "trunc_bool": True,
        "inst": instructions["QuestionAnswerabilityDataLoader"][2],
        "include_tags": False,
        "grad_accum": 16,
        "lr": 2e-5,
        "epoch": 11
    },
    "multitask_CommonsenseReasoningDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CommonsenseReasoningDataLoader"][1],
        "include_tags": False,
        "grad_accum": 32,
        "lr": 2e-5,
        "epoch": 15
    },
    "multitask_BestCounterNarrativeDataLoader": {
        "trunc_bool": True,
        "inst": instructions["BestCounterNarrativeDataLoader"][1],
        "include_tags": True,
        "grad_accum": 8,
        "lr": 2e-5,
        "epoch": 12
    },
    "multitask_CHOCounterNarrativeDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CHOCounterNarrativeDataLoader"][1],
        "include_tags": True,
        "grad_accum": 16,
        "lr": 2e-5,
        "epoch": 14
    },
    "multitask_CounterNarrativeGrammaticalityDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterNarrativeGrammaticalityDataLoader"][0],
        "include_tags": True,
        "grad_accum": 8,
        "lr": 2e-5,
        "epoch": 6
    },
    "multitask_CounterNarrativeSpecificityDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterNarrativeSpecificityDataLoader"][2],
        "include_tags": False,
        "grad_accum": 32,
        "lr": 2e-5,
        "epoch": 12
    },
    "multitask_CounterNarrativeSuitabilityDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterNarrativeSuitabilityDataLoader"][1],
        "include_tags": True,
        "grad_accum": 32,
        "lr": 2e-5,
        "epoch": 8
    },
    "multitask_CounterNarrativeInformativenessDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterNarrativeInformativenessDataLoader"][2],
        "include_tags": False,
        "grad_accum": 4,
        "lr": 2e-5,
        "epoch": 19
    },
    "multitask_CounterNarrativeOffensivenessDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterNarrativeOffensivenessDataLoader"][1],
        "include_tags": True,
        "grad_accum": 4,
        "lr": 2e-5,
        "epoch": 19
    },
    "multitask_CounterNarrativeStanceDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterNarrativeStanceDataLoader"][2],
        "include_tags": False,
        "grad_accum": 32,
        "lr": 2e-5,
        "epoch": 10
    },
    "multitask_HateSpeechOffensivenessDataLoader": {
        "trunc_bool": True,
        "inst": instructions["HateSpeechOffensivenessDataLoader"][0],
        "include_tags": True,
        "grad_accum": 4,
        "lr": 2e-5,
        "epoch": 1
    },
    "multitask_CounterfactualStoryRewritingCounterfactualDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterfactualStoryRewritingCounterfactualDataLoader"][1],
        "include_tags": True,
        "grad_accum": 8,
        "lr": 2e-5,
        "epoch": 20
    },
    "multitask_CounterfactualStoryRewritingEndingDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterfactualStoryRewritingEndingDataLoader"][0],
        "include_tags": True,
        "grad_accum": 8,
        "lr": 2e-4,
        "epoch": 3
    },
    "multitask_CounterfactualStoryRewritingPlotDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterfactualStoryRewritingPlotDataLoader"][3],
        "include_tags": True,
        "grad_accum": 16,
        "lr": 2e-5,
        "epoch": 14
    },
    "multitask_CounterfactualStoryRewritingPremiseDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterfactualStoryRewritingPremiseDataLoader"][2],
        "include_tags": False,
        "grad_accum": 32,
        "lr": 2e-5,
        "epoch": 3
    },
    "multitask_CounterfactualStoryRewritingSecondDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterfactualStoryRewritingSecondDataLoader"][2],
        "include_tags": False,
        "grad_accum": 8,
        "lr": 2e-5,
        "epoch": 16
    },
    "multitask_DefeasibleInferenceAttenuatorEffectivenessDataLoader": {
        "trunc_bool": True,
        "inst": instructions["DefeasibleInferenceAttenuatorEffectivenessDataLoader"][0],
        "include_tags": False,
        "grad_accum": 32,
        "lr": 2e-5,
        "epoch": 17
    },
    "multitask_DefeasibleInferenceIntensifierEffectivenessDataLoader": {
        "trunc_bool": True,
        "inst": instructions["DefeasibleInferenceIntensifierEffectivenessDataLoader"][1],
        "include_tags": False,
        "grad_accum": 32,
        "lr": 2e-5,
        "epoch": 16
    },
    "multitask_HellaSwagDataLoader": {
        "trunc_bool": True,
        "inst": instructions["HellaSwagDataLoader"][2],
        "include_tags": False,
        "grad_accum": 32,
        "lr": 2e-5,
        "epoch": 19
    },
    "multitask_overall": {
        "grad_accum": 8,
        "lr": 2e-5,
        "epoch": 16
    },
}