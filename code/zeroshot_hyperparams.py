from instructions import *

zeroshot_hyperparams = {
    "AdviceHelpfullnessDataLoader": {
        "trunc_bool": True,
        "inst": instructions["AdviceHelpfullnessDataLoader"][2],
        "include_tags": False,
        "grad_accum": 4,
        "lr": 2e-5,
        "epoch": 16
    },
    "AnswerGrammaticalityDataLoader": {
        "trunc_bool": True,
        "inst": instructions["AnswerGrammaticalityDataLoader"][2],
        "include_tags": True,
        "grad_accum": 8,
        "lr": 2e-5,
        "epoch": 19
    },
    "AnswerValidityDataLoader": {
        "trunc_bool": True,
        "inst": instructions["AnswerValidityDataLoader"][2],
        "include_tags": True,
        "grad_accum": 16,
        "lr": 3e-4,
        "epoch": 17
    },
    "QuestionAnswerabilityDataLoader": {
        "trunc_bool": True,
        "inst": instructions["QuestionAnswerabilityDataLoader"][2],
        "include_tags": False,
        "grad_accum": 32,
        "lr": 2e-5,
        "epoch": 9
    },
    "CommonsenseReasoningDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CommonsenseReasoningDataLoader"][1],
        "include_tags": False,
        "grad_accum": 8,
        "lr": 2e-5,
        "epoch": 20
    },
    "BestCounterNarrativeDataLoader": {
        "trunc_bool": True,
        "inst": instructions["BestCounterNarrativeDataLoader"][1],
        "include_tags": True,
        "grad_accum": 8,
        "lr": 2e-5,
        "epoch": 15
    },
    "CHOCounterNarrativeDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CHOCounterNarrativeDataLoader"][1],
        "include_tags": True,
        "grad_accum": 8,
        "lr": 2e-5,
        "epoch": 5
    },
    "CounterNarrativeGrammaticalityDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterNarrativeGrammaticalityDataLoader"][0],
        "include_tags": True,
        "grad_accum": 32,
        "lr": 2e-5,
        "epoch": 20
    },
    "CounterNarrativeSpecificityDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterNarrativeSpecificityDataLoader"][2],
        "include_tags": False,
        "grad_accum": 16,
        "lr": 2e-5,
        "epoch": 5
    },
    "CounterNarrativeSuitabilityDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterNarrativeSuitabilityDataLoader"][1],
        "include_tags": True,
        "grad_accum": 32,
        "lr": 2e-5,
        "epoch": 15
    },
    "CounterNarrativeInformativenessDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterNarrativeInformativenessDataLoader"][2],
        "include_tags": False,
        "grad_accum": 16,
        "lr": 2e-5,
        "epoch": 19
    },
    "CounterNarrativeOffensivenessDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterNarrativeOffensivenessDataLoader"][1],
        "include_tags": True,
        "grad_accum": 16,
        "lr": 3e-4,
        "epoch": 12
    },
    "CounterNarrativeStanceDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterNarrativeStanceDataLoader"][2],
        "include_tags": False,
        "grad_accum": 32,
        "lr": 2e-5,
        "epoch": 16
    },
    "HateSpeechOffensivenessDataLoader": {
        "trunc_bool": True,
        "inst": instructions["HateSpeechOffensivenessDataLoader"][0],
        "include_tags": True,
        "grad_accum": 32,
        "lr": 3e-4,
        "epoch": 1
    },
    "CounterfactualStoryRewritingCounterfactualDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterfactualStoryRewritingCounterfactualDataLoader"][1],
        "include_tags": True,
        "grad_accum": 4,
        "lr": 2e-5,
        "epoch": 15
    },
    "CounterfactualStoryRewritingEndingDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterfactualStoryRewritingEndingDataLoader"][0],
        "include_tags": True,
        "grad_accum": 32,
        "lr": 2e-5,
        "epoch": 8
    },
    "CounterfactualStoryRewritingPlotDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterfactualStoryRewritingPlotDataLoader"][3],
        "include_tags": True,
        "grad_accum": 8,
        "lr": 2e-5,
        "epoch": 1
    },
    "CounterfactualStoryRewritingPremiseDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterfactualStoryRewritingPremiseDataLoader"][2],
        "include_tags": False,
        "grad_accum": 4,
        "lr": 2e-5,
        "epoch": 4
    },
    "CounterfactualStoryRewritingSecondDataLoader": {
        "trunc_bool": True,
        "inst": instructions["CounterfactualStoryRewritingSecondDataLoader"][2],
        "include_tags": False,
        "grad_accum": 4,
        "lr": 2e-5,
        "epoch": 15
    },
    "DefeasibleInferenceAttenuatorEffectivenessDataLoader": {
        "trunc_bool": True,
        "inst": instructions["DefeasibleInferenceAttenuatorEffectivenessDataLoader"][0],
        "include_tags": False,
        "grad_accum": 8,
        "lr": 2e-5,
        "epoch": 14
    },
    "DefeasibleInferenceIntensifierEffectivenessDataLoader": {
        "trunc_bool": True,
        "inst": instructions["DefeasibleInferenceIntensifierEffectivenessDataLoader"][1],
        "include_tags": False,
        "grad_accum": 32,
        "lr": 2e-5,
        "epoch": 18
    },
    "HellaSwagDataLoader": {
        "trunc_bool": True,
        "inst": instructions["HellaSwagDataLoader"][2],
        "include_tags": False,
        "grad_accum": 32,
        "lr": 2e-5,
        "epoch": 6
    },
}