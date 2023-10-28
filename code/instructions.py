sep_token = ' sep_token '

instructions = {
    "AdviceHelpfullnessDataLoader": [
            "You are given a situation and advice. Determine how helpful the given advice is for the situation." + sep_token,
            "Given a situation and advice, determine how helpful the advice is for the situation." + sep_token,
            "Determine how helpful the advice is given a situation and advice." + sep_token
        ],
    "AnswerGrammaticalityDataLoader": [
            "You are given a narrative, a question and an answer to that question. Determine the grammaticality of the answer." + sep_token,
            "Given a narrative, a question and an answer to that question, determine the grammaticality of the answer." + sep_token,
            "Determine the grammaticality of an answer given a narrative, a question and an answer to that question." + sep_token
        ],
    "AnswerValidityDataLoader": [
            "You are given a narrative, a question and an answer to that question. Determine the plausibility of the answer." + sep_token,
            "Given a narrative, a question and an answer to that question, determine the plausibility of the answer." + sep_token,
            "Determine the plausibility of the answer given a narrative, a question and an answer to that question." + sep_token
        ],
    "QuestionAnswerabilityDataLoader": [
            "You are given a narrative and a question. Determine the comprehensibility of the question and whether the narrative contains the answer for the question." + sep_token,
            "Given a narrative and a question, determine the comprehensibility of the question and whether the narrative contains the answer for the question." + sep_token,
            "Determine the comprehensibility of the question and whether the narrative contains the answer for the question given a narrative and a question." + sep_token
        ],
    "HellaSwagDataLoader": [
            "You are given a context and a follow up sentence. Determine how appropriate the follow up sentence is for the context." + sep_token,
            "Given a context and a follow up sentence, determine how appropriate the follow up sentence is for the context." + sep_token,
            "Determine how appropriate the follow up sentence is for the context given a context and a follow up sentence." + sep_token,
            "You are given a paragraph. Determine how appropriate the last sentence is given the rest of the paragraph." + sep_token,
            "Given a paragraph, determine how appropriate the last sentence is." + sep_token,
            "Detemrine how appropriate the last sentence is given the rest of the paragraph." + sep_token,
            "How appropriate is the last sentence given the rest of the paragraph." + sep_token
        ],
    "CommonsenseReasoningDataLoader": [
            "You are given a concept set, a reference sentence, and a sentence. Determine the plausibility of the sentence." + sep_token,
            "Given a concept set, a reference sentence, and a sentence, determine the plausibility of the sentence." + sep_token,
            "Determine the plausibility of the sentence given a concept set, a reference sentence, and a sentence." + sep_token
        ],
    "BestCounterNarrativeDataLoader": [
            "You are given a hate speech, the target of the hate speech, and a counter narrative in response to the hate speech. Determine how good the counter narrative is." + sep_token,
            "Given a hate speech, the target of the hate speech, and a counter narrative in response to the hate speech, determine how good the counter narrative is." + sep_token,
            "Determine how good the counter narrative is given a hate speech, the target of the hate speech, and a counter narrative in response to the hate speech." + sep_token
        ],
    "CHOCounterNarrativeDataLoader": [
            "You are given a hate speech, the target of the hate speech, and a counter narrative in response to the hate speech. Determine how appropriate the counter narrative is in a real case scenario." + sep_token,
            "Given a hate speech, the target of the hate speech, and a counter narrative in response to the hate speech, determine how appropriate the counter narrative is in a real case scenario." + sep_token,
            "Determine how appropriate the counter narrative is in a real case scenario given a hate speech, the target of the hate speech, and a counter narrative in response to the hate speech." + sep_token
        ],
    "CounterNarrativeSuitabilityDataLoader": [
            "You are given a hate speech, the target of the hate speech, and a counter narrative in response to the hate speech. Determine how suitable the counter narrative is to the hate speech in terms of semantic relatedness and in terms of not spreading hate." + sep_token,
            "Given a hate speech, the target of the hate speech, and a counter narrative in response to the hate speech, determine how suitable the counter narrative is to the hate speech in terms of semantic relatedness and in terms of not spreading hate." + sep_token,
            "Determine how suitable the counter narrative is to the hate speech in terms of semantic relatedness and in terms of not spreading hate given a hate speech, the target of the hate speech, and a counter narrative in response to the hate speech." + sep_token
        ],
        # TODO: These instructions may not be the best - I dont know how well I summarized the CN guidelines
    "CounterNarrativeSpecificityDataLoader": [
            "You are given a hate speech, the target of the hate speech, and a counter narrative in response to the hate speech. Determine how specific the arguments brought by the counter narrative are in response to the hate speech." + sep_token,
            "Given a hate speech, the target of the hate speech, and a counter narrative in response to the hate speech, determine how specific the arguments brought by the counter narrative are in response to the hate speech." + sep_token,
            "Determine how specific the arguments brought by the counter narrative are in response to the hate speech given a hate speech, the target of the hate speech, and a counter narrative in response to the hate speech." + sep_token
        ],
    "CounterNarrativeGrammaticalityDataLoader": [
            "You are given a hate speech, the target of the hate speech, and a counter narrative in response to the hate speech. Determine the grammaticality of the counter narrative." + sep_token,
            "Given a hate speech, the target of the hate speech, and a counter narrative in response to the hate speech, determine the grammaticality of the counter narrative." + sep_token,
            "Determine the grammaticality of the counter narrative given a hate speech, the target of the hate speech, and a counter narrative in response to the hate speech." + sep_token
        ],
    "CounterNarrativeInformativenessDataLoader": [
            "You are given a hate speech and a counter narrative in response to the hate speech. Determine how informative the counter narrative is." + sep_token,
            "Given a hate speech and a counter narrative in response to the hate speech, determine  how informative the counter narrative is." + sep_token,
            "Determine how informative the counter narrative is given a hate speech and a counter narrative in response to the hate speech." + sep_token
        ],
    "HateSpeechOffensivenessDataLoader": [
            "You are given a hate speech. Determine how offensive the hate speech is." + sep_token,
            "Given a hate speech, determine how offensive the hate speech is." + sep_token,
            "Determine how offensive the given hate speech is." + sep_token
    ],
    "CounterNarrativeStanceDataLoader": [
            "You are given a hate speech and a counter narrative in response to the hate speech. Evaluate the stance of the counter narrative." + sep_token,
            "Given a hate speech and a counter narrative in response to the hate speech, evaluate the stance of the counter narrative." + sep_token,
            "Evaluate the stance of the counter narrative given a hate speech and a counter narrative in response to the hate speech." + sep_token
        ],
    "CounterNarrativeOffensivenessDataLoader": [
            "You are given a hate speech and a counter narrative in response to the hate speech. Determine how offensive the counter narrative is." + sep_token,
            "Given a hate speech and a counter narrative in response to the hate speech, determine how offensive the counter narrative is." + sep_token,
            "Determine how offensive the counter narrative is given a hate speech and a counter narrative in response to the hate speech." + sep_token
        ],
    "CounterfactualStoryRewritingCounterfactualDataLoader": [
            "You have a story that consists of a premise, a second sentence, and an ending. A second story with the same premise is provided but the second sentence is replaced with a counterfactual sentence, and the ending is rewritten. Determine how reasonable the rewritten ending is given the premise and counterfactual sentence provided." + sep_token,
            "You have a story that consists of a premise, a second sentence, and an ending. A second story with the same premise is provided but the second sentence is replaced with a counterfactual sentence, and the ending is rewritten, determine how reasonable the rewritten ending is given the premise and counterfactual sentence provided." + sep_token,
            "Determine how reasonable the rewritten ending is given the premise and counterfactual sentence provided given a story that consists of a premise, a second sentence, and an ending as well as a second story with the same premise is provided but the second sentence is replaced with a counterfactual sentence, and the ending is rewritten." + sep_token,
            # Might make sense to delete previous ones!
            "You are given a story and an ending. Determine how reasonable the ending is for the given the story." + sep_token,
            "Given a story and an ending, determine how reasonable the ending is given the story." + sep_token,
            "Determine how reasonable the ending is given the story and an ending." + sep_token
        ],

    "CounterfactualStoryRewritingSecondDataLoader": [
            "You have a story that consists of a premise, a second sentence, and an ending. A second story with the same premise is provided but the second sentence is replaced with a counterfactual sentence, and the ending is rewritten. Determine how well the rewritten ending keeps in mind the details provided in the counterfactual." + sep_token,
            "You have a story that consists of a premise, a second sentence, and an ending. A second story with the same premise is provided but the second sentence is replaced with a counterfactual sentence, and the ending is rewritten, determine how well the rewritten ending keeps in mind the details provided in the counterfactual." + sep_token,
            "Determine how well the rewritten ending keeps in mind the details provided in the counterfactual given a story that consists of a premise, a second sentence, and an ending as well as a second story with the same premise is provided but the second sentence is replaced with a counterfactual sentence, and the ending is rewritten. " + sep_token,
            # Might make sense to delete previous ones
            "You are given a story and an ending. Determine how well the ending keeps in mind the details in the second sentence." + sep_token,
            "Given a story and an ending, determine how well the ending keeps in mind the details in the second sentence." + sep_token,
            "Determine well the ending keeps in mind the details in the second sentence given a story and an ending." + sep_token
        ],
    "CounterfactualStoryRewritingPremiseDataLoader": [
            "You have a story that consists of a premise, a second sentence, and an ending. A second story with the same premise is provided but the second sentence is replaced with a counterfactual sentence, and the ending is rewritten. Determine how well the rewritten ending keeps in mind the details provided in the premise." + sep_token,
            "You have a story that consists of a premise, a second sentence, and an ending. A second story with the same premise is provided but the second sentence is replaced with a counterfactual sentence, and the ending is rewritten, determine how well the rewritten ending keeps in mind the details provided in the premise." + sep_token,
            "Determine how well the rewritten ending keeps in mind the details provided in the premise given a story that consists of a premise, a second sentence, and an ending as well as a second story with the same premise is provided but the second sentence is replaced with a counterfactual sentence, and the ending is rewritten. " + sep_token,
            # Might make sense to delete previous ones
            "You are given a story and an ending. Determine how well the ending keeps in mind the details in the first sentence." + sep_token,
            "Given a story and an ending, determine how well the ending keeps in mind the details in the first sentence." + sep_token,
            "Determine well the ending keeps in mind the details in the first sentence given a story and an ending." + sep_token
        ],
    "CounterfactualStoryRewritingPlotDataLoader": [
            "You have a story that consists of a premise, a second sentence, and an ending. A second story with the same premise is provided but the second sentence is replaced with a counterfactual sentence, and the ending is rewritten. Determine how related the plot in the rewritten ending is to the plot in the premise." + sep_token,
            "You have a story that consists of a premise, a second sentence, and an ending. A second story with the same premise is provided but the second sentence is replaced with a counterfactual sentence, and the ending is rewritten, determine how related the plot in the rewritten ending is to the plot in the premise." + sep_token,
            "Determine how related the plot in the rewritten ending is to the plot in the premise given a story that consists of a premise, a second sentence, and an ending as well as a second story with the same premise is provided but the second sentence is replaced with a counterfactual sentence, and the ending is rewritten." + sep_token,
            # Might make sense to delete previous ones
            "You are given a story, an original ending and a rewritten ending. Determine how well the plot in the rewritten ending relates to the plot of the original ending." + sep_token,
            "Given a story, an original ending and a rewritten ending, determine how well the plot in the rewritten ending relates to the plot of the original ending." + sep_token,
            "Determine how well the plot in the rewritten ending relates to the plot of the original ending give a story, an original ending and a rewritten ending. " + sep_token,
        ],
    "CounterfactualStoryRewritingEndingDataLoader": [
            "You have a story that consists of a premise, a second sentence, and an ending. A second story with the same premise is provided but the second sentence is replaced with a counterfactual sentence, and the ending is rewritten. Determine how reasonably the rewritten ending outlines a sequence of events." + sep_token,
            "You have a story that consists of a premise, a second sentence, and an ending. A second story with the same premise is provided but the second sentence is replaced with a counterfactual sentence, and the ending is rewritten, determine how reasonably the rewritten ending outlines a sequence of events." + sep_token,
            "Determine how reasonably the rewritten ending outlines a sequence of events given a story that consists of a premise, a second sentence, and an ending as well as a second story with the same premise is provided but the second sentence is replaced with a counterfactual sentence, and the ending is rewritten. " + sep_token,
            # Might make sense to delete previous ones
            "You are given a story and an ending. Determine how well the ending outlines a reasonable sequence of events." + sep_token,
            "Given a story and an ending, determine how well the ending outlines a reasonable sequence of events." + sep_token,
            "Determine how well the ending outlines a reasonable sequence of events given a story and an ending." + sep_token
        ],
    "DefeasibleInferenceIntensifierEffectivenessDataLoader": [
            "You are given a premise, a hypothesis, and an update sentence. Determine how much the much the update sentence strengthens the hypothesis." + sep_token,
            "Given a premise, a hypothesis, and an update sentence, determine how much the much the update sentence strengthens the hypothesis." + sep_token,
            "Determine how much the update sentence strengthens the hypothesis given a premise, a hypothesis and an update sentence." + sep_token
        ],
    "DefeasibleInferenceAttenuatorEffectivenessDataLoader": [
            "You are given a premise, a hypothesis, and an update sentence. Determine how much the much the update sentence weakens the hypothesis." + sep_token,
            "Given a premise, a hypothesis, and an update sentence, determine how much the much the update sentence weakens the hypothesis." + sep_token,
            "Determine how much the update sentence weakens the hypothesis given a premise, a hypothesis and an update sentence." + sep_token
        ]
}