import torch
from transformers import AutoModel,BartTokenizer,DataCollatorWithPadding,TrainingArguments,Trainer
from torch import nn
from scipy.stats import pearsonr,spearmanr
from ComparisonModel import ComparisonModel,ComparisonDataCollatorWithPadding
import gc
import sys
import json 

from data_collection.advice_generation.AdviceHelpfullnessDataLoader import AdviceHelpfullnessDataLoader
from data_collection.answering_why_questions.AnswerGrammaticalityDataLoader import AnswerGrammaticalityDataLoader
from data_collection.answering_why_questions.AnswerValidityDataLoader import AnswerValidityDataLoader
from data_collection.answering_why_questions.QuestionAnswerabilityDataLoader import QuestionAnswerabilityDataLoader
from data_collection.commonsense_nli.HellaSwagDataLoader import HellaSwagDataLoader
from data_collection.commonsense_reasoning.CommonsenseReasoningDataLoader import CommonsenseReasoningDataLoader
from data_collection.counter_hate_speech.BestCounterNarrativeDataLoader import BestCounterNarrativeDataLoader
from data_collection.counter_hate_speech.CHOCounterNarrativeDataLoader import CHOCounterNarrativeDataLoader
from data_collection.counter_hate_speech.CounterNarrativeGrammaticalityDataLoader import CounterNarrativeGrammaticalityDataLoader
from data_collection.counter_hate_speech.CounterNarrativeSpecificityDataLoader import CounterNarrativeSpecificityDataLoader
from data_collection.counter_hate_speech.CounterNarrativeSuitabilityDataLoader import CounterNarrativeSuitabilityDataLoader
from data_collection.counter_online_hate_speech.CounterNarrativeInformativenessDataLoader import CounterNarrativeInformativenessDataLoader
from data_collection.counter_online_hate_speech.CounterNarrativeOffensivenessDataLoader import CounterNarrativeOffensivenessDataLoader
from data_collection.counter_online_hate_speech.CounterNarrativeStanceDataLoader import CounterNarrativeStanceDataLoader
from data_collection.counter_online_hate_speech.HateSpeechOffensivenessDataLoader import HateSpeechOffensivenessDataLoader
from data_collection.counterfactual_story_rewriting.CounterfactualStoryRewritingCounterfactualDataLoader import CounterfactualStoryRewritingCounterfactualDataLoader
from data_collection.counterfactual_story_rewriting.CounterfactualStoryRewritingEndingDataLoader import CounterfactualStoryRewritingEndingDataLoader
from data_collection.counterfactual_story_rewriting.CounterfactualStoryRewritingPlotDataLoader import CounterfactualStoryRewritingPlotDataLoader
from data_collection.counterfactual_story_rewriting.CounterfactualStoryRewritingPremiseDataLoader import CounterfactualStoryRewritingPremiseDataLoader
from data_collection.counterfactual_story_rewriting.CounterfactualStoryRewritingSecondDataLoader import CounterfactualStoryRewritingSecondDataLoader
from data_collection.defeasible_inference.DefeasibleInferenceAttenuatorEffectivenessDataLoader import DefeasibleInferenceAttenuatorEffectivenessDataLoader
from data_collection.defeasible_inference.DefeasibleInferenceIntensifierEffectivenessDataLoader import DefeasibleInferenceIntensifierEffectivenessDataLoader
from instructions import *
import pdb


if sys.argv[1] == "run_all":
    data_loaders = [AdviceHelpfullnessDataLoader,AnswerGrammaticalityDataLoader,AnswerValidityDataLoader,QuestionAnswerabilityDataLoader,HellaSwagDataLoader,
                    CommonsenseReasoningDataLoader,BestCounterNarrativeDataLoader,CHOCounterNarrativeDataLoader,CounterNarrativeGrammaticalityDataLoader,CounterNarrativeSpecificityDataLoader,
                    CounterNarrativeSuitabilityDataLoader,CounterNarrativeInformativenessDataLoader,CounterNarrativeOffensivenessDataLoader,CounterNarrativeStanceDataLoader,HateSpeechOffensivenessDataLoader,
                    CounterfactualStoryRewritingCounterfactualDataLoader,CounterfactualStoryRewritingEndingDataLoader,CounterfactualStoryRewritingPlotDataLoader,CounterfactualStoryRewritingPremiseDataLoader,
                    CounterfactualStoryRewritingSecondDataLoader,DefeasibleInferenceAttenuatorEffectivenessDataLoader,DefeasibleInferenceIntensifierEffectivenessDataLoader]
    data_loader_names = ["AdviceHelpfullnessDataLoader","AnswerGrammaticalityDataLoader","AnswerValidityDataLoader","QuestionAnswerabilityDataLoader","HellaSwagDataLoader",
                        "CommonsenseReasoningDataLoader","BestCounterNarrativeDataLoader","CHOCounterNarrativeDataLoader","CounterNarrativeGrammaticalityDataLoader","CounterNarrativeSpecificityDataLoader",
                        "CounterNarrativeSuitabilityDataLoader","CounterNarrativeInformativenessDataLoader","CounterNarrativeOffensivenessDataLoader","CounterNarrativeStanceDataLoader","HateSpeechOffensivenessDataLoader",
                        "CounterfactualStoryRewritingCounterfactualDataLoader","CounterfactualStoryRewritingEndingDataLoader","CounterfactualStoryRewritingPlotDataLoader","CounterfactualStoryRewritingPremiseDataLoader",
                        "CounterfactualStoryRewritingSecondDataLoader","DefeasibleInferenceAttenuatorEffectivenessDataLoader","DefeasibleInferenceIntensifierEffectivenessDataLoader"]

elif sys.argv[1] == "run_one":
    data_loaders = [AnswerGrammaticalityDataLoader,AnswerValidityDataLoader,QuestionAnswerabilityDataLoader,
                    CommonsenseReasoningDataLoader,BestCounterNarrativeDataLoader,CHOCounterNarrativeDataLoader,CounterNarrativeGrammaticalityDataLoader,CounterNarrativeSpecificityDataLoader,
                    CounterNarrativeSuitabilityDataLoader,CounterNarrativeInformativenessDataLoader,CounterNarrativeOffensivenessDataLoader,CounterNarrativeStanceDataLoader,
                    CounterfactualStoryRewritingCounterfactualDataLoader,CounterfactualStoryRewritingEndingDataLoader,CounterfactualStoryRewritingPlotDataLoader,CounterfactualStoryRewritingPremiseDataLoader,
                    CounterfactualStoryRewritingSecondDataLoader,DefeasibleInferenceIntensifierEffectivenessDataLoader]
    data_loader_names = ["AnswerGrammaticalityDataLoader","AnswerValidityDataLoader","QuestionAnswerabilityDataLoader",
                    "CommonsenseReasoningDataLoader","BestCounterNarrativeDataLoader","CHOCounterNarrativeDataLoader","CounterNarrativeGrammaticalityDataLoader","CounterNarrativeSpecificityDataLoader",
                    "CounterNarrativeSuitabilityDataLoader","CounterNarrativeInformativenessDataLoader","CounterNarrativeOffensivenessDataLoader","CounterNarrativeStanceDataLoader",
                    "CounterfactualStoryRewritingCounterfactualDataLoader","CounterfactualStoryRewritingEndingDataLoader","CounterfactualStoryRewritingPlotDataLoader","CounterfactualStoryRewritingPremiseDataLoader",
                    "CounterfactualStoryRewritingSecondDataLoader","DefeasibleInferenceIntensifierEffectivenessDataLoader"]
elif sys.argv[1] == "run_two":
    data_loaders = [DefeasibleInferenceAttenuatorEffectivenessDataLoader,AdviceHelpfullnessDataLoader,HateSpeechOffensivenessDataLoader,HellaSwagDataLoader]
    data_loader_names = ["DefeasibleInferenceAttenuatorEffectivenessDataLoader","AdviceHelpfullnessDataLoader","HateSpeechOffensivenessDataLoader","HellaSwagDataLoader"]

import pdb

def compute_correlation(preds, anns):
    pearsons_corr,_ = pearsonr(preds,anns)
    spearmans_corr,_ = spearmanr(preds,anns)

def run_task(file_object, data_loader_name,trunc_bool,inst,dataloader):
    print("\n*** Begin " + "," + data_loader_name + " ***\n")
    print("\n Truncate right: " + str(trunc_bool))
    print("\n Instruction: " + inst)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        do_eval=True,
        do_predict=False,
        do_train=True,
        output_dir="./"
        )
    data_collator = ComparisonDataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = dataloader(tokenizer,inst,trunc_bool,True)
    train_dataset,val_dataset,test_dataset = dataloader.tokenize_data()

    trainer = Trainer(
        model=ComparisonModel(),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        )

    # Train
    if training_args.do_train:
        print("*** Train ***")

        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics(data_loader_name + "_train", metrics)

    # Validation
    if training_args.do_eval:
        print("*** Evaluate ***")

        print("\n Truncate right: " + str(trunc_bool))
        print("\n Instruction: " + inst)

        tp,fn=0,0
        ranks = []
        preds = []
        for input in val_dataset:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                trainer.model.to(device)
                data_point = {
                        "good_id": torch.tensor([input["good_id"]]).to(device),
                        "good_am": torch.tensor([input["good_am"]]).to(device),
                        "bad_id": torch.tensor([input["bad_id"]]).to(device),
                        "bad_am": torch.tensor([input["bad_am"]]).to(device)
                    }
            else:
                data_point = {
                        "good_id": torch.tensor([input["good_id"]]),
                        "good_am": torch.tensor([input["good_am"]]),
                        "bad_id": torch.tensor([input["bad_id"]]),
                        "bad_am": torch.tensor([input["bad_am"]])
                    }

            good_score = trainer.model.score(data_point['good_id'],data_point['good_am'])
            bad_score = trainer.model.score(data_point['bad_id'],data_point['bad_am'])

            if good_score>bad_score: tp += 1
            else: fn += 1
            
            # Correlation
            if "good_rank" in input:
                ranks.append(input["good_rank"])
                ranks.append(input["bad_rank"])
                preds.append(good_score.item())
                preds.append(bad_score.item())

        accuracy = tp/(tp+fn)

        metrics = {
            "true positives_1": tp,
            "false negatives_1": fn,
            "accuracy_1": accuracy,
            }

        if len(ranks) > 0:
            pearsons_corr,spearmans_corr = compute_correlation(preds,ranks)
            metrics["Pearsons correlation"] = pearsons_corr
            metrics["Spearmans correlation"] = spearmans_corr

        trainer.log_metrics(data_loader_name + "_eval", metrics)


    # Log to file
    file_object.write(data_loader_name + "\n")
    if trunc_bool:
        file_object.write("Truncate right: True\n")
    else:
        file_object.write("Truncate right: False\n")
    file_object.write("Instruction: " + inst + "\n")
    file_object.write(json.dumps(metrics))
    file_object.write("\n\n")


    if training_args.do_predict:
        print("*** Predict ***")
        # TODO (same as validation)
    
    gc.collect()
    torch.cuda.empty_cache()

    return pearsons_corr,spearmans_corr



file_object = open("results.txt", "w+")

for i,dataloader_i in enumerate(data_loaders):
    trunc_bools = [True,False]
    for inst in instructions[data_loader_names[i]]:
        for trunc_bool in trunc_bools:
            run_task(file_object,data_loader_names[i],trunc_bool,inst,dataloader_i)
            

file_object.close()