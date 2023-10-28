import torch
from transformers import AutoModel,BartTokenizer,DataCollatorWithPadding,TrainingArguments,Trainer,set_seed,TrainerCallback

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
from hyperparams import *

import pdb

def single_task_eval(val_dataset, trainer):
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

    return metrics

class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = self._trainer
        val_datasets = self._trainer.eval_dataset
        
        print("*** Begin Post Epoch Evaluate ***")

        for i,val_dataset in enumerate(val_datasets):
            print("Evaluating " + str(i) + " dataset\n")
            metrics = single_task_eval(val_dataset, trainer)
            trainer.log_metrics(str(i) + "_eval", metrics)
            print(metrics)

        print("*** Done Post Epoch Evaluate ***")
            

def compute_correlation(preds, anns):
    pearsons_corr,_ = pearsonr(preds,anns)
    spearmans_corr,_ = spearmanr(preds,anns)
    return pearsons_corr,spearmans_corr

def run_task(data_loader_names,data_collator,train_dataset,val_datasets,lr,grad_accum,epochs):
    print("\n*** Begin ***\n")
    print("\n Learning Rate: " + str(lr))
    print("\n Gradient Accumulation: " + str(grad_accum))
    print("\n Epochs: " + str(epochs))

    training_args = TrainingArguments(
        do_eval=True,
        do_predict=False,
        do_train=True,
        output_dir="./",
        per_device_train_batch_size=4,
        save_strategy="no",

        learning_rate=lr,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs
        )

    trainer = Trainer(
        model=ComparisonModel(),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_datasets,
        data_collator=data_collator,
        )
    trainer.add_callback(CustomCallback(trainer)) 

    # Train
    if training_args.do_train:
        print("*** Train ***")

        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("zeroshot_train", metrics)

    # Validation
    if training_args.do_eval:
        print("*** Evaluate ***")
        print("\n Learning Rate: " + str(lr))
        print("\n Gradient Accumulation: " + str(grad_accum))
        print("\n Epochs: " + str(epochs))

        for val_dataset in val_datasets:
            metrics = single_task_eval(val_dataset, trainer)

            trainer.log_metrics("zeroshot_eval", metrics)

    
    gc.collect()
    torch.cuda.empty_cache()


def main():
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



    multitask_train_dataset,multitask_val_datasets,multitask_test_datasets =  [],[],[]

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    data_collator = ComparisonDataCollatorWithPadding(tokenizer=tokenizer)

    for i,data_loader_i in enumerate(data_loaders):
        hyperparam = hyperparams[data_loader_names[i]]
        trunc_bool,inst,include_tags = hyperparam["trunc_bool"],hyperparam["inst"],hyperparam["include_tags"]


        dataloader = data_loader_i(tokenizer,inst,trunc_bool,include_tags)
        train_dataset,val_dataset,test_dataset = dataloader.tokenize_data()

        multitask_train_dataset += train_dataset
        multitask_val_datasets.append(val_dataset)

    hyperparam = hyperparams["multitask_overall"]
    grad_accum,lr,epoch = hyperparam["grad_accum"],hyperparam["lr"],hyperparam["epoch"]


    print("SYS ARG" + sys.argv[1])
    if sys.argv[1] == "456": 

        random_seed = 456
        print("Random seed: " + str(random_seed))
        print("Random seed: " + str(random_seed))
        print("Random seed: " + str(random_seed))
        set_seed(random_seed)
    if sys.argv[1] == "789": 
        random_seed = 789
        print("Random seed: " + str(random_seed))
        print("Random seed: " + str(random_seed))
        print("Random seed: " + str(random_seed))
        set_seed(random_seed)
    if sys.argv[1] == "999": 
        random_seed = 999
        print("Random seed: " + str(random_seed))
        print("Random seed: " + str(random_seed))
        print("Random seed: " + str(random_seed))
        set_seed(random_seed)
    
    run_task(data_loader_names,data_collator,multitask_train_dataset,multitask_val_datasets,lr,grad_accum,epoch)



if __name__ == '__main__':
    main()
