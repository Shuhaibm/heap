# Data from paper: Thinking Like a Skeptic: Defeasible Inference in Natural Language
# https://aclanthology.org/2020.findings-emnlp.418.pdf
# Total: 7176 data points

# Annotators are provided with a premise-hypothesis pair, type (strengthener or weakener),
# and a generated update sentence that strengthens/weakens the hypothesis. The annotator
# is asked to rate the generated strengthener/weakener on a 5-point Likert scale:
# Weakens a lot --> Weakens a little --> Neutral --> Strengthens a little --> Strengthens a lot


# Task 1: Effectiveness of generated update for intensifiers

import jsonlines
from data_collection.DataLoader import DataLoader
from collections import defaultdict

# 2655 Comparison Data points

class DefeasibleInferenceAttenuatorEffectivenessDataLoader(DataLoader):
    def get_data(self):
        data = []
        human_eval_results_file = 'data_collection/defeasible_inference/data/def_inf_eval_results.jsonl'
        f = jsonlines.open(human_eval_results_file)
        for line in f.iter():
            if line["pred_label"] == 'attenuator': 
                data.append({'premise':line['premise'],'hypothesis':line['hypothesis'],'update':line['update'],'human_label':-1*self.get_overall_ann(line['human_label'])})
        return data


    def get_comparison_data(self):
        data = self.get_data()

        premise_hypothesis = defaultdict(list)
        for data_point in data:
            premise_hypothesis_pair = data_point['premise'] + data_point['hypothesis']
            premise_hypothesis[premise_hypothesis_pair].append(data_point)

        comparison_data = []
        for key in premise_hypothesis:
            data_one = premise_hypothesis[key][0]
            data_two = premise_hypothesis[key][1]
            data_three = premise_hypothesis[key][2]

            if data_one['human_label'] > data_two['human_label']:
                comparison_data.append({
                    'premise':data_one['premise'],'hypothesis':data_one['hypothesis'],
                    'better_attenuator':data_one['update'],
                    'worse_attenuator':data_two['update'],
                    'good_rank':data_one['human_label'],
                    'bad_rank':data_two['human_label']
                    })
            
            if data_one['human_label'] > data_three['human_label']:
                comparison_data.append({
                    'premise':data_one['premise'],'hypothesis':data_one['hypothesis'],
                    'better_attenuator':data_one['update'],
                    'worse_attenuator':data_three['update'],
                    'good_rank':data_one['human_label'],
                    'bad_rank':data_three['human_label']
                    })
            
            if data_two['human_label'] > data_one['human_label']:
                comparison_data.append({
                    'premise':data_one['premise'],'hypothesis':data_one['hypothesis'],
                    'better_attenuator':data_two['update'],
                    'worse_attenuator':data_one['update'],
                    'good_rank':data_two['human_label'],
                    'bad_rank': data_one['human_label']
                    })
            
            if data_two['human_label'] > data_three['human_label']:
                comparison_data.append({
                    'premise':data_one['premise'],'hypothesis':data_one['hypothesis'],
                    'better_attenuator':data_two['update'],
                    'worse_attenuator':data_three['update'],
                    'good_rank':data_two['human_label'],
                    'bad_rank':data_three['human_label']
                    })

            if data_three['human_label'] > data_one['human_label']:
                comparison_data.append({
                    'premise':data_one['premise'],'hypothesis':data_one['hypothesis'],
                    'better_attenuator':data_three['update'],
                    'worse_attenuator':data_one['update'],
                    'good_rank':data_three['human_label'],
                    'bad_rank':data_one['human_label']
                    })
            
            if data_three['human_label'] > data_two['human_label']:
                comparison_data.append({
                    'premise':data_one['premise'],'hypothesis':data_one['hypothesis'],
                    'better_attenuator':data_three['update'],
                    'worse_attenuator':data_two['update'],
                    'good_rank':data_three['human_label'],
                    'bad_rank':data_two['human_label']
                    })

        return comparison_data

    def concatenate_data(self):
        data = self.get_comparison_data()
        concat_data = []

        for data_point in data:
            if self.include_tags:
                good_sample = "premise" + self.sep_token + data_point["premise"] + self.sep_token + "hypothesis" + self.sep_token + data_point["hypothesis"] + self.sep_token + "update sentence" + self.sep_token + data_point["better_attenuator"]
                bad_sample = "premise" + self.sep_token + data_point["premise"] + self.sep_token + "hypothesis" + self.sep_token + data_point["hypothesis"] + self.sep_token + "update sentence" + self.sep_token + data_point["worse_attenuator"]
            else:
                good_sample = data_point["premise"] + self.sep_token + data_point["hypothesis"] + self.sep_token + data_point["better_attenuator"]
                bad_sample = data_point["premise"] + self.sep_token + data_point["hypothesis"] + self.sep_token + data_point["worse_attenuator"]

            concat_data.append({"good_sample":good_sample,"bad_sample":bad_sample,"good_rank":data_point["good_rank"],"bad_rank":data_point["bad_rank"]})
        
        return concat_data