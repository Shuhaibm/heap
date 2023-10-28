# Data from paper: CommonGen: A Constrained Text Generation Challenge for Generative Commonsense Reasoning
# https://aclanthology.org/2020.findings-emnlp.165.pdf

# Task: Generative commonsense plausibility

# Annotators were given a concept set, a reference sentences and two sentences
# Asked to compare the two sentences with respect to their commonsense plausibility
#   -1: Sentence 1 is much less realistic than Sentence 2
# -0.5:
#    0: the two sentences sense are equally realistically plausible
#  0.5:
#    1: Sentence 1 is much more realistic than Sentence 2

# 1079 Comparison Data points

import csv
from data_collection.DataLoader import DataLoader
from collections import defaultdict

class CommonsenseReasoningDataLoader(DataLoader):
    def get_comparison_data(self):
        # One data point:
            #   {
            #     concept_set: ,
            #     reference: ,
            #     more_plausible_sentence: ,
            #     less_plausible_sentence: 
            #   }

        data = []
        human_eval_results_files = ['data_collection/commonsense_reasoning/data/[1] Human Evaluation Sheet - generation_pairs.csv', 'data_collection/commonsense_reasoning/data/[2] Human Evaluation Sheet - generation_pairs.csv',
                                    'data_collection/commonsense_reasoning/data/[3] Human Evaluation Sheet - Sheet1.csv', 'data_collection/commonsense_reasoning/data/[4] Human Evaluation Sheet - generation_pairs.csv']

        for human_eval_results_file in human_eval_results_files:
            with open(human_eval_results_file) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                concept_set,reference = None,None
                for row in reader:
                    if row[0][:12] == "Concept set:":
                        concept_set = row[0][13:]
                    elif row[0][:10] == "Reference:":
                        reference = row[0][11:]
                    elif row != ["","",""] and concept_set and reference:
                        data.append([concept_set,reference,row[0],row[1],row[2]])
        
        combined_data = defaultdict(list) # TODO: I am getting the average of all the annotations, is that okay?
        for data_elem in data:
            sample = ''.join(data_elem[:-1])
            combined_data[sample].append(data_elem)
            
        
        avg_data = []
        for key in combined_data:
            
            avg = sum([float(data_point[4]) for data_point in combined_data[key] if data_point[4]!=""])/len(combined_data[key]) # Am ignoring when data_point[4] == "" --> is this a data error?
            
            data_sample = combined_data[key][0]
            if avg > 0:
                avg_data.append({"concept_set": data_sample[0], "reference": data_sample[1],"more_plausible_sentence":data_sample[2],"less_plausible_sentence":data_sample[3]})
            elif avg < 0:
                avg_data.append({"concept_set": data_sample[0], "reference": data_sample[1],"more_plausible_sentence":data_sample[3],"less_plausible_sentence":data_sample[2]})

        return avg_data

    def concatenate_data(self):
        data = self.get_comparison_data()
        concat_data = []

        for data_point in data:
            concept_set = data_point["concept_set"]
            reference = data_point["reference"]
            more_plausible_sentence = data_point["more_plausible_sentence"]
            less_plausible_sentence = data_point["less_plausible_sentence"]
            
            if self.include_tags:
                good_sample = "concept set" + self.sep_token + concept_set + self.sep_token + "reference sentence" + self.sep_token + reference + self.sep_token + "sentence" + self.sep_token + more_plausible_sentence
                bad_sample = "concept set" + self.sep_token + concept_set + self.sep_token + "reference sentence" + self.sep_token + reference + self.sep_token + "sentence" + self.sep_token + less_plausible_sentence
            else:
                good_sample = concept_set + self.sep_token + reference + self.sep_token + more_plausible_sentence
                bad_sample = concept_set + self.sep_token + reference + self.sep_token + less_plausible_sentence
            
            concat_data.append({"good_sample":good_sample,"bad_sample":bad_sample})
        
        return concat_data