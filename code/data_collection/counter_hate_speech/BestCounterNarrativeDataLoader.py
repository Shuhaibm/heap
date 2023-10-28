# Data from paper: Using Pre-Trained Language Models for Producing Counter Narratives Against Hate Speech: a Comparative Study
# https://aclanthology.org/2022.findings-acl.245.pdf

import csv

# Evaluation of Counter Narratives Against Hate Speech

# Annotators were provided with hate speech, target of the hate speech,
# and generated responses to the hate speech (from 5 LMs). They were
# asked to evaluate the response based on the criteria:
#    Suitableness (SUI): (5 point likert scale)
#                       measure how suitable a CN is to the HS in terms of
#                       semantic relatedness and in terms of adherence to
#                       CN guidelines (see https://getthetrollsout.org/stoppinghate)
#    Grammaticality (GRM): (5 point likert scale)
#                       how grammatically correct a generated CN is
#    Specificity (SPE): (5 point likert scale)
#                       how specific are the arguments brought by the CN in response
#                       to the HS
#    Choose-or-not (CHO): (binary)
#                       whether the annotator would select that CN to post-edit
#                       and use in a real case scenario
#    Is-best (BEST): (binary)
#                       whether the CN is absolute best among the ones generated


# Task 1: Best Counter Narrative
# 1000 Comparison Data points

import csv
from data_collection.DataLoader import DataLoader
from collections import defaultdict

class BestCounterNarrativeDataLoader(DataLoader):
    def get_data(self):
        data = []
        human_eval_results_file = "data_collection/counter_hate_speech/data/Pre-Trained_LMs_for_Counter_Narratives_human_evaluation_data.csv"
        with open(human_eval_results_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0: continue

                if row[16] == "1": best = "gpt2"
                elif row[21] == "1": best = "T5"
                elif row[26] == "1": best = "BART"
                elif row[31] == "1": best = "dialoGPT"
                elif row[36] == "1": best = "BERT"

                data.append({"hate_speech":row[0],"target":row[1],"dialoGPT":row[2],"BART":row[3],"BERT":row[4],"T5":row[5],"gpt2":row[6],"best_one":best})
        return data

    def get_comparison_data(self):
        data = self.get_data()

        combined_data_dict = defaultdict(list)
        for data_point in data:
            hs_and_target = data_point["hate_speech"] + data_point["target"]
            combined_data_dict[hs_and_target].append(data_point)
        combined_data = []
        for key in combined_data_dict:
            new_data_point = combined_data_dict[key][0]
            new_data_point["best_two"] = combined_data_dict[key][1]["best_one"]
            combined_data.append(new_data_point)

        comparison_data = []
        generator_list =  ["dialoGPT","BART","BERT","T5","gpt2"]
        for data_point in combined_data:
            if data_point["best_one"] == data_point["best_two"]:
                best_generator = data_point["best_one"]
                best_cn = data_point[best_generator]

                for generator in generator_list:
                    if generator != best_generator: 
                        comparison_data.append({"hate_speech":data_point["hate_speech"],"target":data_point["target"],"best_cn":best_cn,"other_cn":data_point[generator]})
            else:
                best_generator_one,best_generator_two = data_point["best_one"],data_point["best_two"]
                best_cn_one,best_cn_two = data_point[best_generator_one],data_point[best_generator_two]

                for generator in generator_list:
                    if generator != best_generator_one and generator != best_generator_two:
                        comparison_data.append({"hate_speech":data_point["hate_speech"],"target":data_point["target"],"best_cn":best_cn_one,"other_cn":data_point[generator]})
                for generator in generator_list:
                    if generator != best_generator_one and generator != best_generator_two:
                        comparison_data.append({"hate_speech":data_point["hate_speech"],"target":data_point["target"],"best_cn":best_cn_two,"other_cn":data_point[generator]})

        return comparison_data

    def concatenate_data(self):
        data = self.get_comparison_data()
        concat_data = []

        for data_point in data:
            hate_speech = data_point["hate_speech"]
            target = data_point["target"]
            best_cn = data_point["best_cn"]
            other_cn = data_point["other_cn"]

            if self.include_tags:
                good_sample = "hate speech"  + self.sep_token + hate_speech + self.sep_token + "target"  + self.sep_token + target  + self.sep_token + "counter narrative"  + self.sep_token + best_cn
                bad_sample = "hate speech"  + self.sep_token + hate_speech + self.sep_token + "target"  + self.sep_token + target  + self.sep_token + "counter narrative"  + self.sep_token + other_cn
            else:
                good_sample = hate_speech + self.sep_token + target  + self.sep_token + best_cn
                bad_sample = hate_speech + self.sep_token + target  + self.sep_token + other_cn

            concat_data.append({"good_sample":good_sample,"bad_sample":bad_sample})
        
        return concat_data