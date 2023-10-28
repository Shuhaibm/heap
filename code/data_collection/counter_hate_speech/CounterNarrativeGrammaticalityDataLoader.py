# Data from paper: Using Pre-Trained Language Models for Producing Counter Narratives Against Hate Speech: a Comparative Study
# https://aclanthology.org/2022.findings-acl.245.pdf
# Total: 1000

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


# Task 3: Counter Narrative Grammaticality
# 863 Comparison Data points

import csv
from data_collection.DataLoader import DataLoader
from collections import defaultdict

class CounterNarrativeGrammaticalityDataLoader(DataLoader):
    def get_data(self):
        data = []
        human_eval_results_file = "data_collection/counter_hate_speech/data/Pre-Trained_LMs_for_Counter_Narratives_human_evaluation_data.csv"
        with open(human_eval_results_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0: continue

                data.append({"hate_speech":row[0],"target":row[1],"dialoGPT":row[2],"BART":row[3],"BERT":row[4],"T5":row[5],"gpt2":row[6],
                    "dialoGPT_grammar":int(row[29]),"BART_grammar":int(row[24]),"BERT_grammar":int(row[34]),"T5_grammar":int(row[19]),"gpt2_grammar":int(row[14])})
        return data

    def get_comparison_data(self):
        data = self.get_data()

        combined_data_dict = defaultdict(list)
        for data_point in data:
            hs_and_target = data_point["hate_speech"] + data_point["target"]
            combined_data_dict[hs_and_target].append(data_point)
        combined_data = []
        for key in combined_data_dict:
            first_data_point = combined_data_dict[key][0]
            second_data_point = combined_data_dict[key][1]
            new_data_point = first_data_point

            new_data_point["dialoGPT_grammar"] = (first_data_point["dialoGPT_grammar"] + second_data_point["dialoGPT_grammar"])//2
            new_data_point["BART_grammar"] = (first_data_point["BART_grammar"] + second_data_point["BART_grammar"])//2
            new_data_point["BERT_grammar"] = (first_data_point["BERT_grammar"] + second_data_point["BERT_grammar"])//2
            new_data_point["T5_grammar"] = (first_data_point["T5_grammar"] + second_data_point["T5_grammar"])//2
            new_data_point["gpt2_grammar"] = (first_data_point["gpt2_grammar"] + second_data_point["gpt2_grammar"])//2

            combined_data.append(new_data_point)

        comparison_data = []
        generator_list =  ["dialoGPT","BART","BERT","T5","gpt2"]
        for data_point in combined_data:
            for i,generator_one in enumerate(generator_list):
                for generator_two in generator_list[i+1:]:
                    generator_one_gramm,generator_two_gramm = data_point[generator_one+"_grammar"],data_point[generator_two+"_grammar"]

                    if generator_one_gramm > generator_two_gramm:
                        comparison_data.append({
                            "hate_speech":data_point["hate_speech"],"target":data_point["target"],
                            "grammatical_cn":data_point[generator_one],
                            "less_grammatical_cn":data_point[generator_two],
                            "good_rank": generator_one_gramm,
                            "bad_rank": generator_two_gramm,
                            })
                    elif generator_two_gramm > generator_one_gramm:
                        comparison_data.append({
                            "hate_speech":data_point["hate_speech"],"target":data_point["target"],
                            "grammatical_cn":data_point[generator_two],
                            "less_grammatical_cn":data_point[generator_one],
                            "good_rank": generator_two_gramm,
                            "bad_rank": generator_one_gramm,
                            })

        return comparison_data

    def concatenate_data(self):
        data = self.get_comparison_data()
        concat_data = []

        for data_point in data:
            hate_speech = data_point["hate_speech"]
            target = data_point["target"]
            best_cn = data_point["grammatical_cn"]
            other_cn = data_point["less_grammatical_cn"]

            if self.include_tags:
                good_sample = "hate speech"  + self.sep_token + hate_speech + self.sep_token + "target"  + self.sep_token + target  + self.sep_token + "counter narrative"  + self.sep_token + best_cn
                bad_sample = "hate speech"  + self.sep_token + hate_speech + self.sep_token + "target"  + self.sep_token + target  + self.sep_token + "counter narrative"  + self.sep_token + other_cn
            else:
                good_sample = hate_speech + self.sep_token + target  + self.sep_token + best_cn
                bad_sample = hate_speech + self.sep_token + target  + self.sep_token + other_cn

            concat_data.append({"good_sample":good_sample,"bad_sample":bad_sample,"good_rank":data_point["good_rank"],"bad_rank":data_point["bad_rank"]})
        
        return concat_data