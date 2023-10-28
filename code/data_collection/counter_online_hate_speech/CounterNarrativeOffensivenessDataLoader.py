# Data from paper: Towards Automatic Generation of Messages Countering Online Hate Speech and Microaggressions
# https://aclanthology.org/2022.woah-1.2.pdf

import json
from data_collection.DataLoader import DataLoader

# Task 2: Counter Narrative Offensiveness

# Annotators were given hate speech or microagression, and they were asked to evaluate
# the counternarrative on a five-point Likert scale for the following criteria:
#   Offensiveness
#   Stance
#   Informativeness

# 685 Comparison Data points

class CounterNarrativeOffensivenessDataLoader(DataLoader):
    def get_data(self):
        data = []
        human_eval_results_files = [
            'data_collection/counter_online_hate_speech/data/counter_conan.json', 'data_collection/counter_online_hate_speech/data/counter_sbic.json']
        for human_eval_results_file in human_eval_results_files:
            with open(human_eval_results_file, 'r') as f:
                json_data = json.load(f)

                for elem in json_data:
                    data.append({"hate_speech": json_data[elem]["post"]["text"], 
                                    "counternarrative_1":json_data[elem]["GPT-2"]["text"],
                                    "offensiveness_1": sum(json_data[elem]["GPT-2"]["score"]["off"])//3,
                                    "counternarrative_2":json_data[elem]["GPT-Neo"]["text"],
                                    "offensiveness_2": sum(json_data[elem]["GPT-Neo"]["score"]["off"])//3,
                                    "counternarrative_3":json_data[elem]["GPT-3"]["text"],
                                    "offensiveness_3": sum(json_data[elem]["GPT-3"]["score"]["off"])//3})
                    
        return data


    def get_comparison_data(self):
        data = self.get_data()
        comparison_data = []
        for i,data_point in enumerate(data):
            if data_point["offensiveness_1"] > data_point["offensiveness_2"]:
                comparison_data.append({
                    "hate_speech":data_point["hate_speech"], 
                    "more_offensive":data_point["counternarrative_1"],
                    "less_offensive":data_point["counternarrative_2"],
                    "good_rank": data_point["offensiveness_1"],
                    "bad_rank": data_point["offensiveness_2"]
                    })
            if data_point["offensiveness_1"] > data_point["offensiveness_3"]:
                comparison_data.append({
                    "hate_speech":data_point["hate_speech"], 
                    "more_offensive":data_point["counternarrative_1"],
                    "less_offensive":data_point["counternarrative_3"],
                    "good_rank": data_point["offensiveness_1"],
                    "bad_rank": data_point["offensiveness_3"]
                    })
            
            if data_point["offensiveness_2"] > data_point["offensiveness_1"]:
                comparison_data.append({
                    "hate_speech":data_point["hate_speech"], 
                    "more_offensive":data_point["counternarrative_2"],
                    "less_offensive":data_point["counternarrative_1"],
                    "good_rank": data_point["offensiveness_2"],
                    "bad_rank": data_point["offensiveness_1"]
                    })

            if data_point["offensiveness_2"] > data_point["offensiveness_3"]:
                comparison_data.append({
                    "hate_speech":data_point["hate_speech"], 
                    "more_offensive":data_point["counternarrative_2"],
                    "less_offensive":data_point["counternarrative_3"],
                    "good_rank": data_point["offensiveness_2"],
                    "bad_rank": data_point["offensiveness_3"]
                    })
            
            if data_point["offensiveness_3"] > data_point["offensiveness_1"]:
                comparison_data.append({
                    "hate_speech":data_point["hate_speech"], 
                    "more_offensive":data_point["counternarrative_3"],
                    "less_offensive":data_point["counternarrative_1"],
                    "good_rank": data_point["offensiveness_3"],
                    "bad_rank": data_point["offensiveness_1"]
                    })
            if data_point["offensiveness_3"] > data_point["offensiveness_2"]:
                comparison_data.append({
                    "hate_speech":data_point["hate_speech"], 
                    "more_offensive":data_point["counternarrative_3"],
                    "less_offensive":data_point["counternarrative_2"],
                    "good_rank": data_point["offensiveness_3"],
                    "bad_rank": data_point["offensiveness_2"]
                    })

        return comparison_data

    def concatenate_data(self):
        data = self.get_comparison_data()
        concat_data = []

        for data_point in data:
            if self.include_tags:
                good_sample = "hate speech" + self.sep_token + data_point["hate_speech"] + self.sep_token + "counter narrative" + self.sep_token + data_point["more_offensive"]
                bad_sample = "hate speech" + self.sep_token + data_point["hate_speech"] + self.sep_token + "counter narrative" + self.sep_token + data_point["less_offensive"]
            else:
                good_sample = data_point["hate_speech"] + self.sep_token + data_point["more_offensive"]
                bad_sample = data_point["hate_speech"] + self.sep_token + data_point["less_offensive"]

            concat_data.append({"good_sample":good_sample,"bad_sample":bad_sample,"good_rank":data_point["good_rank"],"bad_rank":data_point["bad_rank"]})
        
        return concat_data