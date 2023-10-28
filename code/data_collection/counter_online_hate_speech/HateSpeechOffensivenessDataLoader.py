# Data from paper: Towards Automatic Generation of Messages Countering Online Hate Speech and Microaggressions
# https://aclanthology.org/2022.woah-1.2.pdf


# Task 4: Hate Speech Offensiveness

# Annotators were given hate speech or microagression, and they were asked to evaluate
# the offensiveness on a five-point Likert scale.


# 29970 Comparison Data points

import json
from data_collection.DataLoader import DataLoader
import math

class HateSpeechOffensivenessDataLoader(DataLoader):
    def get_data(self):
        data = []
        human_eval_results_files = [
            'data_collection/counter_online_hate_speech/data/counter_conan.json', 'data_collection/counter_online_hate_speech/data/counter_sbic.json']
        for human_eval_results_file in human_eval_results_files:
            with open(human_eval_results_file, 'r') as f:
                json_data = json.load(f)
                for elem in json_data:
                    data.append({"hate_speech":json_data[elem]["post"]["text"], "offensiveness_score":json_data[elem]["post"]["score"]})

        return data

    def get_comparison_data(self):
        data = self.get_data()
        for i,data_point in enumerate(data):
            data[i]["offensiveness_score"] = sum(data_point["offensiveness_score"])//len(data_point["offensiveness_score"])            

        n = len(data)
        train_data =data[:math.floor(n*0.66)]
        eval_and_test_data = data[math.floor(n*0.66):]
        
        comparison_train_data = []
        for i,data_point_one in enumerate(train_data):
            for data_point_two in train_data[i+1:]:
                offensiveness_one,offensiveness_two = data_point_one["offensiveness_score"],data_point_two["offensiveness_score"]

                if offensiveness_one > offensiveness_two:
                    comparison_train_data.append({
                        "more_offensive":data_point_one["hate_speech"],
                        "less_offensive":data_point_two["hate_speech"],
                        "good_rank":offensiveness_one,
                        "bad_rank":offensiveness_two
                        })
                elif offensiveness_two > offensiveness_one:
                    comparison_train_data.append({
                        "more_offensive":data_point_two["hate_speech"],
                        "less_offensive":data_point_one["hate_speech"],
                        "good_rank":offensiveness_two,
                        "bad_rank":offensiveness_one})
        
        comparison_test_data = []
        for i,data_point_one in enumerate(eval_and_test_data):
            for data_point_two in eval_and_test_data[i+1:]:
                offensiveness_one,offensiveness_two = data_point_one["offensiveness_score"],data_point_two["offensiveness_score"]

                if offensiveness_one > offensiveness_two:
                    comparison_test_data.append({
                        "more_offensive":data_point_one["hate_speech"],
                        "less_offensive":data_point_two["hate_speech"],
                        "good_rank":offensiveness_one,
                        "bad_rank":offensiveness_two
                        })
                elif offensiveness_two > offensiveness_one:
                    comparison_test_data.append({
                        "more_offensive":data_point_two["hate_speech"],
                        "less_offensive":data_point_one["hate_speech"],
                        "good_rank":offensiveness_two,
                        "bad_rank":offensiveness_one})

        return comparison_train_data+comparison_test_data

    def concatenate_data(self):
        data = self.get_comparison_data()
        concat_data = []

        for data_point in data:
            if self.include_tags:
                good_sample = "hate speech" + self.sep_token + data_point["more_offensive"]
                bad_sample = "hate speech" + self.sep_token + data_point["less_offensive"]
            else:
                good_sample = data_point["more_offensive"]
                bad_sample = data_point["less_offensive"]

            concat_data.append({"good_sample":good_sample,"bad_sample":bad_sample,"good_rank":data_point["good_rank"],"bad_rank":data_point["bad_rank"]})
        return concat_data

    def tokenize_data(self):
        # One data point:
            # {
            #     good_id: "",
            #     good_am: "",
            #     bad_id: "",
            #     bad_am: ""
            # }
        tokenizer = self.tokenizer
        instructions = self.instructions

        tokenized_data = []
        
        data = self.concatenate_data()
        for data_point in data:
            good_sample_tokenized = tokenizer(data_point["good_sample"])
            bad_sample_tokenized = tokenizer(data_point["bad_sample"])

            
            [instr_id,instr_am] = [[],[]]
            if instructions: [instr_id,instr_am] = tokenizer(instructions).values()

            if self.truncate_right:
                good_id = instr_id + good_sample_tokenized["input_ids"][-512+len(instr_id):]
                good_am = instr_am + good_sample_tokenized["attention_mask"][-512+len(instr_am):]
                bad_id = instr_id + bad_sample_tokenized["input_ids"][-512+len(instr_id):]
                bad_am = instr_am + bad_sample_tokenized["attention_mask"][-512+len(instr_am):]
            else:
                good_id = instr_id + good_sample_tokenized["input_ids"][:512-len(instr_id)]
                good_am = instr_am + good_sample_tokenized["attention_mask"][:512-len(instr_am)]
                bad_id = instr_id + bad_sample_tokenized["input_ids"][:512-len(instr_id)]
                bad_am = instr_am + bad_sample_tokenized["attention_mask"][:512-len(instr_am)]


            if "good_rank" in data_point and "bad_rank" in data_point:
                tokenized_data.append({
                    "good_id": good_id,
                    "good_am": good_am,
                    "bad_id": bad_id,
                    "bad_am": bad_am,
                    "good_rank":data_point["good_rank"],
                    "bad_rank":data_point["bad_rank"]
                })
            else:
                tokenized_data.append({
                    "good_id": good_id,
                    "good_am": good_am,
                    "bad_id": bad_id,
                    "bad_am": bad_am
                })


        n = len(tokenized_data)
        train_tokenized_data = tokenized_data[:13033]
        eval_tokenized_data = tokenized_data[13033:math.floor(n*0.9)]
        test_tokenized_data = tokenized_data[math.floor(n*0.9):]

        return train_tokenized_data,eval_tokenized_data,test_tokenized_data
    