# Data from HellaSwag: Can a Machine Really Finish Your Sentence? (ACL 2019)
# https://rowanzellers.com/hellaswag/

# For each data sample, you are given a context with four follow up sentences,
# one of which is teh correct follow up. The model must select the best ending 
# to the context

# Note: the test dataset does not include the correct answers, so will ignore it for now.
# Leaderboard for accuracy on test set at link above

# Task 1: Picking the best ending to the context

# Data from paper: HellaSwag: Can a Machine Really Finish Your Sentence?
# https://arxiv.org/pdf/1905.07830.pdf
# 149841 comparison data points

import jsonlines
from data_collection.DataLoader import DataLoader

class HellaSwagDataLoader(DataLoader):
    def get_data(self):
        data = []
        human_eval_results_files = ['data_collection/commonsense_nli/data/hellaswag_train.jsonl','data_collection/commonsense_nli/data/hellaswag_val.jsonl']
        for human_eval_results_file in human_eval_results_files:
            f = jsonlines.open(human_eval_results_file)
            for line in f.iter():
                data.append({
                    "context":line["ctx"],"correct_i": line["label"],"endings": line["endings"]
                    })
        return data

    def get_comparison_data(self):
        data = self.get_data()
        comparison_data = []
        for data_point in data:
            correct_i = data_point["correct_i"]
            correct_ending = data_point["endings"][correct_i]
            for i,ending in enumerate(data_point["endings"]):
                if i == correct_i: continue
                comparison_data.append({
                    "context":data_point["context"],"correct_ending": correct_ending,"wrong_ending": ending
                    })
        
        return comparison_data


    def concatenate_data(self):
        # One data point:
            #   {
            #     good_sample: ,
            #     bad_sample: 
            #   }
        data = self.get_comparison_data()
        concat_data = []

        for data_point in data:
            context = data_point["context"]
            correct_ending = data_point["correct_ending"]
            wrong_ending = data_point["wrong_ending"]
            
            if self.include_tags:
                good_sample = "context" + self.sep_token + context  + self.sep_token + "follow up sentence" + self.sep_token + correct_ending
                bad_sample = "context" + self.sep_token + context  + self.sep_token + "follow up sentence" + self.sep_token + wrong_ending
            else:
                good_sample = context  + self.sep_token + correct_ending
                bad_sample = context  + self.sep_token + wrong_ending

            concat_data.append({"good_sample":good_sample,"bad_sample":bad_sample})

        return concat_data