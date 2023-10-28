# Data from paper: TuringAdvice: A Generative and Dynamic Evaluation of Language Use
# https://aclanthology.org/2021.naacl-main.386.pdf


# Task 1: Helpfullness of advice

# Annotators were provided with:
#   - A popular reddit situation
#   - Top-rated advice from reddit
#   - Generated advice from each model each model
# Mechanical Turk workers compare helpfullness of each pair of human-written reddit advice/model-generated advice

# Total: 1200 data points

import jsonlines
from data_collection.DataLoader import DataLoader

class AdviceHelpfullnessDataLoader(DataLoader):
    def get_comparison_data(self):
        # One data point:
            #   {
            #     situation: ,
            #     advice_one: ,
            #     advice_two: ,
            #     is_advice_two_better_than_one: 
            #   }
        data = []
        human_eval_results_file = 'data_collection/advice_generation/data/feb-14-2020-workerids.jsonl'
        f = jsonlines.open(human_eval_results_file)
        for line in f.iter():
            situation = line["situation"]["title"] + line["situation"]["selftext"]
            reddit_advice = line["best_advice"]["bestadvice_body"]
            for generating_model in line["model_advice"]:
                data.append({"situation": situation, "advice_one": reddit_advice, "advice_two":line["model_advice"][generating_model], "is_advice_two_better_than_one":line["turk_ratings"][generating_model]["is_preferred"]})
        
        return data


    def concatenate_data(self):
        # One data point:
            #   {
            #     good_sample: ,
            #     bad_sample: 
            #   }
        data = self.get_comparison_data()
        concat_data = []

        for data_point in data:
            situation = data_point["situation"]

            if not data_point["is_advice_two_better_than_one"]:
                good_advice = data_point["advice_one"]
                bad_advice = data_point["advice_two"]
            if data_point["is_advice_two_better_than_one"]:
                good_advice = data_point["advice_two"]
                bad_advice = data_point["advice_one"]

            if self.include_tags:
                good_sample = "situation" + self.sep_token + situation + self.sep_token + "advice" + self.sep_token + good_advice
                bad_sample = "situation" + self.sep_token + situation + self.sep_token + "advice" + self.sep_token + bad_advice
            else:
                good_sample = situation + self.sep_token + good_advice
                bad_sample = situation + self.sep_token + bad_advice

            concat_data.append({"good_sample":good_sample,"bad_sample":bad_sample})

        return concat_data