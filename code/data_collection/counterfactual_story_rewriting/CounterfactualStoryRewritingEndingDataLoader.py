# Data from paper: Counterfactual Story Reasoning and Generation
# https://arxiv.org/pdf/1909.04076.pdf
# Total: 4400

# Annotators were provided with a five sentence story. A second story with the same PREMISE is also provided,
# but its second sentence has been changed to reflect a new event (the COUNTERFACTUAL). Finally, the rest of
# the story is rewritten with respect to this change (the REWRITTEN continuation).

# Answer 5 questions about the original story and the new story based on the counterfactual second sentence.
#   (1) Which rewritten continuation better keeps in mind the details provided in the PREMISE sentence? (answer.premise)
#   (2) Which rewritten continuation better keeps in mind the details provided in the COUNTERFACTUAL sentence? (answer.second)
#   (3) Which of the plots presented in the rewritten continuations are more related to the plot of the ORIGINAL ending? (answer.plot)
#   (4) Which rewritten continuation, on its own, outlines a more reasonable sequence of events? (answer.ending)
#   (5) Given these considerations, which rewritten continuation is more reasonable given the PREMISE and COUNTERFACTUAL sentence provided? (answer.counterfactual)

# Answer definitions
#   A: Rewritten A
#   B: Rewritten B
#   S: Both continuations are at least partially relevant to the PREMISE sentence, but there is little to no difference between the two continutation in terms of their relevance to the PREMISE sentence.
#  IR: Both continuations are irrelevant to the PREMISE sentence, making it difficult to evaluate this property in a comparative way
# - can treat S and IR as the same thing if I try to consider third case where both are equally as good

# Task 4: Select the rewritten continuation that outlines a more reasonable sequence of events (answer.ending)

import csv
from data_collection.DataLoader import DataLoader

class CounterfactualStoryRewritingEndingDataLoader(DataLoader):
    def get_comparison_data(self):
        data = []
        human_eval_results_files = ['data_collection/counterfactual_story_rewriting/data/test_comparison_eval.csv']
        for human_eval_results_file in human_eval_results_files:
            with open(human_eval_results_file) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for i, row_arr in enumerate(reader):
                    if i == 0: 
                        headers = row_arr
                        continue
                    row = {}
                    for i in range(len(row_arr)): row[headers[i]] = row_arr[i]
                    
                    if row["Answer.ending"] == "A":
                        data.append({
                            "premise":row["Input.X1"],"initial_second":row["Input.X2"],"original_end":row["Input.Y"],
                            "counterfactual":row["Input.XX2"],
                            "better_rewritten": row["Input.YY1"],
                            "worse_rewritten": row["Input.YY2"]
                        })
                    if row["Answer.ending"] == "B":
                        data.append({
                            "premise":row["Input.X1"],"initial_second":row["Input.X2"],"original_end":row["Input.Y"],
                            "counterfactual":row["Input.XX2"],
                            "better_rewritten": row["Input.YY2"],
                            "worse_rewritten": row["Input.YY1"]
                        })
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
            premise,initial_second,original_end,counterfactual = data_point["premise"],data_point["initial_second"],data_point["original_end"],data_point["counterfactual"]
            better_rewritten,worse_rewritten = data_point["better_rewritten"],data_point["worse_rewritten"]

            if self.include_tags:
                good_sample = "story" + self.sep_token + premise + counterfactual + self.sep_token + "ending" + self.sep_token + better_rewritten
                bad_sample = "story" + self.sep_token + premise + counterfactual + self.sep_token + "ending" + self.sep_token + worse_rewritten
            else:
                good_sample = premise + counterfactual + self.sep_token + better_rewritten
                bad_sample = premise + counterfactual + self.sep_token + worse_rewritten

            concat_data.append({"good_sample":good_sample,"bad_sample":bad_sample})

        return concat_data