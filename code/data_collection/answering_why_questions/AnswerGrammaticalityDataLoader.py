# Data from paper: TellMeWhy: A Dataset for Answering Why-Questions in Narratives
# https://aclanthology.org/2021.findings-acl.53v2.pdf

# Task 1: Answer Grammaticality

# Annotators are presented a story, a related question,
# and the three answers that were collected
# Annotators then rated the grammaticality and validity (whether the provided
#  answer is a plausible answer for the question) on a 5 point
#  Likert scale.

# 598 Comparison Data points

import csv
import math
from collections import defaultdict
import json
from data_collection.DataLoader import DataLoader

class AnswerGrammaticalityDataLoader(DataLoader):
    def get_data(self):
        data = []
        human_eval_results_file = 'data_collection/answering_why_questions/data/test_caters_subset.csv'
        with open(human_eval_results_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0: continue
                data.append({"narrative":row[0], "question":row[1], "answer":row[2], "is_question_answerable":row[3], "val_ann":row[10], "gramm_ann":row[11]})
        return data

    def get_comparison_data(self):
        # One data point:
            #   {
            #     narrative: ,
            #     question: ,
            #     more_plausible_answer: ,
            #     less_plausible_answer: 
            #   }
            data = self.get_data()
            
            questions = defaultdict(list)

            for data_i in data:
                narrative_and_question = data_i["narrative"] + self.sep_token + data_i["question"]
                questions[narrative_and_question].append(data_i)
            
            comparison_data = []
            for datasets in questions.values():
                first_answer,second_answer,third_answer = datasets[0]["answer"],datasets[1]["answer"],datasets[2]["answer"]
                first_answer_gram = self.get_overall_ann(json.loads(datasets[0]["gramm_ann"]))
                second_answer_gram = self.get_overall_ann(json.loads(datasets[1]["gramm_ann"]))
                third_answer_gram = self.get_overall_ann(json.loads(datasets[2]["gramm_ann"]))

                if first_answer_gram > second_answer_gram:
                    comparison_data.append({
                            "narrative": datasets[0]["narrative"],"question": datasets[0]["question"],
                            "more_grammatical_answer": first_answer,
                            "less_grammatical_answer": second_answer,
                            "more_grammatical_answer_rank": first_answer_gram,
                            "less_grammatical_answer_rank": second_answer_gram
                        })
                if first_answer_gram > third_answer_gram:
                    comparison_data.append({
                            "narrative": datasets[0]["narrative"],"question": datasets[0]["question"],
                            "more_grammatical_answer": first_answer,
                            "less_grammatical_answer": third_answer,
                            "more_grammatical_answer_rank": first_answer_gram,
                            "less_grammatical_answer_rank": third_answer_gram
                        })

                if second_answer_gram > first_answer_gram:
                    comparison_data.append({
                            "narrative": datasets[0]["narrative"],"question": datasets[0]["question"],
                            "more_grammatical_answer": second_answer,
                            "less_grammatical_answer": first_answer,
                            "more_grammatical_answer_rank": second_answer_gram,
                            "less_grammatical_answer_rank": first_answer_gram
                        })
                if second_answer_gram > third_answer_gram:
                    comparison_data.append({
                            "narrative": datasets[0]["narrative"],"question": datasets[0]["question"],
                            "more_grammatical_answer": second_answer,
                            "less_grammatical_answer": third_answer,
                            "more_grammatical_answer_rank": second_answer_gram,
                            "less_grammatical_answer_rank": third_answer_gram
                        })

                if third_answer_gram > first_answer_gram:
                    comparison_data.append({
                            "narrative": datasets[0]["narrative"],"question": datasets[0]["question"],
                            "more_grammatical_answer": third_answer,
                            "less_grammatical_answer": first_answer,
                            "more_grammatical_answer_rank": third_answer_gram,
                            "less_grammatical_answer_rank": first_answer_gram
                        })
                if third_answer_gram > second_answer_gram:
                    comparison_data.append({
                            "narrative": datasets[0]["narrative"],"question": datasets[0]["question"],
                            "more_grammatical_answer": third_answer,
                            "less_grammatical_answer": second_answer,
                            "more_grammatical_answer_rank": third_answer_gram,
                            "less_grammatical_answer_rank": second_answer_gram
                        })
            return comparison_data    

    def concatenate_data(self):       
        data = self.get_comparison_data()
        concat_data = []

        for data_point in data:
            narrative = data_point["narrative"]
            question = data_point["question"]
            more_grammatical_answer = data_point["more_grammatical_answer"]
            less_grammatical_answer = data_point["less_grammatical_answer"]
            more_grammatical_answer_rank = data_point["more_grammatical_answer_rank"]
            less_grammatical_answer_rank = data_point["less_grammatical_answer_rank"]

            if self.include_tags:
                good_sample = "narrative" + self.sep_token + narrative + self.sep_token + "question" + self.sep_token + question + self.sep_token + "answer" + self.sep_token + more_grammatical_answer
                bad_sample = "narrative" + self.sep_token + narrative + self.sep_token + "question" + self.sep_token + question + self.sep_token + "answer" + self.sep_token + less_grammatical_answer
            else:
                good_sample = narrative + self.sep_token + question + self.sep_token + more_grammatical_answer
                bad_sample = narrative + self.sep_token + question + self.sep_token + less_grammatical_answer
            
            concat_data.append({"good_sample":good_sample,"bad_sample":bad_sample,"good_rank":more_grammatical_answer_rank,"bad_rank":less_grammatical_answer_rank})

        return concat_data