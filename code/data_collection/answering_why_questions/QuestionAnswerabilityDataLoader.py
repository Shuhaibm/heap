# Data from paper: TellMeWhy: A Dataset for Answering Why-Questions in Narratives
# https://aclanthology.org/2021.findings-acl.53v2.pdf

# Task 3: Question Answerability

# Annotators were presented a story and 3 why questions related to it.
# For each question, they were asked to provide judgments about the
# comprehensibility of the question, and whether the narrative explicitly
# contained the answer. They were also asked to select the sentences
# from the narrative which influenced their answer, if available.

# 1917 Comparison Data points

import csv
from data_collection.DataLoader import DataLoader
from collections import defaultdict

class QuestionAnswerabilityDataLoader(DataLoader):
    def get_data(self):
        data = []
        human_eval_results_file = 'data_collection/answering_why_questions/data/test_caters_subset.csv'
        with open(human_eval_results_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0: continue
                data.append({"narrative":row[0], "question": row[1], "is_question_answerable": row[3]})
        return data


    def get_comparison_data(self):
        # One data point:
            #   {
            #     narrative: ,
            #     more_answerable_question: ,
            #     less_answerable_question:
            #   }
        data = self.get_data()

        narratives = defaultdict(list)
        for data_i in data:
            narratives[data_i["narrative"]].append(data_i)

        comparison_data = []
        for datasets in narratives.values():
            for i,data_i in enumerate(datasets):
                for data_j in datasets[i+1:]:
                    new_data = {"narrative":data_i["narrative"]}

                    if data_i["is_question_answerable"] ==  data_j["is_question_answerable"]:
                        continue
                    if data_i["is_question_answerable"] == "Answerable" and data_j["is_question_answerable"] == "Not Answerable":
                        new_data["more_answerable_question"] = data_i["question"]
                        new_data["less_answerable_question"] = data_j["question"]
                    if data_i["is_question_answerable"] == "Not Answerable" and data_j["is_question_answerable"] == "Answerable":
                        new_data["more_answerable_question"] = data_j["question"]
                        new_data["less_answerable_question"] = data_i["question"]
                    
                    comparison_data.append(new_data)
        
        return comparison_data

    
    def concatenate_data(self):
        data = self.get_comparison_data()
        concat_data = []

        for data_point in data:
            narrative = data_point["narrative"]
            more_answerable_question = data_point["more_answerable_question"]
            less_answerable_question = data_point["less_answerable_question"]

            if self.include_tags:
                good_sample = "narrative" + self.sep_token + narrative + self.sep_token + "question" + self.sep_token + more_answerable_question
                bad_sample = "narrative" + self.sep_token + narrative + self.sep_token + "question" + self.sep_token + less_answerable_question
            else:
                good_sample = narrative + self.sep_token + more_answerable_question
                bad_sample = narrative + self.sep_token + less_answerable_question

            concat_data.append({"good_sample":good_sample,"bad_sample":bad_sample})
        
        return concat_data