from abc import ABC, abstractmethod
import math

class DataLoader(ABC):
    def __init__(self,tokenizer,instructions, truncate_right, include_tags):
        self.tokenizer = tokenizer
        self.instructions = instructions
        self.truncate_right = truncate_right
        self.include_tags = include_tags
        
        self.sep_token = ' sep_token '
    
    @abstractmethod
    def concatenate_data(self):
        # One data point:
            # {
            #     good_sample: "",
            #     bad_sample: ""
            # }
        pass
    
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
        train_tokenized_data = tokenized_data[:math.floor(n*0.8)]
        eval_tokenized_data = tokenized_data[math.floor(n*0.8):math.floor(n*0.9)]
        test_tokenized_data = tokenized_data[math.floor(n*0.9):]

        return train_tokenized_data,eval_tokenized_data,test_tokenized_data
    
    def get_overall_ann(self,annotation):
        return round(sum(annotation)//len(annotation))