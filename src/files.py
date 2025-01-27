#%%
import os

import pandas as pd


# Simple representation of a file folder
# We only have one test file, and one validation file, those are the two last
class SimpleFiles():
    def __init__(self, root, split):
        self.root = root
        self.files = list(map(lambda x: root+"/"+x, os.listdir(self.root)))
       
    def get(self, count=-1):
        return self.files[:count]

    def get_train(self, count=-1):
        if count == -1:
            return self.files[:-2]
        else:
            return self.files[:min(len(self.files)-2, count)]

    def get_val(self):
        return [self.files[-2]]

    def get_test(self):
        return [self.files[-1]]



# This is not used anymore, this represent the full maestro dataset
# It reads injson file the train/test/val split, and used it
# It is also possible to only choose a specific year to restrict the data

class MAESTROFiles():
    """Provide filenames for the dataset"""
    def __init__(self, root, year=-1):
        """Init dataset by setting a root"""
        self.root = root
        self.data = pd.read_json(root + "/maestro-v2.0.0.json")
        self.year = year
    def get_train(self, count=-1):
        """Get songs tagged as train data"""
        train_df = self.data[self.data['split'] == 'train']
        if (self.year > 0): train_df = train_df[train_df['year'] == self.year]
        train  = list(train_df['audio_filename'])
        return train[:count]

    def get_test(self, count=-1):
        """Get songs tagged as test data"""
        test_df = self.data[self.data['split'] == 'test']
        if (self.year > 0): test_df = test_df[test_df['year'] == self.year]
        test  = list(test_df['audio_filename'])
        return test[:count]

    def get_validation(self, count=-1):
        """Get songs tagged as validation data"""
        validation_df = self.data[self.data['split'] == 'validation']
        if (self.year > 0): validation_df = validation_df[validation_df['year'] == self.year]
        validation  = list(validation_df['audio_filename'])
        return validation[:count]
    def get(self, count=-1):
        df = self.data
        if (self.year > 0): df = self.data[self.data['year'] == self.year]
        out = list(df['audio_filename'])
        return out
