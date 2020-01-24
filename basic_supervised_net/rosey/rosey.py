#!/usr/bin/env python

import sys
sys.path.append('/home/michael/Github/ai_training_notebooks/standard_utils')
from model_base import BaseSequentialModel


class Rosey(BaseSequentialModel):

    def __init__(self):
        super()