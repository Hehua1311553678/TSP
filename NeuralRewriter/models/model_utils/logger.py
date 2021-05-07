# -*- coding: utf-8 -*-
# @Time    : 2021/4/25 20:20
# @Author  : Hua He
# @Email   : hehua@nuaa.edu.cn
# @File    : logger.py
# @Software: PyCharm
import numpy as np
import argparse
import sys
import os
import re
import json
import pandas as pd


class Logger(object):
	"""
	The class for recording the training process.
	"""
	def __init__(self, args):
		self.log_interval = args.log_interval
		self.log_name = "./logs/" + args.log_name
		self.best_reward = 0
		self.records = []
		if not os.path.exists("./logs/"):
			os.makedirs("./logs/")


	def write_summary(self, summary):
		print("global-step: %(global_step)d, avg-reward: %(avg_reward).3f" % summary)
		self.records.append(summary)
		df = pd.DataFrame(self.records)
		df.to_csv(self.log_name, index=False)
		self.best_reward = max(self.best_reward, summary['avg_reward'])
