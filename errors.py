"""
Created on Sun Mar 26 07:56:34 PM CEST 2023

@file: errors.py

@author: Yoquec
"""


# 🚨 custom errors -------------------------------------------------------------------
class CriterionAlreadyAssignedError(Warning):
    def __init__(self, message: str):
        super().__init__(message)


class NoCriterionAssigned(Warning):
    def __init__(self, message: str):
        super().__init__(message)


class OptimizerAlreadyAssignedError(Warning):
    def __init__(self, message: str):
        super().__init__(message)


class NoOptimizerAssigned(Warning):
    def __init__(self, message: str):
        super().__init__(message)
