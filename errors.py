"""
Created on Sun Mar 26 07:56:34 PM CEST 2023

@file: errors.py

@author: Yoquec
"""


# 🚨 custom errors -------------------------------------------------------------------
class CriterionAlreadyAssignedError(Warning):
    def __init__(self, message: str):
        super().__init__(message)


class NoCriterionAssignedError(Warning):
    def __init__(self, message: str):
        super().__init__(message)


class OptimizerAlreadyAssignedError(Warning):
    def __init__(self, message: str):
        super().__init__(message)


class NoOptimizerAssignedError(Warning):
    def __init__(self, message: str):
        super().__init__(message)


class CodedLayerAlreadyAssignedError(Warning):
    def __init__(self, message: str):
        super().__init__(message)


class NoCodedLayerAssignedError(Warning):
    def __init__(self, message: str):
        super().__init__(message)
