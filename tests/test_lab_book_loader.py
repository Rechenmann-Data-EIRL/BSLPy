# -*- coding: utf-8 -*-
import unittest
import datetime

from src.bsl_python.lab_book_loader import LabBookLoader


class TestStringMethods(unittest.TestCase):
    def test_load_notebook(self):
        filepath = 'C:/Users/jujud/Documents/Consulting/Data'
        filename = 'Labbook_191128EM.xlsx'
        loader = LabBookLoader()
        information = loader.load_notebook(filepath, filename)
        self.assertEqual(191128, information["Experiment"]["Date"])
        self.assertEqual("EM", information["Experiment"]["Experimenter"])
        self.assertEqual("191128EM", information["Experiment"]["ID"])
        self.assertEqual(10, information["Mouse"]["Age"])
        self.assertEqual(datetime.datetime(2019,9,16), information["Mouse"]["DateOfBirth"])
        self.assertEqual(21.2, information["Mouse"]["Weight"])
        self.assertEqual("C57BL/6JRj", information["Mouse"]["Strain"])
        self.assertEqual("F", information["Mouse"]["Gender"])
        self.assertEqual(["KX", "BL", "K", "K", "BL", "K", "K", "K", "K", "K", "K"], information["Anaesthesia"]["What"])
        self.assertEqual([0.2, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                         information["Anaesthesia"]["Quantity"])
        self.assertEqual(
            ["09:37", "09:50", "10:14", "10:45", "11:40", "11:48", "12:29", "13:00", "14:00", "14:46", "15:16"],
            information["Anaesthesia"]["When"])
        self.assertEqual("A4x16-5mm-50-200-177-A64", information["Electrophy"]["Electrode Type"])
        self.assertEqual("L054", information["Electrophy"]["Electrode ID"])
        self.assertEqual([1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2], information["Trials"]["Penetration #"])
        self.assertEqual(list(range(1, 13)), information["Trials"]["Block"])
        self.assertEqual(
            ["ToneCoarse", "ToneFine", "ToneTrain", "CIstim", "CIstim", "CItrain", "CItrain", "CItrain", "CItrain",
             "CItrain", "CItrain", "ToneFine"], information["Trials"]["StimulusSet"])
        self.assertEqual(["10:56", "11:02", "11:21", "14:01", "14:20", "14:42", "14:50", "15:00", "15:09", "16:13",
                          "16:20", "17:00"], information["Trials"]["Start time"])
        self.assertEqual(["AC", "AC"], information["Electrophy"]["Cortical region"])
        self.assertEqual([960, 966], information["Electrophy"]["Tip depth"])
        self.assertEqual([1, 1], information["Electrophy"]["ML coordinates"])
        self.assertEqual([1, 1], information["Electrophy"]["RC coordinates"])
        self.assertEqual(["10:53", "13:54"], information["Electrophy"]["Penetration time"])
        self.assertEqual("AC", information["Craniotomy"]["Brain Area"])
        self.assertEqual("", information["Craniotomy"]["Size"])
