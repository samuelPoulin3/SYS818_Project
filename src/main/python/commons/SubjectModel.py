#! /usr/bin/env python3

from commons import ScanModel

DEBUG = False

class SubjectModel():
    """
        The model of a subject. Contains the images and its informations. 
        The properties are also described in this PDF from OASIS:
        https://www.oasis-brains.org/files/oasis_cross-sectional_facts.pdf

        Properties:
            sid: String
                Subject ID. ex OAS1_xxxx_MRy where xxxx is a number from 0001 
                to 9999 and y is an increment number from the visit number for 
                the subject.
            gender: String
                Gender of the subject. Valid values are: Male, Female
            hand: String
                Handedness. Valid values are: Right, Left
            age: Int
                Age of the subject.
            educ: Int
                Education. Valid values are:
                1: less than high school grad.,
                2: high school grad.,
                3: some college,
                4: college grad.,
                5: beyond college.
            ses: Int
                Socioeconomic status
            mmse: Int
                Mini-Mental State Examination.

            cdr: Float
                Clinical Dementia Rating. Valid values are:
                0: nondemented
                0.5: very mild dementia
                1: mild dementia
                2: moderate dementia
                All participants with dementia (CDR >0) were diagnosed with probable AD.

            e_tiv: Float
                Estimated total intracranial volume in mm^3.

            n_wbv: Float
                Normalized whole brain volume.
            asf: Float
                Atlas scaling factor.
            delay:

            scans:

            
    """
    __slots__ = ('sid', 'gender', 'hand', 'age', 'educ', 'ses', 'mmse', 'cdr', 'e_tiv', 'n_wbv', 'asf', 'delay', 'scans')
    def __init__(self, *args, **kwargs):
        self.sid = None
        self.gender = None
        self.hand = None
        self.age = None
        self.educ = None
        self.ses = None
        self.mmse = None
        self.cdr = None
        self.e_tiv = None
        self.n_wbv = None
        self.asf = None
        self.delay = None
        self.scans = []

    def clone(self, clone_samples=False):
        subject = SubjectModel()
        subject.sid = self.sid
        subject.gender = self.gender
        subject.hand = self.hand
        subject.age = self.age
        subject.educ = self.educ
        subject.ses = self.ses
        subject.mmse = self.mmse
        subject.cdr = self.cdr
        subject.e_tiv = self.e_tiv
        subject.n_wbv = self.n_wbv
        subject.asf = self.asf
        subject.delay = self.delay

        if clone_samples:
            subject.scans = self.scans.copy()

        return subject
