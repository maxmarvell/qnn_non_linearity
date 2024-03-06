from non_linear.fisher import FisherInformation
from non_linear.qfisher import QuantumFisherInformation
from numpy import ndarray

class LearnModelData:
    def __init__(self, 
                 fisher_information:FisherInformation = None, 
                 quantum_fisher_information:QuantumFisherInformation = None,
                 y_pred:ndarray = None):
        
        self.fisher_information = fisher_information
        self.quantum_fisher_information = quantum_fisher_information
        self.y_pred = y_pred



class Node(LearnModelData):
    def __init__(self, layer):
        self.layer = layer
        self.outputs = None
        self.next = None

    def __repr__(self):
        return self.data
    

class NNLinkedList:
    def __init__(self):
        self.head = None