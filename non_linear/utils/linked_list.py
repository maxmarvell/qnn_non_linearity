class Node: 
    def __init__(self, layer):
        self.layer = layer
        self.outputs = None
        self.next = None
        self.fisher_information = None
        self.quantum_fisher_information = None

    def __repr__(self):
        return self.data
    

class NNLinkedList:
    def __init__(self):
        self.head = None