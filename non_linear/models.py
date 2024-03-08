import pennylane as qml
import jax


def variational_ansatz(n_features:int, n_layers:int):

    dev = qml.device("default.qubit.jax", wires=n_features)

    def ansatz(i,weights):
        N = len(weights)

        if (i % 3) == 1:
            for j in range(N):
                qml.RX(weights[j], wires=j)
                for j in range(N-1):
                    qml.CNOT(wires=[j, j+1])
                    qml.CNOT(wires=[N-1, 0])

        elif (i % 3) == 2:
            for j in range(N):
                qml.RZ(weights[j], wires=j)
                for j in range(N//2):
                    qml.CNOT(wires=[j,N//2+j])
                for j in range(N//2+1):
                    qml.CNOT(wires=[N-1,j])

        else:
            for j in range(N):
                qml.RX(weights[j], wires=j)
                qml.CNOT(wires=[0, N-1])
                for j in range(N-1):
                    qml.CNOT(wires=[j+1, j])


    def model(inputs, weights):
        ansatz(0,weights[0])
        for i in range(1,n_layers+1):
            qml.AngleEmbedding(inputs, wires=range(n_features))
            ansatz(i,weights[i])

    return model, (n_layers+1,n_features), dev


def simple_ansatz(n_features:int, n_layers:int):

    dev = qml.device("default.qubit.jax", wires=n_features)

    def model(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_features))
        for i in range(n_layers):
            qml.BasicEntanglerLayers(weights[i:i+1], wires=range(n_features))
    
    return model, qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_features), dev


def data_reupload(n_features:int, n_layers:int):

    dev = qml.device("default.qubit.jax", wires=n_features)

    def model(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_features))
        for i in range(n_layers):
            qml.StronglyEntanglingLayers(weights[i:i+1], wires=range(n_features))
            qml.AngleEmbedding(inputs, wires=range(n_features))
    
    return model, qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_features), dev

def strongly_entangling_ansatz(n_features:int, n_layers:int):

    dev = qml.device("default.qubit.jax", wires=n_features)

    def model(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_features))
        for i in range(n_layers):
            qml.StronglyEntanglingLayers(weights[i:i+1], wires=range(n_features))
    
    return model, qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_features), dev


def mid_measure(n_features:int, n_layers:int, n_repitions:int):

    assert (n_repitions % 2) != 0

    wires=n_features+n_repitions//2

    dev = qml.device("default.qubit.jax", wires=wires)
    
    def repition(inputs,weights,i):
        if not (i % 2):
            qml.StronglyEntanglingLayers(weights[0:1], wires=range(n_features))
            for j in range(1,n_layers+1):
                qml.AngleEmbedding(inputs, wires=range(n_features))
                qml.StronglyEntanglingLayers(weights[j:j+1], wires=range(n_features))
            if i != n_repitions - 1:
                qml.measure(0, reset=True)

        else:
            qml.StronglyEntanglingLayers(weights[0:1,1:], wires=range(1,n_features))
            for j in range(1,n_layers+1):
                qml.AngleEmbedding(inputs[1:], wires=range(1,n_features))
                qml.StronglyEntanglingLayers(weights[j:j+1,1:], wires=range(1,n_features))
            
    def model(inputs, weights):
        for i in range(n_repitions):
            repition(inputs,weights[i],i)
            
    return model, (n_repitions,n_layers,n_features,3), dev



class qnn_compiler():
  
    def __init__(self, model, n_features:int, n_layers:int, target_length:int) -> None:
      self.model, self.parameter_shape, self.dev = model(n_features,n_layers)
      self.n_features = n_features
      self.n_layers = n_layers
      self.target_length = target_length
    

    def classification(self):

        @qml.qnode(self.dev,interface='jax')
        def qnn(inputs,weights):
            self.model(inputs, weights)
            return qml.expval(qml.PauliZ(wires=0)) if self.target_length == 1 else [qml.expval(qml.PauliZ(wires=i)) for i in range(self.target_length)]
        
        return jax.jit(qnn)
        

    def state(self):

        dev = self.dev
        dev.name = "default.qubit"
        @qml.qnode(dev)
        def qnn(inputs,weights):
            self.model(inputs, weights)
            return qml.state()
        
        return qnn
    

    def probabilites(self):

        @qml.qnode(self.dev,interface='jax')
        def qnn(inputs,weights):
            self.model(inputs, weights)
            return qml.probs(wires=[i for i in range(self.n_features)])
        
        return jax.jit(qnn)