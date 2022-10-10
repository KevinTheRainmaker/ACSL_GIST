import torch

# Dense Ver.
class GCN_Layer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, X, normalized_tilde):
        out = self.Linear(X)
        out = normalized_tilde.matmul(out)

        return out


class GCN_Model(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes):
        super().__init__()
        self.GCN1 = GCN_Layer(feature_dim, hidden_dim)
        self.GCN2 = GCN_Layer(hidden_dim, num_classes)

        self.Activation = torch.nn.ReLU()

    def normalize(self, tilde):
        D1 = tilde.sum(dim=0)
        D2 = tilde.sum(dim=1)
        normalized_tilde = ((D1**(0.5)*tilde)*(D2**(0.5)))

        return normalized_tilde

    def forward(self, X, tilde):
        normalized_tilde = self.normalize(tilde)

        out = self.Activation(self.GCN1(X, normalized_tilde))
        out = self.GCN2(out, normalized_tilde)

        return out

# Sparse Ver.: Dense version은 연산량이 너무 많다
class GCN_Layer_sps(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, X, normalized_tilde):
        out = self.Linear(X)
        out = torch.spmm(normalized_tilde,out) # torch.spmm(sps_mat, out)

        return out


class GCN_Model_sps(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes):
        super().__init__()
        self.GCN1 = GCN_Layer_sps(feature_dim, hidden_dim)
        self.GCN2 = GCN_Layer_sps(hidden_dim, num_classes)

        self.Activation = torch.nn.ReLU()

    def normalize(self, tilde):
        D1 = tilde.sum(dim=0)
        D2 = tilde.sum(dim=1)
        normalized_tilde = ((D1**(0.5)*tilde)*(D2**(0.5)))

        return normalized_tilde

    def forward(self, X, tilde):
        normalized_tilde = self.normalize(tilde)
        normalized_tilde = normalized_tilde.to_sparse()
        
        out = self.Activation(self.GCN1(X, normalized_tilde))
        out = self.GCN2(out, normalized_tilde)

        return out



if __name__ == '__main__':

    # for reproducibility
    torch.manual_seed(1234)
    
    # configurations
    num_nodes = 6
    feature_dim = 3

    hidden_dim = 2
    num_classes = 4

    # create sample node feature
    X = torch.Tensor(torch.randn(num_nodes, feature_dim))

    # define sample adjacency matrix
    adj = torch.Tensor([[0, 1, 0, 0, 1, 0],
                        [1, 0, 1, 0, 1, 0],
                        [0, 1, 0, 1, 0, 0],
                        [0, 0, 1, 0, 1, 1],
                        [1, 1, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0, 0]])

    tilde = adj + torch.eye(adj.shape[0])

    # select Dense or Sparse
    is_dense = 0
    if is_dense:
        # build GCN
        GCN = GCN_Model(feature_dim, hidden_dim, num_classes)

        # test
        out = GCN(X, tilde)
        
        print('[result of Dense Version]')
        print(out)
        print(out.shape)
    else:    
        # Sparse Ver.
        GCN_sps = GCN_Model_sps(feature_dim, hidden_dim, num_classes)
        out_sps = GCN_sps(X, tilde)
        
        print('[result of Sparse Version]')
        print(out_sps)
        print(out_sps.shape)