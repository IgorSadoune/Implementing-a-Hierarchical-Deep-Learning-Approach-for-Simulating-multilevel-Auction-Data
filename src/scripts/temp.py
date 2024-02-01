class SVR(torch.nn.Module):
    '''
    '''
    def __init__(self, input_dim):
        super(SVR, self).__init__()
        '''
        '''

        self.input_dim = input_dim
        self.model = torch.nn.Linear(self.input_dim, 2)

    def outlayer(self, x):
        '''
        desc: creates appropriate output activation for Guassian netwrok
        ----
        input: x: output from previous layer -> torch.tensor(prev_layer_in, prev_layer_out)
        -----
        output: out -> torch.tensor(2,1)
        '''
        #get dim of input
        dim_ = len(x.size())

        #separate parameters
        mu, sigma = torch.unbind(x, dim=1)

        #add one dimension to make the right shape
        mu = torch.unsqueeze(mu, dim=1)
        sigma = torch.unsqueeze(sigma, dim=1)
        
        #relu to sigma bacause variance is positive
        sigma = torch.nn.functional.relu(sigma)

        return torch.cat((mu, sigma), dim=dim_-1)

    def forward(self, x):

        return self.outlayer(self.model(x))