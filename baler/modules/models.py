import torch
from torch import nn
from torch.nn import functional as F


class george_SAE(nn.Module):
    def __init__(self, n_features, z_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # encoder
        self.en1 = nn.Linear(n_features, 200, dtype=torch.float64)
        self.en2 = nn.Linear(200, 100, dtype=torch.float64)
        self.en3 = nn.Linear(100, 50, dtype=torch.float64)
        self.en4 = nn.Linear(50, z_dim, dtype=torch.float64)
        # decoder
        self.de1 = nn.Linear(z_dim, 50, dtype=torch.float64)
        self.de2 = nn.Linear(50, 100, dtype=torch.float64)
        self.de3 = nn.Linear(100, 200, dtype=torch.float64)
        self.de4 = nn.Linear(200, n_features, dtype=torch.float64)

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        h1 = F.leaky_relu(self.en1(x))
        h2 = F.leaky_relu(self.en2(h1))
        h3 = F.leaky_relu(self.en3(h2))
        return self.en4(h3)

    def decode(self, z):
        h4 = F.leaky_relu(self.de1(z))
        h5 = F.leaky_relu(self.de2(h4))
        h6 = F.leaky_relu(self.de3(h5))
        out = self.de4(h6)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class george_SAE_BN(nn.Module):
    def __init__(self, n_features, z_dim):
        super(george_SAE_BN, self).__init__()

        # encoder
        self.enc_nn = nn.Sequential(
            nn.Linear(n_features, 200, dtype=torch.float64),
            # nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200, dtype=torch.float64),
            nn.Linear(200, 100, dtype=torch.float64),
            # nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100, dtype=torch.float64),
            nn.Linear(100, 50, dtype=torch.float64),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50, dtype=torch.float64),
            nn.Linear(50, z_dim, dtype=torch.float64),
            # nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(z_dim, dtype=torch.float64),
        )

        # decoder
        self.dec_nn = nn.Sequential(
            nn.Linear(z_dim, 50, dtype=torch.float64),
            # nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(50,dtype=torch.float64),
            nn.Linear(50, 100, dtype=torch.float64),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(100,dtype=torch.float64),
            nn.Linear(100, 200, dtype=torch.float64),
            # nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(200,dtype=torch.float64),
            nn.Linear(200, n_features, dtype=torch.float64),
            # nn.Dropout(p=0.5),
            # nn.BatchNorm1d(n_features,dtype=torch.float64),
            nn.ReLU(),
        )

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        out = self.enc_nn(x)
        return out

    def decode(self, z):
        out = self.dec_nn(z)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class george_SAE_Dropout_BN(nn.Module):
    def __init__(self, n_features, z_dim):
        super(george_SAE_Dropout_BN, self).__init__()

        # encoder
        self.enc_nn = nn.Sequential(
            nn.Linear(n_features, 200, dtype=torch.float64),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(200,dtype=torch.float64),
            nn.Linear(200, 100, dtype=torch.float64),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(100,dtype=torch.float64),
            nn.Linear(100, 50, dtype=torch.float64),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(50,dtype=torch.float64),
            nn.Linear(50, z_dim, dtype=torch.float64),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(z_dim,dtype=torch.float64)
        )

        # decoder
        self.dec_nn = nn.Sequential(
            nn.Linear(z_dim, 50, dtype=torch.float64),
            # nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50, dtype=torch.float64),
            nn.Linear(50, 100, dtype=torch.float64),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100, dtype=torch.float64),
            nn.Linear(100, 200, dtype=torch.float64),
            # nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200, dtype=torch.float64),
            nn.Linear(200, n_features, dtype=torch.float64),
            # nn.Dropout(p=0.5),
            nn.BatchNorm1d(n_features, dtype=torch.float64),
            nn.ReLU(),
        )

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        out = self.enc_nn(x)
        return out

    def decode(self, z):
        out = self.dec_nn(z)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class george_SAE_Dropout(nn.Module):
    def __init__(self, n_features, z_dim):
        super(george_SAE_Dropout, self).__init__()

        # encoder
        self.enc_nn = nn.Sequential(
            nn.Linear(n_features, 200, dtype=torch.float64),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Linear(200, 100, dtype=torch.float64),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            nn.Linear(100, 50, dtype=torch.float64),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Linear(50, z_dim, dtype=torch.float64),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
        )

        # decoder
        self.dec_nn = nn.Sequential(
            nn.Linear(z_dim, 50, dtype=torch.float64),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.Linear(50, 100, dtype=torch.float64),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Linear(100, 200, dtype=torch.float64),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            nn.Linear(200, n_features, dtype=torch.float64),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        out = self.enc_nn(x)
        return out

    def decode(self, z):
        out = self.dec_nn(z)
        return out

    def forward(self, x):
        # z = x.view(batch_size,a,b,c) ? What is this
        return self.decode(x)

    def loss(self, model_children, true_data, reconstructed_data, reg_param):
        mse = nn.MSELoss()
        mse_loss = mse(reconstructed_data, true_data)
        l1_loss = 0
        values = true_data
        for i in range(len(model_children)):
            values = F.relu((model_children[i](values)))
            l1_loss += torch.mean(torch.abs(values))
        loss = mse_loss + reg_param * l1_loss
        return loss


class Conv_AE(nn.Module):
        def __init__(self, n_features, z_dim, *args, **kwargs):
            super(Conv_AE, self).__init__(*args, **kwargs)
            self.q_z_mid_dim = 2000
            self.q_z_output_dim = 72128
            # Encoder
            # Conv Layers
            self.q_z_conv = nn.Sequential(
                  nn.Conv2d(1, 8, kernel_size=(2,5), stride=(1), padding=(1)),
                  #nn.BatchNorm2d(8),
                  nn.ReLU(),
                  nn.Conv2d(8, 16, kernel_size=(3), stride=(1), padding=(1)),
                  nn.BatchNorm2d(16),
                  nn.ReLU(),
                  nn.Conv2d(16, 32, kernel_size=(3), stride=(1), padding=(0)),
                  #nn.BatchNorm2d(32),
                  nn.ReLU()
                  ) 
            # Flatten
            self.flatten = nn.Flatten(start_dim=1)
            # Linear layers
            self.q_z_lin = nn.Sequential( 
                  nn.Linear(self.q_z_output_dim, self.q_z_mid_dim), 
                  nn.ReLU(), 
                  # nn.BatchNorm1d(self.q_z_output_dim),
                  nn.Linear(self.q_z_mid_dim, z_dim),
                  nn.ReLU(),
                  )

            # Decoder
            self.p_x_lin = nn.Sequential(
                  nn.Linear(z_dim, self.q_z_mid_dim),
                  nn.ReLU(),
                  # nn.BatchNorm1d(self.q_z_output_dim),
                  nn.Linear(self.q_z_mid_dim, self.q_z_output_dim),
                  nn.ReLU(),
                  # nn.BatchNorm1d(42720) 
                  )
            # Conv Layers
            self.p_x_conv = nn.Sequential(
                  nn.ConvTranspose2d(32, 16, kernel_size=(3), stride=(1), padding=(0)),
                  nn.BatchNorm2d(16),
                  nn.ReLU(),
                  nn.ConvTranspose2d(16, 8, kernel_size=(3), stride=(1), padding=(1)),
                  nn.BatchNorm2d(8),
                  nn.ReLU(),
                  nn.ConvTranspose2d(8, 1, kernel_size=(2,5), stride=(1), padding=(1))
                  ) 
    
        def encode(self, x):
            # Conv
            out = self.q_z_conv(x)
            # flatten
            out = self.flatten(out)
            #out = out.view(out.size(0), -1)
            # dense 
            out = self.q_z_lin(out)
            return out
    
        def decode(self, z):
            # dense
            out = self.p_x_lin(z)
            # reshape
            out = out.view(out.size(0), 32, 49, 46)  
            # DeConv/UnConv?
            out = self.p_x_conv(out)
            return out
        
        def forward(self, x):
            z = self.encode(x)
            out = self.decode(z)
            return out
