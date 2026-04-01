import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel

class NRLMF:
    def __init__(self, c=5, gamma=1, lambda_d=1, lambda_t=1, r=10, alpha=0.01, beta=0.01, theta=0.01, max_iter=100):

        self.lambda_d = lambda_d
        self.lambda_t = lambda_t
        self.r = r
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter

    def fix_model(self, Y, Sd, St):
        self.Y = Y
        self.Sd = Sd
        self.St = St

        self.num_drugs, self.num_targets = Y.shape

        self.D = np.random.rand(self.num_drugs, self.r)
        self.T = np.random.rand(self.num_targets, self.r)

        for _ in range(self.max_iter):
            self.update_D()
            self.update_T()

    def update_D(self):
        TT = self.T.T @ self.T
        I = np.eye(self.r)

        for i in range(self.num_drugs):
            sim_term = self.alpha * np.sum(
                [self.Sd[i, k] * self.D[k] for k in range(self.num_drugs)],
                axis=0
            )

            A = TT + self.lambda_d * I
            b = self.Y[i] @ self.T + sim_term

            self.D[i] = np.linalg.solve(A, b)

    def update_T(self):
        DD = self.D.T @ self.D
        I = np.eye(self.r)

        for j in range(self.num_targets):
            sim_term = self.beta * np.sum(
                [self.St[j, k] * self.T[k] for k in range(self.num_targets)],
                axis=0
            )

            A = DD + self.lambda_t * I
            b = self.Y[:, j] @ self.D + sim_term

            self.T[j] = np.linalg.solve(A, b)

    def predict(self):
        return self.D @ self.T.T

def main():

    np.random.seed(42)

    #Load the protein, molecule, and interaction data
    protein_data = pd.read_csv('../../data/3-nrlmf/1-input_nrlmf/prot-simmilarity_bindingdb.csv', index_col=0)
    molecule_data = pd.read_parquet('../../data/3-nrlmf/1-input_nrlmf/mol-simmilarity_bindingdb.parquet',engine="pyarrow")
    interaction_data = pd.read_csv('../../data/3-nrlmf/1-input_nrlmf/matriks-chemogenomic_bindingdb.csv', index_col=0)

    #Convert data to numpy arrays
    Y = interaction_data.values.T

    molecule_data = molecule_data.loc[interaction_data.columns, interaction_data.columns]
    protein_data = protein_data.loc[interaction_data.index, interaction_data.index]

    Sd = molecule_data.values
    St = protein_data.values

    Sd = Sd / (Sd.max() + 1e-10)
    St = St / (St.max() + 1e-10)

    protein_ids = interaction_data.index.values   # target
    molecule_ids = interaction_data.columns.values  # drug

    print("Y shape:", Y.shape)
    print("Sd shape:", Sd.shape)
    print("St shape:", St.shape)

    #Define parameters for NRLMF
    params = {
        'c': 5,
        'gamma': 1,
        'lambda_d': 1,
        'lambda_t': 1,
        'r': 10,
        'alpha': 0.01,
        'beta': 0.01,
        'theta': 0.01,
        'max_iter': 100
    }

    #Initialize and train NRLMF model
    nrlmf_model = NRLMF(**params)
    nrlmf_model.fix_model(Y, Sd, St)

    #Perform predictions
    Y_pred = nrlmf_model.predict()
    print("Model training is complete")

    low_prob_pairs = []

    num_positive = int(np.sum(Y))
    neg_candidates = []

    for i, drug in enumerate(molecule_ids):
      for j, prot in enumerate(protein_ids):
        if Y[i, j] == 0:
            neg_candidates.append((drug, prot, Y_pred[i, j]))
    
    neg_df = pd.DataFrame(neg_candidates, columns=['Drug', 'UniProt ID', 'score'])
    neg_df = neg_df.sort_values(by='score', ascending=True)

    #1:1
    low_prob_pairs_df = neg_df.head(num_positive).reset_index(drop=True)
    
    # Create a DataFrame with protein-molecule pairs and their predicted scores
    ## 0=protein, 1=molecule
    low_prob_pairs_df  = pd.DataFrame(
        low_prob_pairs,
        columns=['UniProt ID', 'Drug', 'score']
    )
    print("Hard negative sampling completed")

    low_prob_pairs_df.to_csv("../../data/3-nrlmf/2-output_nrlmf/neg_inter_bindingdb.csv", index=False)
    print("File saved successfully")
