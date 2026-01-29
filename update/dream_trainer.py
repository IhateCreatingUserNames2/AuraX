import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset
import os

from neural_physics import StatePredictor, ActionGenerator

# Caminhos para salvar os modelos treinados
MODEL_PATH = "./aura_brain/"
os.makedirs(MODEL_PATH, exist_ok=True)


class DreamMachine:
    def __init__(self, state_dim=144, action_dim=128):
        self.forward_model = StatePredictor(state_dim, action_dim)
        self.inverse_model = ActionGenerator(state_dim, action_dim)

        # Otimizadores separados
        self.opt_fwd = optim.AdamW(self.forward_model.parameters(), lr=1e-3)
        self.opt_inv = optim.AdamW(self.inverse_model.parameters(), lr=1e-3)

        # Loss Function: MSE (Mean Squared Error) é ideal para vetores contínuos
        self.criterion = nn.MSELoss()

    def load_brains(self):
        """Carrega pesos existentes se houver"""
        try:
            self.forward_model.load_state_dict(torch.load(f"{MODEL_PATH}forward.pth"))
            self.inverse_model.load_state_dict(torch.load(f"{MODEL_PATH}inverse.pth"))
            print("🧠 Cérebros carregados com sucesso.")
        except FileNotFoundError:
            print("🌱 Nenhum cérebro encontrado. Iniciando do zero (Tabula Rasa).")

    def train_cycle(self, memory_logs, epochs=50):
        """
        O Processo de 'Sonhar'.
        memory_logs: lista de tuplas (state_t, action_t, state_next)
        """
        if not memory_logs:
            print("💤 Nada para sonhar hoje.")
            return

        # Converter logs para Tensores PyTorch
        states = torch.tensor([m[0] for m in memory_logs], dtype=torch.float32)
        actions = torch.tensor([m[1] for m in memory_logs], dtype=torch.float32)
        next_states = torch.tensor([m[2] for m in memory_logs], dtype=torch.float32)

        dataset = TensorDataset(states, actions, next_states)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        print(f"🌙 Iniciando ciclo REM (Training {epochs} epochs)...")
        self.forward_model.train()
        self.inverse_model.train()

        for epoch in range(epochs):
            total_loss_fwd = 0
            total_loss_inv = 0

            for s, a, s_next in loader:
                # --- Treino do Forward Model ---
                # A rede tenta adivinhar o próximo estado
                self.opt_fwd.zero_grad()
                predicted_delta = self.forward_model(s, a)
                target_delta = s_next - s  # O que realmente aconteceu

                loss_f = self.criterion(predicted_delta, target_delta)
                loss_f.backward()
                self.opt_fwd.step()
                total_loss_fwd += loss_f.item()

                # --- Treino do Inverse Model ---
                # A rede tenta adivinhar qual ação causou a mudança
                self.opt_inv.zero_grad()
                predicted_action = self.inverse_model(s, s_next)

                loss_i = self.criterion(predicted_action, a)
                loss_i.backward()
                self.opt_inv.step()
                total_loss_inv += loss_i.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Fwd Loss {total_loss_fwd:.4f} | Inv Loss {total_loss_inv:.4f}")

        # Salvar o aprendizado (Consolidação de Memória)
        torch.save(self.forward_model.state_dict(), f"{MODEL_PATH}forward.pth")
        torch.save(self.inverse_model.state_dict(), f"{MODEL_PATH}inverse.pth")
        print("☀️ Acordando... Aprendizado consolidado.")