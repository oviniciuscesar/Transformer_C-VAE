from typing import Optional
import torch
import torch.nn as nn
from TransformerModel import TransformerModel
import os
from typing import List


#directories
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(DIRECTORY, "TorchScript")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "Checkpoints")
PLOTS_DIR = os.path.join(DIRECTORY, "plots")
DATASETS_DIR = os.path.join(DIRECTORY, "dataset")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
 

#-- transformer wrapper --
class TransformerWrapper(nn.Module):
    """
    Transformer wrapper compatible with torch.ts (seq-to-seq) 
    inputs:
        - transformer: nn.Module
        - max_length: int
            Comprimento máximo das sequências geradas
    """
    def __init__(self, transformer: nn.Module, max_length: int = 50, frames: int = 10, input_features: int = 8, target_features: int = 5) -> None:
        super().__init__()
        self.transformer = transformer
        self.max_length = int(max_length)
        self.input_features = int(input_features)
        self.target_features = int(target_features)
        self.frames = int(frames)
        self._methods = ["forward", "steps"]
        self._attributes = ["forward_input_shape", "forward_output_shape", "steps"]
        # buffers para compatibilidade com torchscript
        self.register_buffer("max_input_length", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("max_output_length", torch.tensor(int(self.max_length*self.target_features), dtype=torch.int32))
         # shapes
        self.forward_input_shape = [self.frames*self.input_features]
        self.forward_output_shape = [self.max_length*self.target_features]

        # search for final projection layer and assign to self.final_projection
        proj = None
        if hasattr(self.transformer, "final_projection"):
            proj = getattr(self.transformer, "final_projection")
        if proj is None:
            raise ValueError("Transformer model does not have a recognized final projection layer.")
        self.final_projection = proj

    @torch.jit.export
    def get_methods(self) -> List[str]:
        """Retorna lista de métodos disponíveis"""
        return self._methods

    @torch.jit.export
    def get_attributes(self) -> List[str]:
        """Retorna lista de atributos disponíveis"""
        return self._attributes
    
    @torch.jit.export
    def steps(self, new_max: float):
        """
        set max_length dynamically
        Args:
            new_max: max_length <= MAX_POS
        update the max_output_length buffer and the forward_output_shape.
        """
        if new_max <= 0:
            return
        nm: int = int(new_max)
        
        self.max_length = nm
        # update the max_output_length buffer (scalar tensor)
        self.max_output_length.fill_(int(self.max_length*self.target_features))
        print(f"Updated max_length to {self.max_length}")
        

        # update forward_output_shape (list with a single element)
        new_val = int(self.max_length * self.target_features)
        if len(self.forward_output_shape) > 0:
            self.forward_output_shape[0] = new_val
        else:
            self.forward_output_shape.append(new_val)

    def generate(self, src: torch.Tensor, start_vectors: torch.Tensor, max_length: int, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Gera sequência de vetores completando `start_vectors` até `max_len`.
        Args:
            src: (batch, frames, input features) features FloatTensor
            start_vectors: (batch, initial frame, output features) initial sequence (L0 >= 1).
            max_len: total desired length (>= L0).
            src_mask: optional mask for the encoder.
        Returns:
            out_vectors: FloatTensor (batch, max_len, output features)
        """
        B = src.size(0)
        L0 = start_vectors.size(1)
        F_out = self.target_features
        device = src.device

        if max_length <= 0:
            return torch.empty(B, 0, F_out, dtype=src.dtype, device=device)

        # Se 'start' já for maior que 'max_length', truncar
        if L0 >= max_length:
            return start_vectors[:, :max_length, :].contiguous()

        # 1. Encoder (runs once)
        enc = self.transformer.encoder(src, src_mask)  # (B, Ls, d_model)

        # 2. Buffer de saída
        out_vectors = torch.zeros(B, max_length, F_out, dtype=src.dtype, device=device)
        out_vectors[:, :L0, :] = start_vectors

        # 3. Geração auto-regressiva passo-a-passo
        t = L0
        while t < max_length:
            # Alimentar decoder com vetores atuais (0..t-1)
            tgt_in = out_vectors[:, :t, :]  # (B, t, F_out)
            
            # Criar máscara causal (t x t)
            causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))
            causal_b = causal.unsqueeze(0).expand(B, -1, -1)  # (B,t,t)

            # Executa o modelo
            dec = self.transformer.decoder(tgt_in, enc, causal_b, None) # (batch, t, d_model)
            predictions = self.transformer.final_projection(dec)       # (batch, t, F_out)

            # Pega o ÚLTIMO vetor previsto
            next_vector = predictions[:, -1, :]  # (batch, F_out)

            # Salva o vetor previsto no buffer
            out_vectors[:, t, :] = next_vector
            t += 1
        return out_vectors

    def generate_from_src(
        self, src: torch.Tensor, max_length: int, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convenience: Generates a sequence of 'max_len'
        given *only* the 'src' (flute).
        Starts the decoder generation with a "start vector" (a zero vector).
        The 'max_length' here is the length of the *output*, not the total.
        Args:
            src: (B, Ls, F_in) FloatTensor com features de contexto (flauta).
            max_length: comprimento da sequência de *saída* desejada.
        Returns:
            out_vectors: FloatTensor (batch, max_length, output features)
        """
        B = src.size(0)
        F_out = self.target_features
        device = src.device

        # Cria o vetor de "início" (L0 = 1). 
        # Usamos um vetor de zeros como primeiro "evento" de entrada do decoder.
        start_vector = torch.zeros((B, 1, F_out), dtype=src.dtype, device=device)

        # Gera 'max_length + 1' passos e descarta o primeiro (o 'start_vector')
        # O 'generate' retorna (batch, max_length + 1, output features)
        full_seq = self.generate(src, start_vector, max_length + 1, src_mask)

        # Retorna apenas os passos gerados (ignora o 'start_vector' de entrada)
        return full_seq[:, 1:, :].contiguous() # (batch, max_length, output features)

    @torch.jit.export
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        input:
          - flattened tensor without batch: (N) where N == frames * input_features
          - reshaped to (batch, frames, input_features)
        Returns generated sequence:
            - flattened tensor: (M) where M == frames * target_features
        """
    
        # reshape input to (1, frames, input_features)
        src = src.view(self.frames, self.input_features)
        src = src.unsqueeze(0)  # (batch, frames, input_features) [1, frames, input_features]
        
        # generate output sequence
        out = self.generate_from_src(src, self.max_length, None)

        # reshape output to (M) and return
        out = out.squeeze(0) 
        out = out.flatten()
        return out



if __name__ == "__main__": 
    # 1. Parâmetros do modelo (devem ser os mesmos de Train_Regression.py)
    INPUT_FEATURES = 64
    TARGET_FEATURES = 5
    SEQ_LEN = 10
    MAX_POS = 100
    
    # 2. Carrega a arquitetura do modelo
    model = TransformerModel(
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_ff=128,
        input_features=INPUT_FEATURES,
        target_features=TARGET_FEATURES,
        max_pos=MAX_POS,
        dropout=0.1,
    )
    
    # # 3. Carrega os pesos treinados (do Passo 2)
    weights_path = CHECKPOINT_DIR + "/model.pt"

    model.load_state_dict(torch.load(weights_path))
    model.eval()

    scripted_model = torch.jit.script(model)
    wrapper = TransformerWrapper(scripted_model, max_length=MAX_POS, frames=SEQ_LEN, input_features=INPUT_FEATURES, target_features=TARGET_FEATURES)
    scripted_wrapper = torch.jit.script(wrapper)

    output_path = os.path.join(MODEL_DIR, "transformer.ts")
    scripted_wrapper.save(output_path)
    print(f"Torchscript saved in {output_path}")

    # 5. Teste de Geração
    print("\nIniciando teste de geração...")

    # carrega o modelo torchscript salvo
    loaded_model = torch.jit.load(output_path)
    loaded_model.eval()

    # Cria um input 'src' fictício (batch=1, Ls=10, F_in=4) que simula 10 frames 
    dummy = torch.randn(1, SEQ_LEN, INPUT_FEATURES)
    
    print("Métodos registrados:", loaded_model.get_methods())
    print("Atributos registrados:", loaded_model.get_attributes())

    print("------ teste com 10 passos ------")
    flat = dummy.view(-1)  # (frames * input_features,)
    with torch.no_grad():
        loaded_model.steps(10)  # set max_length dynamically
        out_flat = loaded_model.forward(flat)  # retorna vetor achatado (SEQ_LEN * TARGET_FEATURES)
    print("\nsaida (achatada) length:", out_flat.numel())
    # reconstruir forma 3D
    # out3 = out_flat.view(1, SEQ_LEN, TARGET_FEATURES)
    n_frames_out = int(loaded_model.max_output_length.item()) // TARGET_FEATURES
    out3 = out_flat.view(1, n_frames_out, TARGET_FEATURES)

    print(f"\nShape do input (flauta): {dummy.shape}")
    print("\nsaida reconstruida shape:", out3.shape)

    # assert out3.shape == (1, SEQ_LEN, TARGET_FEATURES)
    
    print("\nTeste de geração concluído com sucesso!")
    print("\nExemplo de saída (primeiros 5 eventos):")
    print(out3[0, :5, :])


    print("--------- teste com 30 passos ---------")
    flat = dummy.view(-1)  # (frames * input_features,)
    with torch.no_grad():
        loaded_model.steps(30)  # set max_length dynamically
        out_flat2 = loaded_model.forward(flat)  # retorna vetor achatado (SEQ_LEN * TARGET_FEATURES)
    print("\nsaida (achatada) length:", out_flat2.numel())
    # reconstruir forma 3D
    n_frames_out2 = int(loaded_model.max_output_length.item()) // TARGET_FEATURES
    out3 = out_flat2.view(1, n_frames_out2, TARGET_FEATURES)

    print(f"\nShape do input (flauta): {dummy.shape}")
    print("\nsaida reconstruida shape:", out3.shape)

    # assert out3.shape == (1, SEQ_LEN, TARGET_FEATURES)
    
    print("\nTeste de geração concluído com sucesso!")
    print("\nExemplo de saída (primeiros 5 eventos):")
    print(out3[0, :5, :])