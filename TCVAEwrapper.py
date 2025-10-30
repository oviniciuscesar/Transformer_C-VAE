from typing import Optional
import torch
import torch.nn as nn
from TCVAEmodel import TransformerCVAE 
import os
from typing import List


# directorios
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(DIRECTORY, "TorchScript")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "Checkpoints")
PLOTS_DIR = os.path.join(DIRECTORY, "plots")
DATASETS_DIR = os.path.join(DIRECTORY, "dataset")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
 

#-- Transformer C-VAE wrapper --
class TcvaeWrapper(nn.Module):
    """
    Transformer C-VAE wrapper compatível com torch.ts (seq-to-seq) 
    inputs:
        - transformer: nn.Module (TransformerCVAE)
        - max_length: int (número máximo de passos de saída)
        - frames: int (tamanho da sequência de entrada)
        - input_features: int (número de features de entrada)
        - target_features: int (número de features de saída)
        - latent_dim: int (dimensão do espaço latente)
    """
    def __init__(self, transformer: nn.Module, max_length: int = 50, frames: int = 10, input_features: int = 64, target_features: int = 5, latent_dim: int = 32) -> None:
        super().__init__()
        self.transformer = transformer # Transformer C-VAE
        self.max_length = int(max_length) # número máximo de passos de saída
        self.input_features = int(input_features) # número de features de entrada
        self.target_features = int(target_features) # número de features de saída
        self.frames = int(frames) # tamanho da sequência de entrada
        self.latent_dim = int(latent_dim) # dimensão do espaço latente
 
        # métodos e atributos registrados para compatibilidade com torch.ts
        self._methods = ["forward", "steps", "forwardz", "latent"]
        self._attributes = ["forward_input_shape", "forward_output_shape", "forwardz_input_shape", "forwardz_output_shape", "steps", "latent_dim"]
        
        # buffers
        self.register_buffer("max_input_length", torch.tensor(0, dtype=torch.int32)) # tamanho de entrada fixo (não usado)
        self.register_buffer("max_output_length", torch.tensor(int(self.max_length*self.target_features), dtype=torch.int32)) # tamanho máximo de saída
        self.register_buffer("latent_dim_buf", torch.tensor(self.latent_dim, dtype=torch.int32)) # dimensão do espaço latente
        self.register_buffer("z_buffer", torch.zeros(self.latent_dim, dtype=torch.float32))
        self.register_buffer("z", torch.tensor(0, dtype=torch.int32))
         
        # shapes de entrada/saída para torch.ts
        self.forward_input_shape = [self.frames*self.input_features] # entrada achatada [N] onde N == frames * input_features
        self.forward_output_shape = [self.max_length*self.target_features] # saída achatada [M] onde M == max_length * target_features
        self.forwardz_input_shape = [self.frames*self.input_features] # entrada achatada + z [N,K]
        self.forwardz_output_shape = [self.max_length*self.target_features] # saída achatada

        # final linear layer
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
        """set max_length dynamically"""
        if new_max <= 0:
            return
        nm: int = int(new_max)
        self.max_length = nm
        self.max_output_length.fill_(int(self.max_length*self.target_features))
        print(f"Updated max_length to {self.max_length}")
        
        new_val = int(self.max_length * self.target_features)
        if len(self.forward_output_shape) > 0:
            self.forward_output_shape[0] = new_val
        else:
            self.forward_output_shape.append(new_val)

    @torch.jit.export
    def latent(self, z_in: torch.Tensor) -> torch.Tensor:
        """recebe z externamente e armazena no buffer"""
        if z_in.dim() == 2 and z_in.size(0) == 1:
            z_in = z_in.squeeze(0)
        if z_in.numel() != self.latent_dim:
            raise RuntimeError("z length mismatch")
        # copia valores para o buffer
        self.z_buffer.copy_(z_in)
        # marca que z foi definido
        self.z.fill_(1)
        return self.z_buffer

    def generate(self, enc_out: torch.Tensor, z: torch.Tensor, start_vectors: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Loop de geração auto-regressivo (lógica interna).
        Recebe o contexto 'enc_out' e o latente 'z' pré-computados.
        Args:
            enc_out: (batch, Ls, d_model) Saída do conditional_encoder.
            z: (batch, latent_dim) Vetor latente (amostrado ou enviado externamente).
            start_vectors: (batch, L0, F_out) Vetor inicial (zeros).
            max_length: número de passos a gerar
        Returns:
            out_vectors: FloatTensor (batch, max_length, features de saída) = predições geradas
        """
        B = enc_out.size(0) # batch size do contexto
        L0 = start_vectors.size(1) # comprimento do vetor inicial
        F_out = self.target_features # número de features de saída
        device = enc_out.device # dispositivo

        # se max_length <= 0, retorna tensor vazio
        if max_length <= 0:
            return torch.empty(B, 0, F_out, dtype=enc_out.dtype, device=device)
        # se L0 >= max_length, retorna apenas o start_vector cortado
        if L0 >= max_length:
            return start_vectors[:, :max_length, :].contiguous()

        # 1. Encoder (JÁ FOI EXECUTADO)
        # 2. Buffer de saída inicializado com zeros
        out_vectors = torch.zeros(B, max_length, F_out, dtype=enc_out.dtype, device=device)
        out_vectors[:, :L0, :] = start_vectors

        # 3. Geração auto-regressiva
        t = L0
        # loop até atingir max_length
        while t < max_length:
            # prepara entrada do decoder: pega todos os vetores gerados até agora
            tgt_in = out_vectors[:, :t, :]  # (batch, t, target_features)
            # máscara causal: usada para evitar atenção futura
            causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))
            # expande a máscara para o tamanho do batch
            causal_b = causal.unsqueeze(0).expand(B, -1, -1)  # (batch,t,t)
            # Executa o decoder C-VAE (passando z)
            dec = self.transformer.decoder(tgt_in, z, enc_out, causal_b, None) 
            # projeta saída do decoder para o espaço das features de saída
            predictions = self.final_projection(dec) # (batch, t, target_features)
            # pega o último vetor previsto
            next_vector = predictions[:, -1, :]  # (batch, target_features)
            # armazena no buffer de saída
            out_vectors[:, t, :] = next_vector
            # incrementa o passo
            t += 1
        # 4. Retorna os vetores gerados
        return out_vectors


    @torch.jit.export
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Método 'forward' exposto para torch.ts (Geração Aleatória).
        Amostra 'z' aleatoriamente.
        entrada:
          - tensor de features de entrada achatado (enviado pelo PD): (N): N == frames * input_features
        Retorna:
            - tensor achatado: (M) - M == max_length * target_features
        """
        B = 1 # Batch size é 1 para inferência em tempo real
        device = src.device
        dtype = src.dtype

        # 1. faz reshape da entrada para (1, frames, input_features)
        src = src.view(B, self.frames, self.input_features)
        
        # 2. Gera Contexto a partir das features de entrada
        enc_out = self.transformer.conditional_encoder(src, None)

        # 3. Amostra 'z' aleatoriamente (o núcleo VAE) [batch, latent_dim]
        z = torch.randn(B, self.latent_dim, device=device, dtype=dtype)
        
        # 4. Cria vetor de início (zeros)
        start_vector = torch.zeros((B, 1, self.target_features), dtype=dtype, device=device)

        # 5. chama generate para criar a sequência completa
        # Gera 'max_length + 1' e descarta o primeiro 
        full_seq = self.generate(enc_out, z, start_vector, self.max_length + 1) # full_seq: (batch, max_length+1, target_features)
        
        # 6. Retorna apenas os passos gerados e achata o tensor para enviar ao PD
        out = full_seq[:, 1:, :].contiguous() # descarta o primeiro passo (start vector)
        return out.flatten() # retorna predição achatada [M]: M == max_length * target_features


    @torch.jit.export
    def forwardz(self, src: torch.Tensor) -> torch.Tensor:
        """
        Método 'forwardz' exposto para o torch.ts (Geração controlada passando 'z' junto com a entrada).
        entrada:
          - src: features de entrada achatadas (N): N == frames * input_features
          - z_in: tensor z achatado (K): K == latent_dim
        Retorna:
            - features de saída achatadas: (M): M == max_length * target_features
        """
        B = 1 # Batch size é 1 para inferência em tempo real
        device = src.device
        dtype = src.dtype

        # 1. faz reshape da entrada para (1, frames, input_features)
        src = src.view(B, self.frames, self.input_features)

        # 2. lê z armazenado em z_buffer: usa buffer armazenado se definido, caso contrário gera z aleatório
        if int(self.z.item()) == 1:
            z = self.z_buffer.unsqueeze(0)  # (1, latent_dim)
        else:
            z = torch.randn(B, self.latent_dim, device=device, dtype=dtype)
        
        # 3. Gera Contexto a partir das features de entrada
        enc_out = self.transformer.conditional_encoder(src, None)

        # 4. Cria vetor de início (zeros)
        start_vector = torch.zeros((B, 1, self.target_features), dtype=dtype, device=device)

        # 5. chama generate para criar a sequência completa a partir de 'z' fornecido
        full_seq = self.generate(enc_out, z, start_vector, self.max_length + 1)
        
        # 6. exclui o primeiro passo (start vector)
        out = full_seq[:, 1:, :].contiguous()
        # retorna apenas os passos gerados e achata o tensor para enviar ao PD  
        return out.flatten()


if __name__ == "__main__": 
    # 1. Parâmetros
    INPUT_FEATURES = 64  # número de features de entrada
    TARGET_FEATURES = 20   # número de features de saída
    SEQ_LEN = 20          # comprimento da sequência de entrada
    MAX_POS = 100         # posição máxima para o codificador
    LATENT_DIM = 32       # dimensão do espaço latente

    # 2. Arquitetura do modelo
    model = TransformerCVAE(
        num_layers_enc=2,
        num_layers_dec=2,
        d_model=64,
        num_heads=2,
        d_ff=128,
        input_features=INPUT_FEATURES,
        target_features=TARGET_FEATURES,
        latent_dim=LATENT_DIM,
        max_pos=MAX_POS,
        dropout=0.1,
    )
    
    # 3. Carrega os pesos do modelo treinado
    weights_path = CHECKPOINT_DIR + "/model.pt"
    
    try:
        model.load_state_dict(torch.load(weights_path))
        print(f"Pesos carregados de {weights_path}")
    except FileNotFoundError:
        print(f"Aviso: Arquivo de pesos '{weights_path}' não encontrado.")
        print("Continuando com pesos aleatórios para o teste.")
    except Exception as e:
        print(f"Erro ao carregar pesos: {e}")
        
    model.eval()

    scripted_model = torch.jit.script(model)
    
    # 4. Cria e scripta o Wrapper (passando latent_dim)
    wrapper = TcvaeWrapper(
        scripted_model, 
        max_length=MAX_POS, 
        frames=SEQ_LEN, 
        input_features=INPUT_FEATURES, 
        target_features=TARGET_FEATURES,
        latent_dim=LATENT_DIM
    )
    scripted_wrapper = torch.jit.script(wrapper)

    # 5. Salva o modelo TorchScript
    output_path = os.path.join(MODEL_DIR, "tc-vae.ts")
    scripted_wrapper.save(output_path)
    print(f"Torchscript C-VAE salvo em {output_path}")

    # 6. Teste de Geração
    print("\nIniciando teste de geração...")
    # carrega o modelo salvo
    loaded_model = torch.jit.load(output_path)
    loaded_model.eval()

    # entrada fictícia para teste
    dummy_src = torch.randn(1, SEQ_LEN, INPUT_FEATURES)
    flat_src = dummy_src.view(-1)
    # z fictício para teste
    dummy_z = torch.randn(LATENT_DIM)
    print("Métodos registrados:", loaded_model.get_methods())
    print("Atributos registrados:", loaded_model.get_attributes())

    # --- Teste 1: 'forward' (Geração Aleatória) ---
    print("\n------ Teste 'forward' (z aleatório) ------")
    loaded_model.steps(10) # seta max_length para 10
    with torch.no_grad():
        out_flat_random = loaded_model.forward(flat_src)
    
    n_frames_out = int(loaded_model.max_output_length.item()) // TARGET_FEATURES
    out3_random = out_flat_random.view(1, n_frames_out, TARGET_FEATURES)

    print(f"Shape do input (features de entrada): {dummy_src.shape}")
    print(f"Shape da saída (features de saída): {out3_random.shape}")
    assert out3_random.shape == (1, 10, TARGET_FEATURES)
    print("Teste 'forward' OK!")

    # --- Teste 2: 'forwardz' (Geração Controlada) ---
    print("\n------ Teste 'forwardz' (z controlado) ------")
    loaded_model.steps(50) # seta max_length para 50
    loaded_model.latent(dummy_z) # seta z controlado
    with torch.no_grad():
        out_flat_z = loaded_model.forwardz(flat_src)
    
    n_frames_out_z = int(loaded_model.max_output_length.item()) // TARGET_FEATURES
    out3_z = out_flat_z.view(1, n_frames_out_z, TARGET_FEATURES)

    print(f"Shape do input (features de entrada): {dummy_src.shape}")
    print(f"Shape do input (latente): {dummy_z.shape}")
    print(f"Shape da saída (features de saída): {out3_z.shape}")
    assert out3_z.shape == (1, 50, TARGET_FEATURES)
    print("Teste 'forwardz' OK!")

    print("\nTeste de geração C-VAE concluído com sucesso!")