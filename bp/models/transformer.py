import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math 
from .blocks import SinusoidalPositionalEncoding

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Separate projections per head
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Linear projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention: (batch, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (batch, num_heads, seq_len, head_dim)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Output projection
        out = self.out_proj(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, ff_dim=64, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output=None):
        x_norm = self.norm1(x)
        if enc_output is None:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        else:
            # cross-attention if enc_output is provided
            attn_out, _ = self.attn(x_norm, enc_output, enc_output)
        x = x + self.dropout(attn_out)

        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        #ffn_out = torch.clamp(ffn_out, min=-1e4, max=1e4)
        x = x + self.dropout(ffn_out)
        return x
    
class Tx_scratch(nn.Module):
    def __init__(self, input_channels=2, seq_len=249, embed_dim=64, num_heads=4, ff_dim=64, n_blocks=3, dropout=0.1, **kwargs):
        super().__init__()
        # Input embeddings
        self.input_proj_d = nn.Sequential(
            nn.Conv1d(input_channels, embed_dim // 2, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=3, padding=1)
        ) #ECG and PPG
        self.input_proj_e = nn.Linear(1, embed_dim) #ABP

        # Positional encodings
        self.pos_enc_e = SinusoidalPositionalEncoding(seq_len=seq_len, d_model=embed_dim)
        self.pos_enc_d = SinusoidalPositionalEncoding(seq_len=seq_len, d_model=embed_dim)

        # Transformer blocks
        self.tx_blocks_e = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(n_blocks)])
        self.tx_blocks_d = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(n_blocks)])

        # Combine outputs of last encoder+decoder blocks
        self.combine_linear = nn.Linear(2*embed_dim, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)

        # Final dense layers
        self.dense1 = nn.Linear(embed_dim, embed_dim)
        self.dense2 = nn.Linear(embed_dim, 1)

    def forward(self, x_d, x_e, compute_gradients=True, training_mode=False):
        # Embed and add positional encoding
        x_d = x_d.permute(0, 2, 1)  # -> (B, C, T)
        x_d = self.input_proj_d(x_d)
        x_d = x_d.permute(0, 2, 1)  # back to -> (B, T, embed_dim)
        x_d = self.pos_enc_d(x_d)
        x_e = self.pos_enc_e(self.input_proj_e(x_e))

        # Pass through Transformer blocks
        for block in self.tx_blocks_e:
            x_e = block(x_e)
        for block in self.tx_blocks_d:
            x_d = block(x_d)

        # Concatenate and apply multi-head attention
        x = torch.cat([x_e, x_d], dim=-1)  # shape (B, T, 128)
        # Project back to embed_dim for MHA
        x = self.combine_linear(x)
        x, _ = self.multihead_attn(x, x, x)

        # Final dense layers
        x = self.dense1(x)
        #x = torch.clamp(x, -1e4, 1e4)
        x = self.dense2(x)

        out = {"P": x.squeeze(1)} #[B,T,1]

        if not training_mode:
            out = {k: v.detach() if torch.is_tensor(v) else v for k, v in out.items()}

        return out
    
class Transformer(nn.Module):
    def __init__(self, input_channels=2, seq_len=249, embed_dim=64, num_heads=4, ff_dim=64, n_blocks=3, dropout=0.1, **kwargs):
        super().__init__()
        self.device = "cpu"
        # Input embeddings
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_channels, embed_dim // 2, kernel_size=3, padding=1), 
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=3, padding=1)
        ) #ECG and PPG

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(seq_len=seq_len, d_model=embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True, 
            dropout=dropout, 
            activation='gelu')
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

        # Output projection
        self.final_layer = nn.Linear(embed_dim, 1)

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward(self, x, compute_gradients=True, training_mode=False):
        # Embed and add positional encoding
        x = x.permute(0, 2, 1)  # -> (B, C, T)
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)  # back to -> (B, T, embed_dim)
        x = self.pos_enc(x)
        
        # Pass through Transformer blocks
        x = self.encoder(x)

        # Project
        x = self.final_layer(x)

        out = {"P": x.squeeze(1)} #[B,T]

        if not training_mode:
            out = {k: v.detach() if torch.is_tensor(v) else v for k, v in out.items()}

        return out

class TransformerLightning(pl.LightningModule):
    def __init__(self, model_config, optimizer_config, loss_fn = None):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)
        self.save_hyperparameters(ignore=['loss_fn'])
        self.model = Transformer(**model_config)
        self.loss_fn = loss_fn
        self.optimizer_config = optimizer_config

    def training_mode(self):
        self.train()
        self.unfreeze()

    def inference_mode(self):
        self.eval()
        self.freeze()

    def forward(self, batch):
        inputs = self.get_model_inputs(batch)
        preds = self.model(inputs, compute_gradients=True, training_mode=self.training)
        return preds
    
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        # Only move what you need
        batch['input'] = batch['input'].to(self.device)
        batch['output'] = batch['output'].to(self.device)
        # batch['output_mask'] = batch['output_mask'].to(self.device)
        # batch['x_locs'] = batch['x_locs'].to(self.device)
        # batch['t'] = batch['t'].to(self.device)
        # batch['dt'] = batch['dt'].to(self.device)
        # batch['phi_ref'] = batch['phi_ref'].to(self.device)
        # batch['phi_mask'] = batch['phi_mask'].to(self.device)
        return batch

    def training_step(self, batch, batch_idx):
        torch.autograd.set_detect_anomaly(True)
        #Get bactch size
        batch_size = batch["input"].shape[0]

        preds = self.forward(batch)
        preds = self.prepare_loss_inputs(batch, preds)
        loss, loss_dict = self.loss_fn(model=self.model, **preds)

        # log losses here, don't log losses in callback. 
        self.log_dict({f"train/{k}": v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}, prog_bar=False, batch_size=batch_size)

        return {"loss": loss, "loss_dict": loss_dict, "preds": preds}
    
    def validation_step(self, batch, batch_idx):

        #Get bactch size
        batch_size = batch["input"].shape[0]

        preds = self.forward(batch)
        preds = self.prepare_loss_inputs(batch, preds)
        loss, loss_dict = self.loss_fn(**preds)

        self.log_dict({f"val/{k}": v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)

        return {"loss": loss, "loss_dict": loss_dict, "preds": preds}
    
    def configure_optimizers(self):
        opt_cls = getattr(torch.optim, self.optimizer_config.get("type", "AdamW"))
        optimizer = opt_cls(self.parameters(), **self.optimizer_config["params"])

        sched_cls = getattr(torch.optim.lr_scheduler, self.optimizer_config.get("scheduler", "OneCycleLR"))
        scheduler = sched_cls(optimizer, **self.optimizer_config["scheduler_params"])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",     # or \"epoch\" — OneCycleLR uses \"step\"
                "frequency": 1,
                "name": "lr",
            }
        }
        
    
    def get_model_inputs(self, batch):
        """
        Extract inputs from batch dictionary: input waveform tensor

        Returns:
            input_waveform: shape [B, C, L]
            output_waveform
        """
        input_waveform = batch["input"]
        
        return input_waveform

    def prepare_loss_inputs(self, batch, preds):
        """
        Merge model outputs and batch fields of choice into a dict:

        Args:
            batch (dict): dataloader batch
            preds (dict): model output
        Returns:
            merged: input dict for loss_fn ({"P": ..., "output": ...})
        """
        merged = dict(preds) #shallow copy of model output -simply {P: [B, C, L]}

        merged.update({
            "output": batch["output"].unsqueeze(-1),
            # "output_mask": batch["output_mask"],
            # "dt": batch["dt"],
            "t_seconds": batch.get("t"),
            "t_seconds_input": batch.get("t_input"),
            "input": batch.get("input"),  # optional: for input-driven constraints
            # "r_peaks": batch.get("r_peaks"),
            # "phi_ref": batch.get("phi_ref"),
            # "phi_mask": batch.get("phi_mask"),
        })

        return merged
    
if __name__ == "__main__":
    # %%
    import sys, os
    import torch
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models.transformer import Transformer
    # %%
    B, T = 2, 249
    enc_channels = 2

    # Dummy inputs
    x_e = torch.randn(B, T, 1)       # ABP (encoder input)
    x_d = torch.randn(B, T, enc_channels)  # PPG+ECG (decoder input)


    model = Transformer(
        input_channels=2,
        seq_len=T,
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        n_blocks=3,
        dropout=0.1
    )

    # with torch.no_grad():
    out = model(x_d, compute_gradients=True, training_mode=False)
    for k, v in out.items():
        print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")
        print("Works")

    from losses.CustomMSE import CustomMSE
    import torch



    # Instantiate model
    model_config = {
        "input_channels": 2,
        "seq_len":T,
        "embed_dim":128,
        "num_heads":4,
        "ff_dim":256,
        "n_blocks":3,
        "dropout":0.1
    }
    # %%
    optimizer_config = {"type": "Adam", "params": {"lr": 1e-3}}
    loss_fn = CustomMSE()

    model = TransformerLightning(model_config, optimizer_config, loss_fn)
    # %%
    batch = {
        "input": torch.randn(2, 249, 2),
        "coords": torch.rand(2, 249, 64, 2),
        "output": torch.randn(2, 249),
        "output_mask": torch.ones(2, 249),
        "dt": torch.tensor([0.01, 0.01]),
        "t": torch.linspace(0, 1, 249).unsqueeze(0).repeat(2, 1),
        "t_input": torch.linspace(0, 1, 249).unsqueeze(0).repeat(2, 1)
    }
    # %%
    preds = model.forward(batch)
    # %%
    loss = model.training_step(batch, 0)
    print("Loss:", loss["loss"], "Loss_dict:", loss["loss_dict"])

# def count_parameters(model):
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")

# if __name__ == "__main__":
#     import torch
#     B, T = 2, 249
#     enc_channels = 2

#     # Dummy inputs
#     x_e = torch.randn(B, T, 1)       # ABP (encoder input)
#     x_d = torch.randn(B, T, enc_channels)  # PPG+ECG (decoder input)

#     # Instantiate model
#     model = Transformer(
#         input_channels=enc_channels,
#         seq_len=T,
#         embed_dim=128,
#         num_heads=4,
#         ff_dim=256,
#         n_blocks=3,
#         dropout=0.1
#     )

#     # Forward pass (optional)
#     y = model(x_e, x_d)
#     print("Output shape:", y.shape)

#     # Count parameters
#     count_parameters(model)
