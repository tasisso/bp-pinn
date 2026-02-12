import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class ResBlock(nn.Module):
    def __init__(self, cin, cout, num_conv, kernel_size=5, stride=1):
        super().__init__()
        layers = []
        self.proj = None
        if cin != cout:
          self.proj = nn.Conv1d(cin, cout, kernel_size=1, stride=stride)
        for _ in range(num_conv):
            layers.append(nn.Conv1d(cin, cout, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2))
            layers.append(nn.PReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        residual = x if self.proj is None else self.proj(x)
        return residual + self.conv(x)


##First down block [batch, cin, length] as ex. [1, 2, 249] -> [1, 16, 249]
class DownBlock(nn.Module):
    def __init__(self, cin, cout, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, cin, cout, kernel_size=2, stride=2, output_padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(cin, cout, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, output_padding=output_padding),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

class VNet1d(nn.Module):
    def __init__(self, input_channels=2, channels0=16, output_channels=1, forecasting=False, **kwargs):

        super().__init__()
        self.device = "cpu"
        
        self.encoder = nn.ModuleList([ResBlock(input_channels, channels0, num_conv=1),
                                      DownBlock(channels0, channels0*2),
                                      ResBlock(channels0*2, channels0*2, num_conv=2),
                                      DownBlock(channels0*2, channels0*4),
                                      ResBlock(channels0*4, channels0*4, num_conv=3),
                                      DownBlock(channels0*4, channels0*8),
                                      ResBlock(channels0*8, channels0*8, num_conv=3),
                                      DownBlock(channels0*8, channels0*16),
                                      ResBlock(channels0*16, channels0*16, num_conv=3)])
        
        #Projections for horizontal connections
        self.projections = nn.ModuleList([
            nn.Conv1d(channels0*(2**i), channels0*(2**(i+1)), kernel_size=1)
            for i in range(3, -1, -1)   
        ])
        
        self.decoder = nn.ModuleList([UpBlock(channels0*16, channels0*16),
                                     ResBlock(channels0*16, channels0*16, num_conv=3),
                                     UpBlock(channels0*16, channels0*8, output_padding=1),
                                     ResBlock(channels0*8, channels0*8, num_conv=3),
                                     UpBlock(channels0*8, channels0*4, output_padding=1),
                                     ResBlock(channels0*4, channels0*4, num_conv=2),
                                     UpBlock(channels0*4, channels0*2, output_padding=1),
                                     ResBlock(channels0*2, channels0*2, num_conv=1)])
    
        if forecasting:
            self.upsample = nn.Sequential(
                nn.ConvTranspose1d(in_channels=channels0*2, out_channels=channels0*2, kernel_size=2, stride=2, output_padding=1),
                nn.PReLU()
            )
        else:
            self.upsample = nn.Identity()
            
            # nn.ConvTranspose1d(
            #     in_channels=channels0*2,
            #     out_channels=channels0*2,  # keep same channels before final projection
            #     kernel_size=2,
            #     stride=2,
            #     output_padding=(1 if T % 2 == 1 else 0)  # handle odd T
            # )
        

        self.final_layer = nn.Sequential(
                nn.Conv1d(in_channels=channels0*2, out_channels=output_channels, kernel_size=1),
                nn.PReLU()
            )
        
    
    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward(self, x, compute_gradients=True, training_mode=False):
        x = x.permute(0, 2, 1) #x is [B, L, C] -> [B, C, L]
        res_cons = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            if (i % 2 == 0 and i < 8):
                #residual connections of all 5x5 convolution blocks except last
                res_cons.append(x)
        for i, block in enumerate(self.decoder):
            #print(f"Dec{i}, Shape of x: {x.shape}")
            if i % 2 == 1:
                res_con = res_cons.pop()
                res_proj = self.projections[i//2](res_con)
                res_proj = res_proj[...,:x.shape[-1]]

                #print(f"Shape of proj: {res_proj.shape}")
                x = x + res_proj
            x = block(x)

        #Project to forecast dimension if forecasting
        x = self.upsample(x)

        x = self.final_layer(x)
        #x.shape = [B, 1, T]
        out = {"P": x.squeeze(1)}

        if not training_mode:
            out = {k: v.detach() if torch.is_tensor(v) else v for k, v in out.items()}

        return out
    

    
class Vnet1dLightning(pl.LightningModule):
    def __init__(self, model_config, optimizer_config, loss_fn = None):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn'])
        self.model = VNet1d(**model_config)
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
        loss, loss_dict = self.loss_fn(model = self.model, **preds)

        # log losses here, don't log losses in callback. 
        self.log_dict({f"train/{k}": v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}, prog_bar=False, batch_size=batch_size)

        # make sure callbacks are consistent for logging/visualizing output dict bellow
        preds["P"] = preds["P"].unsqueeze(2) #change shape to [B, T, 1] for compatibility with [B,T,M] callbacks

        return {"loss": loss, "loss_dict": loss_dict, "preds": preds}
    
    def validation_step(self, batch, batch_idx):

        #Get bactch size
        batch_size = batch["input"].shape[0]

        preds = self.forward(batch)
        preds = self.prepare_loss_inputs(batch, preds)
        loss, loss_dict = self.loss_fn(**preds)

        self.log_dict({f"val/{k}": v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch_size)
        preds["P"] = preds["P"].unsqueeze(2)

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
            "output": batch["output"],
            "output_mask": batch["output_mask"],
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
    from models.vnet1d import VNet1d
    # %%
    B, T, C = 2, 249, 2
    inputs = torch.rand(B, T, C)

    model = VNet1d(
        input_dim=2, channels0=16, output_dim=1
    )

    # with torch.no_grad():
    out = model(inputs, compute_gradients=True, training_mode=False)
    for k, v in out.items():
        print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")
        print("Works")

    # %%
    from models.vnet1d import Vnet1dLightning
    from losses.CustomMSE import CustomMSE
    import torch
    # %%
    model_config = {
        "input_dim": 2,
        "channels0": 16,
        "output_dim": 1
    }
    # %%
    optimizer_config = {"type": "Adam", "params": {"lr": 1e-3}}
    loss_fn = CustomMSE()

    model = Vnet1dLightning(model_config, optimizer_config, loss_fn)
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

# %%s
