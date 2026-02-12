import torch
import pytorch_lightning as pl
from models.deeponet import DeepONetLightning
from data.lightning import WaveformDataModule  # if using the native one
from torch.utils.data import DataLoader, TensorDataset
from losses.NavierStokesPhysics import NavierStokes1DPhysicsLoss
import numpy as np
import csv
import pandas as pd
from pathlib import Path 
from omegaconf import OmegaConf
from tqdm import tqdm
import os


MODEL_DICT = {
    "deeponet": DeepONetLightning,
}

def build_dataloader_from_numpy(inputs_list, batch_size=32, device="cuda"):
    # Example: inputs_list = [branch_input, trunk_input]
    # Each should be a NumPy array of shape [N, ...]
    tensors = [torch.tensor(arr, dtype=torch.float32, device=device) for arr in inputs_list]
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size)

def make_pred_df(batch, batch_idx, pred_dict_cpu, t_offset, pred_cols, include_groundtruth):
    """
    Extracts predictions from a batch sample in dataframe
    - Optionally includes ground truth waveforms in prediction dataframe
    """
    T = batch["t"].size(1)
    dt = batch["dt"][batch_idx]
    t = torch.arange(T) * dt + t_offset
    start_timestamp = batch["start"][batch_idx]

    sample_out = {"t": start_timestamp + pd.to_timedelta(t.numpy(), unit='s')}

    if include_groundtruth:
        for key in ["input", "output", "phi_ref"]:
            arr = batch[key][batch_idx]
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            # if arr.ndim == 1 and arr.shape[-1] == T:
            if key == "output":
                sample_out["ABP"] = arr
            elif key == "phi_ref":
                sample_out["phi_ref"] = arr.squeeze(1)
            elif key == "input":
                # elif arr.ndim == 2 and arr.shape[-1] == 2:
                sample_out["II"] = arr[:, 0]
                sample_out["PLETH"] = arr[:, 1]

    for key in pred_cols:
        if key not in pred_dict_cpu:
            raise KeyError(f"Missing key {key} in sample output.")
        arr = pred_dict_cpu[key][batch_idx]
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        if arr.ndim == 2 and arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
        sample_out[key] = arr
    
    df_out = pd.DataFrame(sample_out)

    return df_out

def streaming_inference(model, dataloader, save_path, pred_cols, include_groundtruth = True):

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    previous_key = None
    t_offset = 0.0
    pred_buffer = []

    def write_patient_data(pred_buffer, patient_key):
      sid, hid = patient_key
      predfile = save_path / f"{sid}_{hid}_out.csv"
      print(f"Writing {predfile}...")
      pd.concat(pred_buffer).to_csv(predfile, index=False)
      print("Done. Resetting buffer.")

    for batch in tqdm(dataloader, desc="Running inference and writing to file", total=None):
        batch_on_device = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        pred_dict = model(batch_on_device)  # returns dict of predictions
        pred_dict_cpu = {k: v.detach().cpu() for k, v in pred_dict.items()}
        batch_size = next(iter(pred_dict_cpu.values())).shape[0]  # assumes all preds same B
        
        #Flatten per sample
        for i in range(batch_size):

            patient_key = (batch["subject_id"][i], batch["hadm_id"][i])

            #Encounter new patient key -> write out previous key
            if previous_key is not None and patient_key != previous_key:
                write_patient_data(pred_buffer, previous_key)
                #Reset buffer and t-axis
                pred_buffer = []
                t_offset = 0.0
            
            #Extract raw predictions and optionally ground truth signals
            df_out = make_pred_df(batch, i, pred_dict_cpu, t_offset, pred_cols, include_groundtruth)

            #Stitch dfs together
            pred_buffer.append(df_out)

            # Update time offset and key for next sample
            t_offset += batch["t"].size(1) * batch["dt"][i]
            previous_key = patient_key

    #Write out final patient
    if pred_buffer:
        write_patient_data(pred_buffer, previous_key)

if __name__ == "__main__":
    """
    usage: 
    python predict.py \
        --config=../experiments/full-v3/from_scratch.yaml \
        --ckpt=../experiments/full-v3/checkpoints/best-v1.ckpt \
        --batch_size=1 \
        --gpu=3
    """
    import argparse
    parser = argparse.ArgumentParser(description="Run DeepONet Inference")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--use_numpy", action="store_true", default=False, help="Use raw numpy inputs instead of datamodule")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=None, help="GPU index to use (overrides config)")
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    save_dir = cfg.train_config.get("checkpoint_dir", ".")
    print(f"Save dir: {save_dir}")
    ckpt_path = args.ckpt or cfg.train_config.get("saved_model", None)
    # Override GPU if specified in CLI
    gpu_index = args.gpu if args.gpu is not None else cfg.train_config.get("gpu", 0)
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")    # Load model
    # Override batch size if specified in CLI
    if args.batch_size > 1: # otherwise use default from config
        cfg.dataloader_config["test_batch_size"] = args.batch_size
    
    model = cfg.model_config["arch"]

    print("Initializing model")
    loss_fn = NavierStokes1DPhysicsLoss(**cfg.loss_config)
    model = DeepONetLightning(
            model_config=cfg.model_config,
            loss_fn=loss_fn,
            optimizer_config=cfg.optimizer_config,
            sample_size=cfg.train_config.get("sample_size", None)
        )
    ckpt_path = "../experiments/overfit/impute_model.pth"
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.inference_mode() # set sample size to 1, eval() and freeze()
    # model = MODEL_DICT[model].load_from_checkpoint(ckpt_path)
    # model.to(device)
    # model.inference_mode() # set sample size to 1, eval() and freeze()

    # Build dataloader
    if args.use_numpy:
        # NOT IMPLEMENTED
        # Example: hardcoded paths or numpy arrays (replace with actual data loading)
        branch_input = np.load("branch_inputs.npy")
        trunk_input = np.load("trunk_inputs.npy")
        test_loader = build_dataloader_from_numpy([branch_input, trunk_input], batch_size=args.batch_size)
    else:
        datamodule = WaveformDataModule(**cfg.dataloader_config)
        datamodule.setup()
        test_loader = datamodule.test_dataloader()
    
    print("Running inference")
    # Run inference
    streaming_inference(model, test_loader, f"{save_dir}/inference", 
                        pred_cols=["P"], #["P", "Q", "dPdt", "dQdt", "R1", "R2", "C", "A0", "Kr", "beta", "Pext", "rho", "phi"]
                        include_groundtruth=True)
    print("Done")
    