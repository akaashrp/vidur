import argparse
import datetime
import os
from dataclasses import dataclass
from typing import List, Any
import numpy as np
import pandas as pd
import torch
import ray
from tqdm import tqdm

from vidur.profiling.common.cuda_timer import CudaTimer
from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.common.timer_stats_store import TimerStatsStore
from vidur.profiling.bandwidth.bandwidth_wrapper import BandwidthWrapper, BandwidthTestConfig, get_bandwidth_test_configs

WARMUP_STEPS = 2
ACTIVE_STEPS = 5

def parse_args():
    parser = argparse.ArgumentParser(description="CPU-GPU Bandwidth Profiling")
    parser.add_argument(
        "--disable_ray",
        action="store_true",
        help="Disable Ray",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for profiling",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="profiling_outputs",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "microsoft/phi-2",
            "internlm/internlm-20b",
            "Qwen/Qwen-72B",
            "meta-llama/Llama-2-7b-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "meta-llama/Llama-2-70b-hf",
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-70B",
            "facebook/opt-6.7b",
            "facebook/opt-30b",
        ],
        help="Models to profile",
    )
    args = parser.parse_args()

    args.output_dir = (f"{args.output_dir}/mlp/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(args.output_dir, exist_ok=True)

    return args

def profile_model(
    args: argparse.Namespace,
    model: str,
    test_configs: List[BandwidthTestConfig],
    dtype: torch.dtype,
    pbar: Any,
) -> pd.DataFrame:
    """Profile bandwidth for a specific model."""
    model_config = ModelConfig.from_model_name(model)
    
    promises = []
    all_results = []
    
    wrapper_actor = ray.remote(
        num_cpus=1,
        num_gpus=1,
    )(
        BandwidthWrapper
    ).options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})
    
    wrappers = [
        wrapper_actor.remote(model_config, dtype)
        for _ in range(args.num_gpus)
    ]
    
    for config in test_configs:
        worker_id = len(promises) % args.num_gpus
        promise = wrappers[worker_id].profile.remote(config)
        promises.append(promise)

        print("Promises:", promises)
        
        if len(promises) >= args.num_gpus:
            results = ray.get(promises)
            print("Results:", results)
            all_results.extend(results)
            promises = []
            
        pbar.update(1)
    
    if promises:
        results = ray.get(promises)
        all_results.extend(results)
    
    # Convert results to DataFrame
    df = pd.DataFrame(all_results)
    print(df["time_stats"])

    df = (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )
    
    # Calculate bandwidth in GB/s
    df["bandwidth_gbps"] = (
        df["data_size"] * df["batch_size"] / 
        (df["time_stats.mean"] * 1e-9) / 
        (1024 * 1024 * 1024)
    )
    
    return df

def main():
    args = parse_args()
    dtype = torch.float16
    
    if not args.disable_ray:
        ray.init()
    
    all_configs = {}
    for model in args.models:
        model_config = ModelConfig.from_model_name(model)
        configs = get_bandwidth_test_configs(model_config, max_size=1024)
        all_configs[model] = configs
    
    pbar = tqdm(total=sum(len(configs) for configs in all_configs.values()))
    
    for model in args.models:
        result_df = profile_model(
            args,
            model,
            all_configs[model],
            dtype,
            pbar,
        )
        
        # Save results
        os.makedirs(f"{args.output_dir}/{model}", exist_ok=True)
        result_df.to_csv(f"{args.output_dir}/{model}/bandwidth.csv", index=False)
        
        # Create bandwidth profile summary
        bandwidth_profile = result_df.groupby(["direction", "data_size"])["bandwidth_gbps"].agg(
            ["mean", "std", "min", "max"]
        ).reset_index()
        bandwidth_profile.to_csv(
            f"{args.output_dir}/{model}/bandwidth_profile.csv",
            index=False
        )
    
    if not args.disable_ray:
        ray.shutdown()

if __name__ == "__main__":
    main()
