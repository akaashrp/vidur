import argparse
import datetime
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import ray
from tqdm import tqdm

import sarathi.metrics.cuda_timer

from vidur.profiling.common.cuda_timer import CudaTimer

# monkey patching the CudaTimer class to use the sarathi implementation
sarathi.metrics.cuda_timer.CudaTimer = CudaTimer

from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.common.timer_stats_store import TimerStatsStore

WARMUP_STEPS = 2
ACTIVE_STEPS = 5

@dataclass
class BandwidthTestConfig:
    """Configuration for bandwidth testing."""
    data_size: int  # Size in bytes
    direction: str  # 'h2d' (host to device) or 'd2h' (device to host)
    batch_size: int  # Number of concurrent transfers
    
    def __str__(self) -> str:
        return f"size={self.data_size}_dir={self.direction}_batch={self.batch_size}"

class BandwidthWrapper:
    def __init__(
        self,
        model_config: ModelConfig,
        dtype: torch.dtype,
    ):
        # self.time_stats_store = TimerStatsStore(profile_method="kineto")
        self.timer = CudaTimer("vidur_bandwidth")
        self._model_config = model_config
        self._dtype = dtype
        try:
            self._device = torch.device("cuda")
        except RuntimeError:
            raise RuntimeError("CUDA is not available on this machine.")
        
    def _get_test_tensors(self, config: BandwidthTestConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create test tensors for bandwidth measurement."""
        if config.direction == 'h2d':
            src = torch.randn(
                config.batch_size, config.data_size // self._dtype.itemsize,
                dtype=self._dtype,
                device='cpu'
            )
            dst = torch.empty_like(src, device=self._device)
        else:
            src = torch.randn(
                config.batch_size, config.data_size // self._dtype.itemsize,
                dtype=self._dtype,
                device=self._device
            )
            dst = torch.empty_like(src, device='cpu')
        return src, dst

    @torch.inference_mode()
    def profile(
        self,
        config: BandwidthTestConfig,
    ) -> Dict:
        """Profile bandwidth for given configuration."""
        src, dst = self._get_test_tensors(config)

        self.timer.__enter__()
        
        # Warmup
        for _ in range(WARMUP_STEPS):
            dst.copy_(src)
        torch.cuda.synchronize()
        
        self.timer.time_stats_store.clear_stats()
        
        # time = datetime.datetime.now()
        # Active measurements
        for _ in range(ACTIVE_STEPS):
            dst.copy_(src)
        torch.cuda.synchronize()

        # time = datetime.datetime.now() - time
        # print("time taken", time)
        
        prof = {
            "time_stats": self.timer.time_stats_store.get_stats(),
            "data_size": config.data_size,
            "direction": config.direction,
            "batch_size": config.batch_size,
            "model_name": self._model_config.name,
            "n_embd": self._model_config.embedding_dim,
            "dtype": str(self._dtype),
        }
        return prof

def get_bandwidth_test_configs(
    model_config: ModelConfig,
    min_size: int = 1024,  # 1KB
    max_size: int = 1024 * 1024 * 1024,  # 1GB
    batch_sizes: List[int] = [1, 2, 4, 8, 16],
) -> List[BandwidthTestConfig]:
    """Generate test configurations based on model parameters."""
    configs = []
    
    # Calculate relevant sizes for the model
    sizes = []
    # Add power-of-2 sizes
    size = min_size
    while size <= max_size:
        sizes.append(size)
        size *= 2
        
    # Add model-specific sizes
    head_size = model_config.get_head_size() * model_config.dtype.itemsize
    kv_size_per_layer = (
        model_config.num_kv_heads * model_config.get_head_size() * 2 * model_config.dtype.itemsize
    )
    sizes.extend([
        head_size,
        kv_size_per_layer,
        kv_size_per_layer * model_config.num_layers,
    ])
    
    # Remove duplicates and sort
    sizes = sorted(list(set(sizes)))
    
    # Create configs for both directions
    for size in sizes:
        for batch_size in batch_sizes:
            for direction in ['h2d', 'd2h']:
                configs.append(BandwidthTestConfig(size, direction, batch_size))
    
    return configs
