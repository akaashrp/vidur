from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class BandwidthProfile:
    """Represents variable interconnect bandwidth over time."""
    timestamps: List[float]  # Time points where bandwidth changes
    bandwidths: List[float]  # Bandwidth values in Gbps at each timestamp
    
    def get_bandwidth_at_time(self, time: float) -> float:
        """Get the bandwidth at a specific time using linear interpolation."""
        if time <= self.timestamps[0]:
            return self.bandwidths[0]
        if time >= self.timestamps[-1]:
            return self.bandwidths[-1]
            
        for i in range(len(self.timestamps) - 1):
            if self.timestamps[i] <= time <= self.timestamps[i + 1]:
                # Linear interpolation
                t1, t2 = self.timestamps[i], self.timestamps[i + 1]
                b1, b2 = self.bandwidths[i], self.bandwidths[i + 1]
                ratio = (time - t1) / (t2 - t1)
                return b1 + ratio * (b2 - b1)
        return self.bandwidths[-1]

@dataclass
class MemoryOverhead:
    """Represents memory allocation and deallocation overhead."""
    allocation_time: float  # Time in seconds to allocate memory
    deallocation_time: float  # Time in seconds to deallocate memory
    fragmentation_factor: float = 1.0  # Factor to account for memory fragmentation

class PipelineStage:
    def __init__(self, name: str, compute_time: float):
        self.name = name
        self.compute_time = compute_time
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.is_stalled = False
        self.dependent_stages: List['PipelineStage'] = []

def calculate_variable_transfer_time(
    data_size_bytes: float,
    bandwidth_profile: BandwidthProfile,
    current_time: float
) -> Tuple[float, float]:
    """
    Calculate transfer time with variable bandwidth.
    
    Args:
        data_size_bytes: Size of data to transfer
        bandwidth_profile: Profile of bandwidth over time
        current_time: Current simulation time
    
    Returns:
        Tuple of (transfer_time, end_time)
    """
    remaining_bytes = data_size_bytes
    current_pos = current_time
    
    while remaining_bytes > 0:
        current_bandwidth = bandwidth_profile.get_bandwidth_at_time(current_pos)
        bytes_per_second = current_bandwidth * 1e9 / 8
        
        # Calculate how much can be transferred before next bandwidth change
        next_change_idx = next(
            (i for i, t in enumerate(bandwidth_profile.timestamps) if t > current_pos),
            len(bandwidth_profile.timestamps)
        )
        
        if next_change_idx < len(bandwidth_profile.timestamps):
            time_until_change = bandwidth_profile.timestamps[next_change_idx] - current_pos
            bytes_transferrable = bytes_per_second * time_until_change
            
            if bytes_transferrable >= remaining_bytes:
                transfer_time = remaining_bytes / bytes_per_second
                current_pos += transfer_time
                remaining_bytes = 0
            else:
                remaining_bytes -= bytes_transferrable
                current_pos = bandwidth_profile.timestamps[next_change_idx]
        else:
            # No more bandwidth changes, transfer remaining bytes
            transfer_time = remaining_bytes / bytes_per_second
            current_pos += transfer_time
            remaining_bytes = 0
            
    return current_pos - current_time, current_pos

class OffloadSimulator:
    def __init__(
        self,
        bandwidth_profile: BandwidthProfile,
        memory_overhead: MemoryOverhead
    ):
        self.bandwidth_profile = bandwidth_profile
        self.memory_overhead = memory_overhead
        self.current_time = 0.0
        self.pipeline_stages: List[PipelineStage] = []
        
    def simulate_batch_with_pipeline(
        self,
        gpu_stages: List[PipelineStage],
        cpu_stages: List[PipelineStage],
        model_size_bytes: float,
        activation_sizes: List[float],
        batch_size: int
    ) -> float:
        """
        Simulate processing time for a batch with pipelining and stalls.
        
        Args:
            gpu_stages: List of GPU pipeline stages
            cpu_stages: List of CPU pipeline stages
            model_size_bytes: Size of model parameters to transfer
            activation_sizes: Size of activations for each pipeline stage
            batch_size: Number of samples in batch
            
        Returns:
            Total processing time including all overhead
        """
        # Add memory allocation overhead
        total_memory = (model_size_bytes + sum(activation_sizes)) * \
                      self.memory_overhead.fragmentation_factor
        self.current_time += self.memory_overhead.allocation_time * \
                            (total_memory / 1e9)  # Scale with size
        
        # Initial model transfer to CPU
        transfer_time, end_time = calculate_variable_transfer_time(
            model_size_bytes,
            self.bandwidth_profile,
            self.current_time
        )
        self.current_time = end_time
        
        # Process each sample in the batch
        for sample in range(batch_size):
            # Track dependencies between stages
            for i in range(len(gpu_stages) - 1):
                gpu_stages[i].dependent_stages.append(gpu_stages[i + 1])
            for i in range(len(cpu_stages) - 1):
                cpu_stages[i].dependent_stages.append(cpu_stages[i + 1])
                
            # Simulate pipeline execution
            active_stages = gpu_stages + cpu_stages
            while active_stages:
                # Find stages that can start
                for stage in active_stages:
                    if not stage.start_time and not stage.is_stalled:
                        dependencies_met = all(
                            dep.end_time is not None 
                            for dep in stage.dependent_stages
                        )
                        if dependencies_met:
                            stage.start_time = self.current_time
                            stage.end_time = self.current_time + stage.compute_time
                            
                # Check for pipeline stalls
                for stage in active_stages:
                    if stage.start_time and not stage.end_time:
                        # Check if stage is waiting for resources
                        competing_stages = [
                            s for s in active_stages 
                            if s.start_time and not s.end_time and s != stage
                        ]
                        if competing_stages:
                            stage.is_stalled = True
                            
                # Transfer activations between GPU and CPU stages
                for gpu_stage, cpu_stage in zip(gpu_stages, cpu_stages):
                    if gpu_stage.end_time and not cpu_stage.start_time:
                        transfer_time, end_time = calculate_variable_transfer_time(
                            activation_sizes[gpu_stages.index(gpu_stage)],
                            self.bandwidth_profile,
                            self.current_time
                        )
                        self.current_time = end_time
                        
                # Advance simulation time
                completed_stages = [
                    stage for stage in active_stages 
                    if stage.end_time and stage.end_time <= self.current_time
                ]
                active_stages = [
                    stage for stage in active_stages 
                    if stage not in completed_stages
                ]
                
                if active_stages:
                    next_event = min(
                        stage.end_time for stage in active_stages 
                        if stage.end_time is not None
                    )
                    self.current_time = next_event
                    
        # Add memory deallocation overhead
        self.current_time += self.memory_overhead.deallocation_time * \
                            (total_memory / 1e9)  # Scale with size
                            
        return self.current_time

def calculate_effective_throughput(
    simulator: OffloadSimulator,
    batch_size: int,
    processing_time: float
) -> float:
    """
    Calculate effective throughput accounting for all overhead.
    
    Args:
        simulator: Simulator instance with current state
        batch_size: Number of samples processed
        processing_time: Total processing time
        
    Returns:
        Effective throughput in samples per second
    """
    return batch_size / processing_time

# Example usage:
def create_example_simulation():
    # Create bandwidth profile with variation over time
    bandwidth_profile = BandwidthProfile(
        timestamps=[0.0, 1.0, 2.0, 3.0],
        bandwidths=[16.0, 14.5, 15.0, 16.0]  # Gbps
    )
    
    # Define memory overhead parameters
    memory_overhead = MemoryOverhead(
        allocation_time=0.001,  # 1ms base allocation time
        deallocation_time=0.0005,  # 0.5ms base deallocation time
        fragmentation_factor=1.2  # 20% memory fragmentation overhead
    )
    
    # Create simulator
    simulator = OffloadSimulator(bandwidth_profile, memory_overhead)
    
    # Define pipeline stages
    gpu_stages = [
        PipelineStage("gpu_preprocess", 0.001),
        PipelineStage("gpu_inference", 0.005),
        PipelineStage("gpu_postprocess", 0.001)
    ]
    
    cpu_stages = [
        PipelineStage("cpu_preprocess", 0.002),
        PipelineStage("cpu_inference", 0.008),
        PipelineStage("cpu_postprocess", 0.002)
    ]
    
    return simulator, gpu_stages, cpu_stages
