from dataclasses import dataclass, field

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import DeviceSKUType

logger = init_logger(__name__)


@dataclass
class BaseDeviceSKUConfig(BaseFixedConfig):
    fp16_tflops: int
    total_memory_gb: int


@dataclass
class A40DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 65
    total_memory_gb: int = 16

    @staticmethod
    def get_type():
        return DeviceSKUType.A40
    
@dataclass
class T4DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 65
    total_memory_gb: int = 16

    @staticmethod
    def get_type():
        return DeviceSKUType.T4

@dataclass
class A10DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 125
    total_memory_gb: int = 24

    @staticmethod
    def get_type():
        return DeviceSKUType.A10


@dataclass
class A100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 312
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.A100


@dataclass
class H100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 1000
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.H100
