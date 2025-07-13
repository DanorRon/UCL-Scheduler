"""
UCL Scheduler Package
An automated availability-based rehearsal scheduler.
"""

__version__ = "1.0.0"
__author__ = "Ronan Venkat"
__email__ = "ronanvenkat@gmail.com"

# Import main classes from existing files
from .algorithm.constrained_scheduler import RehearsalScheduler, RehearsalRequest
from .algorithm.optimal_scheduler import OptimizedRehearsalScheduler, OptimizationWeights, TimeOfDayPreferences, RoomPreferences, ContinuityPreferences

__all__ = [
    "RehearsalScheduler",
    "RehearsalRequest", 
    "OptimizedRehearsalScheduler",
    "OptimizationWeights",
    "TimeOfDayPreferences",
    "RoomPreferences",
    "ContinuityPreferences",
] 