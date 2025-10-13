"""
Resource management for optimization and real-time pattern detection.

Handles CPU/memory allocation, priority management, and graceful degradation
when both optimization and real-time detection are running.
"""

from __future__ import annotations

import psutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from loguru import logger
from enum import Enum

class Priority(Enum):
    """Task priority levels"""
    REAL_TIME = 1      # Real-time pattern detection (highest)
    INTERACTIVE = 2    # User-initiated optimizations
    BACKGROUND = 3     # Scheduled optimizations (lowest)

class ResourceType(Enum):
    """Resource types to manage"""
    CPU_CORES = "cpu_cores"
    MEMORY_GB = "memory_gb"
    DISK_IO = "disk_io"

@dataclass
class ResourceAllocation:
    """Resource allocation for a task"""
    cpu_cores: int
    memory_gb: float
    max_threads: int
    priority: Priority
    task_id: str
    start_time: datetime
    estimated_duration: Optional[timedelta] = None

@dataclass
class SystemResources:
    """Current system resource status"""
    total_cpu_cores: int
    available_cpu_cores: int
    total_memory_gb: float
    available_memory_gb: float
    cpu_usage_percent: float
    memory_usage_percent: float

class ResourceManager:
    """
    Manages system resources between optimization and real-time detection.

    Priority order:
    1. Real-time pattern detection (always gets priority)
    2. Interactive user optimizations
    3. Background scheduled optimizations
    """

    def __init__(self):
        self.active_allocations: Dict[str, ResourceAllocation] = {}
        self.pending_requests: List[Dict] = []
        self.system_resources = self._get_system_resources()

        # Configuration for Intel i9-13800HX
        self.cpu_config = self._configure_for_i9_13800hx()

        # Monitoring
        self._monitor_thread = None
        self._monitor_running = False
        self._resource_callbacks: List[Callable] = []

    def _configure_for_i9_13800hx(self) -> Dict:
        """Configure resource limits for Intel i9-13800HX"""

        # Intel i9-13800HX specifications
        # 20 cores total (8 P-cores + 12 E-cores)
        # 28 threads (16 P-core threads + 12 E-core threads)
        # Base frequency: 2.5 GHz, Max turbo: 5.2 GHz

        return {
            "total_cores": 20,
            "performance_cores": 8,   # Hyperthreaded (16 threads)
            "efficiency_cores": 12,   # Single threaded
            "total_threads": 28,

            # Resource allocation strategy
            "real_time_allocation": {
                "max_cores": 4,       # Reserve 4 cores for real-time
                "max_threads": 6,     # Use both P and E cores
                "memory_gb": 4.0
            },

            "optimization_allocation": {
                "max_cores": 16,      # Can use most cores when no real-time
                "max_threads": 24,    # Leave some headroom
                "memory_gb": 16.0,    # Assume 32GB+ system
                "background_max_cores": 12,  # When real-time is active
                "background_max_threads": 16
            }
        }

    def _get_system_resources(self) -> SystemResources:
        """Get current system resource status"""

        cpu_count = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=1)

        memory = psutil.virtual_memory()
        memory_total_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        memory_percent = memory.percent

        return SystemResources(
            total_cpu_cores=cpu_count,
            available_cpu_cores=max(1, int(cpu_count * (100 - cpu_percent) / 100)),
            total_memory_gb=memory_total_gb,
            available_memory_gb=memory_available_gb,
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory_percent
        )

    def request_resources(self, task_id: str, priority: Priority,
                         estimated_duration: Optional[timedelta] = None,
                         cpu_cores: Optional[int] = None,
                         memory_gb: Optional[float] = None) -> Optional[ResourceAllocation]:
        """
        Request resource allocation for a task.

        Returns allocation if granted, None if resources unavailable.
        """

        logger.info(f"Resource request: {task_id} (priority: {priority.name})")

        # Determine optimal allocation based on priority and system status
        allocation = self._calculate_allocation(
            task_id, priority, estimated_duration, cpu_cores, memory_gb
        )

        if allocation is None:
            logger.warning(f"Resource request denied for {task_id}")
            return None

        # Check for conflicts and handle priority preemption
        if self._has_resource_conflict(allocation):
            resolved = self._resolve_conflicts(allocation)
            if not resolved:
                logger.warning(f"Could not resolve resource conflicts for {task_id}")
                return None

        # Grant allocation
        self.active_allocations[task_id] = allocation
        logger.info(f"Resources allocated to {task_id}: {allocation.cpu_cores} cores, "
                   f"{allocation.memory_gb:.1f}GB, {allocation.max_threads} threads")

        return allocation

    def _calculate_allocation(self, task_id: str, priority: Priority,
                            estimated_duration: Optional[timedelta],
                            requested_cores: Optional[int],
                            requested_memory: Optional[float]) -> Optional[ResourceAllocation]:
        """Calculate optimal resource allocation for a task"""

        config = self.cpu_config

        if priority == Priority.REAL_TIME:
            # Real-time gets fixed, guaranteed allocation
            return ResourceAllocation(
                cpu_cores=config["real_time_allocation"]["max_cores"],
                memory_gb=config["real_time_allocation"]["memory_gb"],
                max_threads=config["real_time_allocation"]["max_threads"],
                priority=priority,
                task_id=task_id,
                start_time=datetime.now(),
                estimated_duration=estimated_duration
            )

        elif priority == Priority.INTERACTIVE:
            # Interactive gets good allocation if available
            real_time_active = any(
                alloc.priority == Priority.REAL_TIME
                for alloc in self.active_allocations.values()
            )

            if real_time_active:
                # Share with real-time
                max_cores = config["optimization_allocation"]["background_max_cores"]
                max_threads = config["optimization_allocation"]["background_max_threads"]
            else:
                # Full optimization allocation
                max_cores = config["optimization_allocation"]["max_cores"]
                max_threads = config["optimization_allocation"]["max_threads"]

            return ResourceAllocation(
                cpu_cores=min(requested_cores or max_cores, max_cores),
                memory_gb=min(requested_memory or 8.0, config["optimization_allocation"]["memory_gb"]),
                max_threads=max_threads,
                priority=priority,
                task_id=task_id,
                start_time=datetime.now(),
                estimated_duration=estimated_duration
            )

        else:  # Priority.BACKGROUND
            # Background gets minimal allocation
            real_time_active = any(
                alloc.priority == Priority.REAL_TIME
                for alloc in self.active_allocations.values()
            )

            if real_time_active:
                # Very limited when real-time is active
                return ResourceAllocation(
                    cpu_cores=4,
                    memory_gb=4.0,
                    max_threads=8,
                    priority=priority,
                    task_id=task_id,
                    start_time=datetime.now(),
                    estimated_duration=estimated_duration
                )
            else:
                # Better allocation when no real-time
                return ResourceAllocation(
                    cpu_cores=config["optimization_allocation"]["background_max_cores"],
                    memory_gb=config["optimization_allocation"]["memory_gb"],
                    max_threads=config["optimization_allocation"]["background_max_threads"],
                    priority=priority,
                    task_id=task_id,
                    start_time=datetime.now(),
                    estimated_duration=estimated_duration
                )

    def _has_resource_conflict(self, new_allocation: ResourceAllocation) -> bool:
        """Check if new allocation conflicts with existing ones"""

        total_cores = sum(alloc.cpu_cores for alloc in self.active_allocations.values())
        total_memory = sum(alloc.memory_gb for alloc in self.active_allocations.values())

        total_cores += new_allocation.cpu_cores
        total_memory += new_allocation.memory_gb

        return (total_cores > self.cpu_config["total_cores"] or
                total_memory > self.system_resources.total_memory_gb * 0.8)

    def _resolve_conflicts(self, new_allocation: ResourceAllocation) -> bool:
        """Resolve resource conflicts using priority-based preemption"""

        # Sort existing allocations by priority (higher priority = lower number)
        existing = sorted(
            self.active_allocations.values(),
            key=lambda x: x.priority.value,
            reverse=True  # Lower priority tasks first
        )

        for allocation in existing:
            # Can only preempt lower priority tasks
            if allocation.priority.value > new_allocation.priority.value:
                logger.info(f"Pausing lower priority task {allocation.task_id} for {new_allocation.task_id}")

                # Pause the lower priority task
                self.pause_task(allocation.task_id, f"Preempted by higher priority task {new_allocation.task_id}")

                # Check if conflict is resolved
                if not self._has_resource_conflict(new_allocation):
                    return True

        return False

    def pause_task(self, task_id: str, reason: str = "Resource conflict"):
        """Pause a task to free up resources"""

        if task_id in self.active_allocations:
            allocation = self.active_allocations[task_id]
            logger.info(f"Pausing task {task_id}: {reason}")

            # Move to pending with pause information
            self.pending_requests.append({
                "allocation": allocation,
                "pause_reason": reason,
                "paused_at": datetime.now()
            })

            # Remove from active
            del self.active_allocations[task_id]

            # Notify callbacks about resource change
            self._notify_resource_change(task_id, "paused")

    def resume_task(self, task_id: str) -> bool:
        """Resume a paused task if resources are available"""

        # Find in pending
        for i, req in enumerate(self.pending_requests):
            if req["allocation"].task_id == task_id:
                allocation = req["allocation"]

                # Check if resources are now available
                if not self._has_resource_conflict(allocation):
                    # Resume task
                    self.active_allocations[task_id] = allocation
                    del self.pending_requests[i]

                    logger.info(f"Resumed task {task_id}")
                    self._notify_resource_change(task_id, "resumed")
                    return True

        return False

    def release_resources(self, task_id: str):
        """Release resources when task completes"""

        if task_id in self.active_allocations:
            allocation = self.active_allocations[task_id]
            del self.active_allocations[task_id]

            logger.info(f"Released resources for completed task {task_id}")

            # Try to resume paused tasks
            self._try_resume_pending_tasks()

            self._notify_resource_change(task_id, "completed")

    def _try_resume_pending_tasks(self):
        """Try to resume paused tasks when resources become available"""

        # Sort pending by priority
        sorted_pending = sorted(
            self.pending_requests,
            key=lambda x: x["allocation"].priority.value
        )

        for req in sorted_pending:
            if self.resume_task(req["allocation"].task_id):
                break  # Only resume one at a time

    def get_resource_status(self) -> Dict:
        """Get current resource allocation status"""

        self.system_resources = self._get_system_resources()

        active_tasks = []
        for task_id, allocation in self.active_allocations.items():
            active_tasks.append({
                "task_id": task_id,
                "priority": allocation.priority.name,
                "cpu_cores": allocation.cpu_cores,
                "memory_gb": allocation.memory_gb,
                "max_threads": allocation.max_threads,
                "duration": datetime.now() - allocation.start_time
            })

        pending_tasks = []
        for req in self.pending_requests:
            allocation = req["allocation"]
            pending_tasks.append({
                "task_id": allocation.task_id,
                "priority": allocation.priority.name,
                "paused_since": req["paused_at"],
                "pause_reason": req["pause_reason"]
            })

        return {
            "system": {
                "cpu_cores_total": self.system_resources.total_cpu_cores,
                "cpu_cores_available": self.system_resources.available_cpu_cores,
                "memory_total_gb": self.system_resources.total_memory_gb,
                "memory_available_gb": self.system_resources.available_memory_gb,
                "cpu_usage_percent": self.system_resources.cpu_usage_percent,
                "memory_usage_percent": self.system_resources.memory_usage_percent
            },
            "allocation": {
                "active_tasks": active_tasks,
                "pending_tasks": pending_tasks,
                "total_allocated_cores": sum(alloc.cpu_cores for alloc in self.active_allocations.values()),
                "total_allocated_memory": sum(alloc.memory_gb for alloc in self.active_allocations.values())
            }
        }

    def _notify_resource_change(self, task_id: str, event: str):
        """Notify registered callbacks about resource changes"""
        for callback in self._resource_callbacks:
            try:
                callback(task_id, event)
            except Exception as e:
                logger.warning(f"Resource callback failed: {e}")

    def register_callback(self, callback: Callable):
        """Register callback for resource change events"""
        self._resource_callbacks.append(callback)