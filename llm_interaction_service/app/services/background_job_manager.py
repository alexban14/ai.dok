"""
Background Job Manager for long-running tasks.
Best practice: Run heavy tasks in separate process to avoid blocking HTTP workers.
"""
import asyncio
import json
import logging
import multiprocessing
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class JobStatus:
    """Job status information."""
    job_id: str
    status: str  # pending, running, completed, failed
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: int = 0
    total: int = 0
    current_file: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict] = None


class BackgroundJobManager:
    """
    Manages long-running background jobs using separate processes.
    
    Best practices:
    - Jobs run in isolated processes (no worker blocking)
    - Status persisted to disk (survives restarts)
    - Progress tracking via shared memory/files
    - Automatic cleanup of old jobs
    """
    
    def __init__(self, jobs_dir: str = "/tmp/indexing_jobs"):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.running_processes: Dict[str, multiprocessing.Process] = {}
    
    def start_job(self, job_id: str, target_func, *args, **kwargs) -> JobStatus:
        """
        Start a background job in a separate process.
        
        Args:
            job_id: Unique job identifier
            target_func: Function to run in background
            *args, **kwargs: Arguments for target_func
            
        Returns:
            JobStatus with initial state
        """
        # Create initial job status
        status = JobStatus(
            job_id=job_id,
            status="pending",
            started_at=datetime.utcnow().isoformat()
        )
        self._save_status(status)
        
        # Start process
        process = multiprocessing.Process(
            target=self._run_job,
            args=(job_id, target_func, args, kwargs),
            daemon=False  # Allow job to complete even if parent dies
        )
        process.start()
        self.running_processes[job_id] = process
        
        logger.info(f"Started background job {job_id} in process {process.pid}")
        return status
    
    def _run_job(self, job_id: str, target_func, args, kwargs):
        """
        Execute job in isolated process.
        This runs in a separate process, not the HTTP worker!
        """
        try:
            # Update status to running
            status = self.get_status(job_id)
            status.status = "running"
            self._save_status(status)
            
            # Run the actual task
            result = target_func(*args, **kwargs)
            
            # Mark as completed
            status = self.get_status(job_id)
            status.status = "completed"
            status.completed_at = datetime.utcnow().isoformat()
            status.result = result
            self._save_status(status)
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            status = self.get_status(job_id)
            status.status = "failed"
            status.error = str(e)
            status.completed_at = datetime.utcnow().isoformat()
            self._save_status(status)
    
    def get_status(self, job_id: str) -> JobStatus:
        """Get current job status from disk."""
        status_file = self.jobs_dir / f"{job_id}.json"
        
        if not status_file.exists():
            return JobStatus(job_id=job_id, status="not_found")
        
        try:
            with open(status_file, 'r') as f:
                data = json.load(f)
                return JobStatus(**data)
        except Exception as e:
            logger.error(f"Failed to read status for {job_id}: {e}")
            return JobStatus(job_id=job_id, status="error", error=str(e))
    
    def update_progress(self, job_id: str, progress: int, total: int, current_file: str = None):
        """Update job progress (called from within job)."""
        status = self.get_status(job_id)
        status.progress = progress
        status.total = total
        status.current_file = current_file
        self._save_status(status)
    
    def _save_status(self, status: JobStatus):
        """Persist job status to disk."""
        status_file = self.jobs_dir / f"{status.job_id}.json"
        try:
            with open(status_file, 'w') as f:
                json.dump(asdict(status), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save status for {status.job_id}: {e}")
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        if job_id in self.running_processes:
            process = self.running_processes[job_id]
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
            del self.running_processes[job_id]
            
            status = self.get_status(job_id)
            status.status = "cancelled"
            status.completed_at = datetime.utcnow().isoformat()
            self._save_status(status)
            return True
        return False
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Remove job status files older than max_age_hours."""
        cutoff = time.time() - (max_age_hours * 3600)
        for status_file in self.jobs_dir.glob("*.json"):
            if status_file.stat().st_mtime < cutoff:
                status_file.unlink()
                logger.info(f"Cleaned up old job file: {status_file.name}")


# Global instance
job_manager = BackgroundJobManager()
