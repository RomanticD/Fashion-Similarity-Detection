import threading
import time
from typing import Dict, Any


class RequestTracker:
    """
    Tracks active requests and their cancellation status.
    Thread-safe singleton to manage ongoing requests across threads.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RequestTracker, cls).__new__(cls)
                cls._instance.active_requests = {}  # request_id -> status dict
                cls._instance.request_lock = threading.Lock()
        return cls._instance

    def register_request(self, request_id: str) -> None:
        """Register a new request by ID"""
        with self.request_lock:
            self.active_requests[request_id] = {
                'cancelled': False,
                'start_time': time.time()
            }
            print(f"Registered request: {request_id}")

    def cancel_request(self, request_id: str) -> bool:
        """Mark a request as cancelled. Returns True if request existed and was cancelled."""
        with self.request_lock:
            if request_id in self.active_requests:
                self.active_requests[request_id]['cancelled'] = True
                print(f"Cancelled request: {request_id}")
                return True
            return False

    def is_cancelled(self, request_id: str) -> bool:
        """Check if a request is marked as cancelled"""
        with self.request_lock:
            if request_id in self.active_requests:
                return self.active_requests[request_id]['cancelled']
            return False

    def complete_request(self, request_id: str) -> None:
        """Remove a completed request from tracking"""
        with self.request_lock:
            if request_id in self.active_requests:
                del self.active_requests[request_id]
                print(f"Completed request: {request_id}")

    def cleanup_old_requests(self, max_age_seconds: int = 3600) -> None:
        """Remove requests older than max_age_seconds"""
        current_time = time.time()
        with self.request_lock:
            to_remove = []
            for request_id, info in self.active_requests.items():
                if current_time - info['start_time'] > max_age_seconds:
                    to_remove.append(request_id)

            for request_id in to_remove:
                del self.active_requests[request_id]
                print(f"Cleaned up stale request: {request_id}")


# Global instance that can be imported and used anywhere
request_tracker = RequestTracker()