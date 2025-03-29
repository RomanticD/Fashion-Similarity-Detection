# Step 1: Modify src/utils/request_tracker.py to add a timeout mechanism
# Add this to the RequestTracker class
import inspect
import threading
import time
from typing import Dict, Any, Callable
from functools import wraps


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

    # Add this new method for cancellable operations
    def run_cancellable(self, request_id: str, operation: Callable, *args, **kwargs):
        """
        Run an operation that periodically checks for cancellation.

        Args:
            request_id: The ID of the request being executed
            operation: The function to call
            *args, **kwargs: Arguments to pass to the operation

        Returns:
            The result of the operation, or None if cancelled

        Raises:
            CancellationException: If the operation was cancelled
        """
        if self.is_cancelled(request_id):
            print(f"Operation cancelled before starting: {request_id}")
            raise CancellationException(f"Request {request_id} was cancelled")

        # Create a separate cancellation check thread
        cancel_check = threading.Event()

        def check_cancel():
            while not cancel_check.is_set():
                if self.is_cancelled(request_id):
                    cancel_check.set()
                    return
                time.sleep(0.1)  # Check every 100ms

        # Start a daemon thread to periodically check for cancellation
        check_thread = threading.Thread(target=check_cancel, daemon=True)
        check_thread.start()

        try:
            # Run the operation
            result = operation(*args, **kwargs)

            # Check once more after operation completes
            if self.is_cancelled(request_id):
                raise CancellationException(f"Request {request_id} was cancelled during operation")

            return result
        finally:
            # Always stop the cancellation check thread
            cancel_check.set()
            check_thread.join(timeout=0.2)  # Allow thread to exit gracefully

    def cancellable(self, request_id_arg='request_id'):
        """
        Decorator to make a function cancellable.

        Args:
            request_id_arg: The name of the parameter containing the request ID

        Returns:
            Decorated function that will check for cancellation
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract the request_id from args or kwargs
                request_id = kwargs.get(request_id_arg)
                if request_id is None:
                    # Try to find by position
                    sig = inspect.signature(func)
                    params = list(sig.parameters.keys())
                    if request_id_arg in params:
                        pos = params.index(request_id_arg)
                        if pos < len(args):
                            request_id = args[pos]

                if request_id is None:
                    # Can't find request_id, just run the function normally
                    return func(*args, **kwargs)

                # Run with cancellation support
                return self.run_cancellable(request_id, func, *args, **kwargs)

            return wrapper

        return decorator


# Add this new exception class
class CancellationException(Exception):
    """Exception raised when a request is cancelled during processing."""
    pass


# Global instance that can be imported and used anywhere
request_tracker = RequestTracker()