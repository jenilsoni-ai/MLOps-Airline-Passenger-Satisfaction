from prometheus_client import Counter, Histogram, start_http_server
import time
from functools import wraps

# Initialize Prometheus metrics
REQUESTS = Counter(
    'prediction_requests_total',
    'Number of prediction requests received'
)

LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction requests',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

ERRORS = Counter(
    'prediction_errors_total',
    'Number of prediction errors'
)

def start_monitoring(port=8001):
    """Start Prometheus metrics server"""
    start_http_server(port)

class MonitoringMiddleware:
    """Middleware to monitor FastAPI endpoints"""
    
    async def __call__(self, request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Only track metrics for prediction endpoint
            if request.url.path == "/predict":
                REQUESTS.inc()
                LATENCY.observe(time.time() - start_time)
                
                if response.status_code >= 400:
                    ERRORS.inc()
            
            return response
            
        except Exception as e:
            if request.url.path == "/predict":
                ERRORS.inc()
            raise e 