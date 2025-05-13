from prometheus_client import Counter, Histogram, start_http_server
import time

# Define metrics
PREDICTION_REQUEST_COUNT = Counter(
    'prediction_request_count',
    'Number of prediction requests received'
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction requests',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

PREDICTION_ERRORS = Counter(
    'prediction_errors_total',
    'Number of prediction errors'
)

class MonitoringMiddleware:
    async def __call__(self, request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            if request.url.path == "/predict":
                PREDICTION_REQUEST_COUNT.inc()
                PREDICTION_LATENCY.observe(time.time() - start_time)
            
            return response
            
        except Exception as e:
            if request.url.path == "/predict":
                PREDICTION_ERRORS.inc()
            raise e

def start_monitoring(port=8000):
    """Start Prometheus metrics server."""
    start_http_server(port) 