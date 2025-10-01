# IRIS_HOST = "localhost"
# IRIS_PORT = 1972
# IRIS_NAMESPACE = "USER"
# IRIS_USER = "_SYSTEM"
# IRIS_PASSWORD = "demo" 
import os
# IRIS_HOST = os.getenv("IRIS_HOST", "iris")
# IRIS_HOST = os.getenv("IRIS_HOST", "localhost")

# IRIS_PORT = int(os.getenv("IRIS_PORT", "1972"))
# IRIS_NAMESPACE = os.getenv("IRIS_NAMESPACE", "USER")
# IRIS_USER = os.getenv("IRIS_USER", "_SYSTEM")
# IRIS_PASSWORD = os.getenv("IRIS_PASSWORD", "SYS")

IRIS_HOST = os.getenv("IRIS_HOST")
IRIS_PORT = int(os.getenv("IRIS_PORT"))
IRIS_NAMESPACE = os.getenv("IRIS_NAMESPACE")
IRIS_USER = os.getenv("IRIS_USER")
IRIS_PASSWORD = os.getenv("IRIS_PASSWORD")

