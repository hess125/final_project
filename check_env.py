import os
import sys
print("Python:", sys.executable)
print("Version:", sys.version)
print("Path:", sys.path[:3])

# Try importing TensorFlow
try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    print("TFLite available:", hasattr(tf, 'lite'))
except Exception as e:
    print("TensorFlow import failed:", e)
