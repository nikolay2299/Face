import logging
from app import app

# Configure logging for easier debugging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # Start the Flask application on port 5001 to avoid conflicts with gunicorn
    app.run(host="0.0.0.0", port=5001, debug=True)
