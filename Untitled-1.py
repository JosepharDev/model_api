logging.error("Error occurred in /prediction: %s", str(e))
        return jsonify({'error': str(e)}), 500


import logging

logging.basicConfig(
    filename='app_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
