"""
Restaurant Inventory Optimization Web Application
Flask backend for DDQN-based ordering recommendations
"""

from flask import Flask, render_template, request, jsonify
import json
import os
from model import InventoryOptimizer

app = Flask(__name__)

# Initialize model
MODEL_PATH = os.path.join('models', 'ddqn_model.pth')
METRICS_PATH = os.path.join('models', 'comparison_metrics.json')

optimizer = InventoryOptimizer(MODEL_PATH)

# Load comparison metrics
with open(METRICS_PATH, 'r') as f:
    comparison_data = json.load(f)


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Get ordering recommendation from DDQN model
    
    Expected JSON:
    {
        "current_inventory": 150,
        "day": "Monday"
    }
    
    Returns:
    {
        "success": true,
        "recommendation": {
            "order_quantity": 150,
            "day": "Monday",
            "current_inventory": 150,
            "predicted_inventory_after_order": 300,
            "confidence": "High",
            "reasoning": "..."
        }
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        current_inventory = int(data.get('current_inventory', 0))
        day = data.get('day', 'Monday')
        
        # Validate inventory range
        if current_inventory < 0 or current_inventory > 500:
            return jsonify({
                'success': False,
                'error': 'Inventory must be between 0 and 500 units'
            }), 400
        
        # Validate day
        valid_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        if day not in valid_days:
            return jsonify({
                'success': False,
                'error': f'Invalid day. Must be one of: {", ".join(valid_days)}'
            }), 400
        
        # Get prediction
        recommendation = optimizer.get_recommendation(current_inventory, day)
        
        return jsonify({
            'success': True,
            'recommendation': recommendation
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid input: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/comparison-data')
def get_comparison_data():
    """
    Get comparison data for all models
    
    Returns:
    {
        "methods": [...],
        "metrics": {...},
        "charts": {...}
    }
    """
    try:
        return jsonify({
            'success': True,
            'data': comparison_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': optimizer.model is not None
    })


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='0.0.0.0', port=5000)
