# Deployment Guide for Render

## Repository Structure

```
inventory-optimization-deep-rl/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── inventory_optimization_deep_rl.ipynb
│   └── hyperparameter_tuning.ipynb
├── app.py                    # Flask application
├── model.py                  # Model handler
├── templates/
│   └── index.html           # Main web interface
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── models/
│   ├── ddqn_model.pth       # Trained model (you'll create this)
│   └── comparison_metrics.json
├── Procfile                  # For Render
├── render.yaml              # Render config
└── .gitignore
```

## Setup Instructions

### 1. Prepare Your Model

After training in Jupyter notebook, save your DDQN model:
```python
# In your notebook after training
ddqn_agent.save('models/ddqn_model.pth')
```

### 2. Create comparison_metrics.json

Run this in your notebook:
```python
import json

comparison_metrics = {
    'methods': ['Heuristic', 'DQN', 'Double DQN', 'Dueling DQN'],
    'profits': [
        heuristic_results['mean_profit'],
        dqn_eval_results['mean_profit'],
        ddqn_eval_results['mean_profit'],
        dueling_eval_results['mean_profit']
    ],
    'service_levels': [
        heuristic_results['mean_service_level'] * 100,
        dqn_eval_results['mean_service_level'] * 100,
        ddqn_eval_results['mean_service_level'] * 100,
        dueling_eval_results['mean_service_level'] * 100
    ],
    'training_progress': {
        'dqn': dqn_training_results['episode_profits'][::100],
        'ddqn': ddqn_training_results['episode_profits'][::100],
        'dueling': dueling_training_results['episode_profits'][::100]
    }
}

with open('models/comparison_metrics.json', 'w') as f:
    json.dump(comparison_metrics, f)
```

### 3. Update requirements.txt

Add to your existing requirements.txt:
```
flask==3.0.0
gunicorn==21.2.0
```

### 4. Create Procfile

```
web: gunicorn app:app
```

### 5. Create render.yaml

```yaml
services:
  - type: web
    name: inventory-optimizer
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
```

### 6. Create .gitignore

```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/
.env
venv/
env/
.vscode/
.idea/
*.log
.DS_Store
results/
tuning_results/
*.pth.tar
```

## Deploy to Render

### Method 1: Web Dashboard (Recommended)

1. Push your code to GitHub
2. Go to [render.com](https://render.com)
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Render will auto-detect Python and use your `render.yaml`
6. Click "Create Web Service"
7. Wait for deployment (5-10 minutes)

### Method 2: Render CLI

```bash
# Install Render CLI
npm install -g @render/cli

# Deploy
render deploy
```

## Local Testing

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py

# Open browser
http://localhost:5000
```

## Troubleshooting

### Model File Too Large

If ddqn_model.pth is >100MB:
1. Use Git LFS: `git lfs track "*.pth"`
2. Or host model on S3/Google Drive and download in app.py:

```python
import requests
import os

MODEL_URL = "YOUR_MODEL_URL"
MODEL_PATH = 'models/ddqn_model.pth'

if not os.path.exists(MODEL_PATH):
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
```

### Memory Issues

Render free tier has 512MB RAM. If you hit limits:
- Use CPU-only PyTorch: `torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu`
- Reduce model size in training

### Port Issues

Render assigns PORT environment variable. Update app.py:

```python
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
```

## Post-Deployment

1. Check logs in Render dashboard
2. Test all endpoints:
   - `/` - Main page
   - `/predict` - Prediction API
   - `/api/comparison-data` - Metrics API
   - `/health` - Health check

3. Share your live URL!

## Custom Domain (Optional)

In Render dashboard:
1. Settings → Custom Domain
2. Add your domain
3. Update DNS records as instructed

