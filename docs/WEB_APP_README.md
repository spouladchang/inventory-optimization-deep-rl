# Web Application Deployment Guide

## 🚀 Quick Start

This Flask web application provides an interactive interface for the DDQN inventory optimization model.

### Prerequisites

1. Trained DDQN model (`ddqn_final.pth` from Jupyter notebook)
2. GitHub account
3. Render account (free tier works!)

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py

# Open browser
http://localhost:5000
```

## 📂 File Structure

```
├── app.py                   # Flask application (main)
├── model.py                 # DDQN model loader & predictor
├── templates/
│   └── index.html          # Web interface
├── static/
│   ├── css/
│   │   └── style.css       # Responsive styling
│   └── js/
│       └── script.js       # Frontend logic
├── models/
│   ├── ddqn_model.pth      # YOUR TRAINED MODEL (add this!)
│   └── comparison_metrics.json  # Model comparison data
├── Procfile                 # Render deployment config
├── render.yaml             # Render service config
├── requirements.txt        # Python dependencies
└── .gitignore             # Git ignore rules
```

## 🔧 Setup Steps

### 1. Add Your Trained Model

After training in your Jupyter notebook:

```python
# In your notebook after training DDQN
ddqn_agent.save('models/ddqn_model.pth')
```

Copy this file to `models/ddqn_model.pth` in your web app directory.

### 2. Generate Comparison Metrics

Run this in your notebook after evaluating all models:

```python
import json

comparison_metrics = {
    'methods': ['Heuristic', 'DQN', 'Double DQN', 'Dueling DQN'],
    'profits': [
        float(heuristic_results['mean_profit']),
        float(dqn_eval_results['mean_profit']),
        float(ddqn_eval_results['mean_profit']),
        float(dueling_eval_results['mean_profit'])
    ],
    'improvements': [
        0,
        float(((dqn_eval_results['mean_profit'] - heuristic_results['mean_profit']) / heuristic_results['mean_profit']) * 100),
        float(((ddqn_eval_results['mean_profit'] - heuristic_results['mean_profit']) / heuristic_results['mean_profit']) * 100),
        float(((dueling_eval_results['mean_profit'] - heuristic_results['mean_profit']) / heuristic_results['mean_profit']) * 100)
    ],
    'service_levels': [
        float(heuristic_results['mean_service_level'] * 100),
        float(dqn_eval_results['mean_service_level'] * 100),
        float(ddqn_eval_results['mean_service_level'] * 100),
        float(dueling_eval_results['mean_service_level'] * 100)
    ],
    'training_progress': {
        'episodes': list(range(0, 10001, 1000)),
        'dqn': [float(x) for x in pd.Series(dqn_training_results['episode_profits']).rolling(1000).mean()[::1000].tolist()],
        'ddqn': [float(x) for x in pd.Series(ddqn_training_results['episode_profits']).rolling(1000).mean()[::1000].tolist()],
        'dueling': [float(x) for x in pd.Series(dueling_training_results['episode_profits']).rolling(1000).mean()[::1000].tolist()]
    }
}

with open('comparison_metrics.json', 'w') as f:
    json.dump(comparison_metrics, f, indent=2)
```

### 3. Test Locally

```bash
python app.py
```

Visit `http://localhost:5000` and test:
- Enter inventory level (0-500)
- Select day
- Click "Get Recommendation"
- Check results appear correctly

### 4. Deploy to Render

#### Option A: Web Dashboard (Recommended)

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add web application"
   git push origin main
   ```

2. **Deploy on Render:**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render auto-detects Python and uses `render.yaml`
   - Click "Create Web Service"
   - Wait 5-10 minutes for deployment

#### Option B: Manual Configuration

If auto-detection fails:

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`
- **Environment:** Python 3

### 5. Verify Deployment

Test these endpoints:

- `/` - Main interface
- `/health` - Should return `{"status": "healthy"}`
- `/predict` - POST endpoint (tested via UI)
- `/api/comparison-data` - Should return metrics

## 🎨 UI/UX Features

### Responsive Design
- ✅ Mobile-friendly (320px and up)
- ✅ Tablet optimized (768px)
- ✅ Desktop enhanced (1024px+)

### Accessibility
- ✅ Semantic HTML5
- ✅ ARIA labels
- ✅ Keyboard navigation
- ✅ High contrast colors

### SEO Optimized
- ✅ Meta descriptions
- ✅ Open Graph tags
- ✅ Semantic structure
- ✅ Fast loading

## 🛠️ Troubleshooting

### Model File Too Large

If `ddqn_model.pth` > 100MB:

**Solution 1: Git LFS**
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

**Solution 2: External Hosting**
Host on Google Drive/S3, download in `model.py`:

```python
import requests
import os

def download_model():
    MODEL_URL = "YOUR_GOOGLE_DRIVE_DIRECT_LINK"
    MODEL_PATH = 'models/ddqn_model.pth'
    
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        os.makedirs('models', exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("Model downloaded!")
```

### Memory Issues (Render Free Tier)

Free tier: 512MB RAM

**Solutions:**
1. Use CPU-only PyTorch:
   ```
   torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
   ```

2. Reduce model size during training

### Port Configuration

Render assigns PORT environment variable:

```python
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
```

### CORS Issues

If accessing from external domain:

```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Add this line
```

## 📊 Monitoring

### Render Dashboard

- View logs in real-time
- Monitor resource usage
- Check deployment history

### Health Checks

Render automatically pings `/health` endpoint

### Error Handling

All errors logged to Render console:
```bash
# View logs
https://dashboard.render.com → Your Service → Logs
```

## 🎯 Features

### Interactive Prediction
- Real-time DDQN recommendations
- Day-specific insights
- Confidence scoring
- Waste risk assessment

### Visual Analytics
- Training progress charts
- Performance comparisons
- Profit analysis
- Service level metrics

### Business Intelligence
- AI reasoning explanations
- Capacity utilization
- Coverage calculations
- Risk indicators

## 🔒 Security Notes

- No authentication implemented (add for production)
- Input validation on backend
- HTTPS enforced by Render
- Environment variables for sensitive data

## 📈 Next Steps

### Production Enhancements

1. **Authentication:**
   ```python
   from flask_httpauth import HTTPBasicAuth
   ```

2. **Database:**
   - Store predictions history
   - User preferences
   - Analytics

3. **Monitoring:**
   - New Relic / Datadog
   - Error tracking (Sentry)
   - Usage analytics

4. **Features:**
   - Weekly planning view
   - Historical analysis
   - Email notifications
   - API keys for integration

## 📞 Support

Issues? Check:
1. Render logs for errors
2. Browser console for frontend issues
3. GitHub Issues tab
4. Documentation at `/about`

---

**Built with ❤️ using Flask, PyTorch, and Deep Reinforcement Learning**
