# Setup Guide - Getting Started

This guide will help you set up, run, and deploy the inventory optimization project.

---

## 📦 What You Have

A **production-ready Flask web application** for DDQN inventory optimization with:

**Core Application:**
- `app.py` - Flask backend with REST API
- `model.py` - DDQN model loader and predictor
- `templates/index.html` - Responsive web interface
- `static/css/style.css` - Modern, mobile-friendly CSS
- `static/js/script.js` - Interactive frontend JavaScript

**Deployment Files:**
- `Procfile` - Render deployment configuration
- `render.yaml` - Render service setup
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules

**Research Notebooks:**
- `notebooks/inventory_optimization_deep_rl.ipynb` - Main analysis
- `notebooks/hyperparameter_tuning.ipynb` - Hyperparameter optimization

**Documentation:**
- `docs/SETUP_GUIDE.md` - This file
- `docs/DEPLOYMENT_GUIDE.md` - Detailed deployment instructions
- `docs/WEB_APP_README.md` - Web application documentation

---

## 🚀 Quick Start (3 Steps)

### Step 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/spouladchang/inventory-optimization-deep-rl.git
cd inventory-optimization-deep-rl

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Your Model

After training your DDQN in the Jupyter notebook:

```python
# In your notebook after training
ddqn_agent.save('models/ddqn_model.pth')
```

The model file should be placed in: `models/ddqn_model.pth`

### Step 3: Run Locally

```bash
# Run the Flask application
python app.py

# Open your browser to:
http://localhost:5000
```

---

## 📂 Repository Structure

```
inventory-optimization-deep-rl/
│
├── README.md                                    # Main project documentation
├── requirements.txt                             # Python dependencies
├── LICENSE                                      # MIT License
├── .gitignore                                   # Git ignore rules
│
├── notebooks/                                   # 📓 Jupyter notebooks
│   ├── inventory_optimization_deep_rl.ipynb    # Main analysis & training
│   └── hyperparameter_tuning.ipynb             # Hyperparameter search
│
├── app.py                                       # 🌐 Flask web application
├── model.py                                     # 🧠 DDQN model handler
├── Procfile                                     # Render deployment config
├── render.yaml                                  # Render service config
│
├── templates/                                   # HTML templates
│   └── index.html                              # Main web interface
│
├── static/                                      # Static assets
│   ├── css/
│   │   └── style.css                           # Responsive styling
│   └── js/
│       └── script.js                           # Frontend logic
│
├── models/                                      # 💾 Trained models
│   ├── ddqn_model.pth                          # Double DQN weights (add this)
│   └── comparison_metrics.json                 # Performance data
│
├── docs/                                        # 📚 Documentation
│   ├── SETUP_GUIDE.md                          # This file - setup instructions
│   ├── DEPLOYMENT_GUIDE.md                     # Detailed deployment guide
│   └── WEB_APP_README.md                       # Web application documentation
│
└── results/                                     # 📊 Generated outputs (gitignored)
    ├── figures/                                # Plots and charts
    ├── metrics/                                # Performance metrics
    └── final_report.txt                        # Comprehensive report
```

---

## 🎓 Usage Options

### Option 1: Run Jupyter Notebooks (Research & Training)

**Full Training** (Recommended for learning):

```bash
# Launch Jupyter
jupyter notebook

# Open notebooks/inventory_optimization_deep_rl.ipynb
# Run all cells to train all models
```

**Training Times** (CPU):
- Heuristic: Instant
- DQN: ~15 minutes
- Double DQN: ~15 minutes
- Dueling DQN: ~18 minutes
- **Total**: ~50 minutes for all models

### Option 2: Run Web Application (Deployment & Demo)

**Local Testing:**

```bash
# From project root
python app.py

# Access at: http://localhost:5000
```

**Production Deployment:**

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions on deploying to Render.

---

## 🛠️ Customization

### Modify Economic Parameters

Edit in `app.py` or notebooks:

```python
CAPACITY = 500           # Maximum storage capacity
UNIT_PRICE = 14         # Selling price per unit
UNIT_COST = 6           # Purchase cost per unit
INV_HOLDING_COST = 0.5  # Holding cost per unit per day
CUSTOMER_LOST_COST = 4  # Stockout penalty per unit
WASTE_COST = 6          # Waste cost per unit
```

### Change UI Colors

Edit `static/css/style.css`:

```css
:root {
    --primary: #2563eb;        /* Main color */
    --secondary: #10b981;      /* Secondary color */
    --accent: #f59e0b;         /* Accent color */
}
```

### Add New Features to Web App

Modify `app.py` to add:
- New API endpoints
- Authentication
- Database integration
- Additional prediction features

---

## 📋 Git Setup & Best Practices

### Initial Setup

```bash
# Initialize repository (if starting fresh)
git init
git add .
git commit -m "Initial commit: Deep RL inventory optimization"

# Connect to GitHub
git remote add origin https://github.com/spouladchang/inventory-optimization-deep-rl.git
git branch -M main
git push -u origin main
```

### Recommended Workflow

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and commit
git add .
git commit -m "Add: descriptive message"

# Push to GitHub
git push origin feature/new-feature

# Merge to main after testing
git checkout main
git merge feature/new-feature
git push origin main
```

### .gitignore Configuration

The project includes a `.gitignore` file that excludes:

```
# Python
__pycache__/
*.pyc
*.pyo

# Virtual environments
venv/
env/

# Generated outputs
results/
tuning_results/

# IDE files
.vscode/
.idea/

# OS files
.DS_Store
Thumbs.db

# Large model checkpoints (use Git LFS if needed)
*.pth.tar
```

---

## 📊 Expected Results

After setup and running, you'll have:

1. **Trained Models**: All algorithms trained and saved
2. **Performance Metrics**: Comprehensive comparison data
3. **Visualizations**: Charts and plots of results
4. **Live Web App**: Interactive prediction interface
5. **Documentation**: Complete analysis in notebooks

### Performance Benchmarks

| Metric | Heuristic | DQN | **DDQN** 🏆 | Dueling |
|--------|-----------|-----|-------------|---------|
| Weekly Profit | $9,265 | $9,725 | **$10,211** | $9,950 |
| Improvement | Baseline | +5.0% | **+10.2%** | +7.4% |
| Service Level | 94.8% | 95.2% | **95.5%** | 95.3% |

---

## 🐛 Troubleshooting

### Common Issues

**"Model file not found"**
```bash
# Ensure model exists
ls models/ddqn_model.pth

# If missing, train it in the notebook first
```

**"Module not found" errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Port 5000 already in use**
```bash
# Find and kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or use a different port in app.py
```

**Jupyter kernel issues**
```bash
# Reinstall kernel
python -m ipykernel install --user --name=inventory-rl
```

---

## 🎯 Next Steps

### Immediate Tasks

1. ✅ Run notebooks to train models
2. ✅ Test web application locally
3. ✅ Deploy to Render (see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md))
4. ✅ Share your live URL

### Optional Enhancements

- Add user authentication
- Implement prediction history logging
- Create weekly planning view
- Add export functionality (PDF/Excel)
- Integrate with restaurant POS system

### Production Considerations

- Set up monitoring (New Relic, Datadog)
- Add error tracking (Sentry)
- Implement caching (Redis)
- Add rate limiting
- Create admin dashboard

---

## 📚 Additional Resources

### Documentation

- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Detailed deployment instructions
- [WEB_APP_README.md](WEB_APP_README.md) - Web application documentation
- Main [README.md](../README.md) - Project overview

### Learning Resources

**Flask:**
- [Official Flask Tutorial](https://flask.palletsprojects.com/tutorial/)
- [Flask Mega-Tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)

**Deployment:**
- [Render Documentation](https://render.com/docs)
- [Gunicorn Guide](https://gunicorn.org/)

**Deep RL:**
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [PyTorch RL Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

---

## 💡 Tips for Success

### For Learning
1. Start with the main notebook
2. Experiment with hyperparameters
3. Try different economic parameters
4. Visualize training progress
5. Compare all algorithms

### For Portfolio
1. Deploy live demo
2. Add screenshots to README
3. Create demo video/GIF
4. Write blog post
5. Share on LinkedIn

### For Production
1. Comprehensive error handling
2. Input validation
3. Logging and monitoring
4. Regular model retraining
5. A/B testing framework

---

## 📞 Getting Help

**Issues with setup?**
- Check [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for deployment issues
- Review [WEB_APP_README.md](WEB_APP_README.md) for web app questions
- Create a GitHub Issue for bugs
- Email: saeedpooladchang78@gmail.com

**Contributing:**
- Fork the repository
- Create a feature branch
- Make your changes
- Submit a pull request

---

## ✨ Key Features

### Technical Excellence
- ✅ Clean architecture (MVC pattern)
- ✅ RESTful API design
- ✅ Responsive CSS (mobile-first)
- ✅ Modern JavaScript (ES6+)
- ✅ Production-ready deployment

### Business Value
- ✅ Real-time predictions
- ✅ AI explainability
- ✅ Performance comparisons
- ✅ Professional interface
- ✅ Actionable insights

---

**You're ready to build and deploy!** 🚀

For detailed deployment instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

---

**Built with Flask, PyTorch, and Deep Reinforcement Learning** ❤️

© 2026 Saeid Pouladchang. MIT License.
