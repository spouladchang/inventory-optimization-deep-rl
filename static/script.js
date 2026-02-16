// Main JavaScript for Inventory Optimizer Web App

// Inventory-Order synchronization
const inventoryInput = document.getElementById('inventory');
const inventorySlider = document.getElementById('inventorySlider');

if (inventoryInput && inventorySlider) {
    inventoryInput.addEventListener('input', (e) => {
        inventorySlider.value = e.target.value;
    });
    
    inventorySlider.addEventListener('input', (e) => {
        inventoryInput.value = e.target.value;
    });
}

// Form submission
const form = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const resultCard = document.getElementById('resultCard');
const errorCard = document.getElementById('errorCard');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const inventory = parseInt(inventoryInput.value);
    const day = document.getElementById('day').value;
    
    // Show loading state
    predictBtn.disabled = true;
    document.querySelector('.btn-text').style.display = 'none';
    document.querySelector('.btn-loading').style.display = 'flex';
    
    // Hide previous results
    resultCard.style.display = 'none';
    errorCard.style.display = 'none';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                current_inventory: inventory,
                day: day
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data.recommendation);
        } else {
            displayError(data.error);
        }
    } catch (error) {
        displayError('Network error: ' + error.message);
    } finally {
        // Reset button state
        predictBtn.disabled = false;
        document.querySelector('.btn-text').style.display = 'block';
        document.querySelector('.btn-loading').style.display = 'none';
    }
});

function displayResults(rec) {
    // Populate result fields
    document.getElementById('orderQuantity').textContent = rec.order_quantity;
    document.getElementById('confidenceText').textContent = rec.confidence;
    document.getElementById('currentInv').textContent = rec.current_inventory + ' units';
    document.getElementById('afterOrderInv').textContent = rec.predicted_inventory_after_order + ' units';
    document.getElementById('expectedDemand').textContent = rec.expected_demand + ' units';
    document.getElementById('coverage').textContent = rec.coverage_days.toFixed(1) + ' days';
    document.getElementById('capacity').textContent = rec.capacity_utilization + '%';
    document.getElementById('wasteRisk').textContent = rec.waste_risk;
    document.getElementById('wasteRisk').className = 'detail-value risk-badge ' + rec.waste_risk;
    document.getElementById('reasoning').textContent = rec.reasoning;
    
    // Set confidence badge style
    const confidenceBadge = document.getElementById('confidenceBadge');
    confidenceBadge.className = 'confidence-badge';
    if (rec.confidence === 'High') {
        confidenceBadge.style.background = 'rgba(16, 185, 129, 0.2)';
    } else if (rec.confidence === 'Medium') {
        confidenceBadge.style.background = 'rgba(245, 158, 11, 0.2)';
    } else {
        confidenceBadge.style.background = 'rgba(239, 68, 68, 0.2)';
    }
    
    // Show result card
    resultCard.style.display = 'block';
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function displayError(message) {
    document.getElementById('errorMessage').textContent = message;
    errorCard.style.display = 'block';
    errorCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Load and display comparison charts
async function loadComparisonData() {
    try {
        const response = await fetch('/api/comparison-data');
        const data = await response.json();
        
        if (data.success) {
            createCharts(data.data);
            updateMetrics(data.data);
        }
    } catch (error) {
        console.error('Error loading comparison data:', error);
    }
}

function updateMetrics(data) {
    const methods = data.methods;
    const profits = data.profits;
    
    document.getElementById('heuristicProfit').textContent = '$' + profits[0].toLocaleString();
    document.getElementById('dqnProfit').textContent = '$' + profits[1].toLocaleString();
    document.getElementById('dqnImprovement').textContent = '+' + data.improvements[1] + '%';
    document.getElementById('ddqnProfit').textContent = '$' + profits[2].toLocaleString();
    document.getElementById('ddqnImprovement').textContent = '+' + data.improvements[2] + '%';
    document.getElementById('duelingProfit').textContent = '$' + profits[3].toLocaleString();
    document.getElementById('duelingImprovement').textContent = '+' + data.improvements[3] + '%';
}

function createCharts(data) {
    // Training Progress Chart
    const trainingCtx = document.getElementById('trainingChart').getContext('2d');
    new Chart(trainingCtx, {
        type: 'line',
        data: {
            labels: data.training_progress.episodes,
            datasets: [
                {
                    label: 'DQN',
                    data: data.training_progress.dqn,
                    borderColor: '#ff7f0e',
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'Double DQN (Winner)',
                    data: data.training_progress.ddqn,
                    borderColor: '#2ca02c',
                    tension: 0.4,
                    fill: false,
                    borderWidth: 3
                },
                {
                    label: 'Dueling DQN',
                    data: data.training_progress.dueling,
                    borderColor: '#9467bd',
                    tension: 0.4,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'top'
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Weekly Profit ($)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Training Episodes'
                    }
                }
            }
        }
    });
    
    // Metrics Comparison Chart
    const metricsCtx = document.getElementById('metricsChart').getContext('2d');
    new Chart(metricsCtx, {
        type: 'bar',
        data: {
            labels: data.methods,
            datasets: [{
                label: 'Service Level (%)',
                data: data.service_levels,
                backgroundColor: '#2563eb'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 90,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Service Level (%)'
                    }
                }
            }
        }
    });
}

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Load comparison data on page load
window.addEventListener('DOMContentLoaded', () => {
    loadComparisonData();
});
