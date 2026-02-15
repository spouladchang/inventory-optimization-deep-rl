"""
Model handler for inventory optimization predictions
Loads trained DDQN model and generates recommendations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


# Neural network architecture (must match training)
class DQNNetwork(nn.Module):
    """Standard DQN Network"""
    
    def __init__(self, state_dim: int = 2, action_dim: int = 11, hidden_dims: list = [256, 256]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)


class InventoryOptimizer:
    """
    Inventory optimization using trained DDQN model
    """
    
    def __init__(self, model_path: str):
        """
        Initialize optimizer with trained model
        
        Args:
            model_path: Path to saved model checkpoint
        """
        self.device = torch.device('cpu')  # Use CPU for web deployment
        self.capacity = 500
        self.order_levels = list(range(0, 501, 50))  # [0, 50, 100, ..., 500]
        self.days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()  # Set to evaluation mode
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint"""
        try:
            # Initialize model
            model = DQNNetwork(state_dim=2, action_dim=len(self.order_levels))
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'policy_net' in checkpoint:
                model.load_state_dict(checkpoint['policy_net'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            return model
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def _get_state(self, day: str, inventory: int) -> np.ndarray:
        """
        Convert day and inventory to model input state
        
        Args:
            day: Day of week (Monday-Friday)
            inventory: Current inventory level
            
        Returns:
            State array [day_normalized, inventory_normalized]
        """
        day_idx = self.days.index(day)
        day_normalized = day_idx / (len(self.days) - 1)
        inventory_normalized = inventory / self.capacity
        
        return np.array([day_normalized, inventory_normalized], dtype=np.float32)
    
    def _get_valid_actions(self, inventory: int) -> list:
        """Get valid order actions that don't exceed capacity"""
        valid = []
        for i, order in enumerate(self.order_levels):
            if inventory + order <= self.capacity:
                valid.append(i)
        return valid if valid else [0]
    
    def _get_confidence_level(self, q_values: np.ndarray, best_action: int) -> Tuple[str, float]:
        """
        Determine confidence level of recommendation
        
        Returns:
            (confidence_label, confidence_score)
        """
        # Get Q-value difference between best and second-best
        q_sorted = np.sort(q_values)
        if len(q_values) > 1:
            q_diff = q_sorted[-1] - q_sorted[-2]
            q_range = q_sorted[-1] - q_sorted[0]
            
            if q_range > 0:
                confidence_score = (q_diff / q_range) * 100
            else:
                confidence_score = 100
        else:
            confidence_score = 100
        
        # Categorize confidence
        if confidence_score >= 70:
            confidence_label = "High"
        elif confidence_score >= 40:
            confidence_label = "Medium"
        else:
            confidence_label = "Low"
        
        return confidence_label, confidence_score
    
    def _generate_reasoning(self, day: str, current_inventory: int, 
                          order_quantity: int, predicted_inventory: int) -> str:
        """Generate human-readable reasoning for recommendation"""
        reasons = []
        
        # Day-specific reasoning
        if day == "Monday":
            reasons.append("Start of week - building up inventory for peak demand.")
        elif day == "Friday":
            reasons.append("End of week - minimizing waste from weekend closure.")
        else:
            reasons.append(f"Mid-week ({day}) - maintaining optimal inventory flow.")
        
        # Inventory level reasoning
        if current_inventory < 100:
            reasons.append(f"Low current stock ({current_inventory} units) requires significant replenishment.")
        elif current_inventory < 250:
            reasons.append(f"Moderate stock ({current_inventory} units) - standard restocking recommended.")
        else:
            reasons.append(f"Good stock level ({current_inventory} units) - minimal order needed.")
        
        # Order quantity reasoning
        if order_quantity == 0:
            reasons.append("No order needed - inventory sufficient for expected demand.")
        elif order_quantity >= 300:
            reasons.append(f"Large order ({order_quantity} units) to meet anticipated high demand.")
        else:
            reasons.append(f"Standard order ({order_quantity} units) for normal operations.")
        
        # Target reasoning
        reasons.append(f"Target: {predicted_inventory} units (optimal for {day} operations).")
        
        return " ".join(reasons)
    
    def get_recommendation(self, current_inventory: int, day: str) -> Dict:
        """
        Get ordering recommendation for given state
        
        Args:
            current_inventory: Current inventory level (0-500)
            day: Day of week (Monday-Friday)
            
        Returns:
            Dictionary with recommendation details
        """
        # Prepare state
        state = self._get_state(day, current_inventory)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get valid actions
        valid_actions = self._get_valid_actions(current_inventory)
        
        # Get Q-values from model
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy().flatten()
        
        # Mask invalid actions
        masked_q = np.full_like(q_values, -np.inf)
        masked_q[valid_actions] = q_values[valid_actions]
        
        # Select best action
        best_action_idx = int(np.argmax(masked_q))
        order_quantity = self.order_levels[best_action_idx]
        
        # Calculate predicted inventory
        predicted_inventory = min(current_inventory + order_quantity, self.capacity)
        
        # Get confidence
        valid_q_values = q_values[valid_actions]
        confidence_label, confidence_score = self._get_confidence_level(
            valid_q_values, best_action_idx
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            day, current_inventory, order_quantity, predicted_inventory
        )
        
        # Expected demand (average from training data)
        expected_demand = 250
        
        # Calculate coverage
        coverage_days = predicted_inventory / expected_demand if expected_demand > 0 else 0
        
        return {
            'order_quantity': int(order_quantity),
            'day': day,
            'current_inventory': int(current_inventory),
            'predicted_inventory_after_order': int(predicted_inventory),
            'confidence': confidence_label,
            'confidence_score': float(round(confidence_score, 1)),
            'reasoning': reasoning,
            'expected_demand': int(expected_demand),
            'coverage_days': float(round(coverage_days, 2)),
            'capacity_utilization': float(round((predicted_inventory / self.capacity) * 100, 1)),
            'waste_risk': 'High' if day == 'Friday' and predicted_inventory > 300 else 
                         'Medium' if predicted_inventory > 400 else 'Low'
        }
    
    def get_weekly_plan(self, starting_inventory: int) -> list:
        """
        Generate ordering plan for entire week
        
        Args:
            starting_inventory: Inventory at start of week
            
        Returns:
            List of recommendations for each day
        """
        inventory = starting_inventory
        weekly_plan = []
        
        for day in self.days:
            recommendation = self.get_recommendation(inventory, day)
            weekly_plan.append(recommendation)
            
            # Simulate inventory for next day (simplified)
            # Order -> sell average demand -> carry over
            inventory = recommendation['predicted_inventory_after_order']
            inventory = max(0, inventory - 250)  # Subtract average demand
        
        return weekly_plan
