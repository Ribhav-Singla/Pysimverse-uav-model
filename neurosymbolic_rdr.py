"""
Neurosymbolic Ripple Down Rules (RDR) System for UAV Navigation
Integrates symbolic rules with RL learning for enhanced navigation performance.
"""

import numpy as np
import json
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

class RuleConditionType(Enum):
    LIDAR_MIN = "lidar_min"
    LIDAR_MEAN = "lidar_mean"
    DISTANCE_TO_GOAL = "distance_to_goal"
    VELOCITY_MAGNITUDE = "velocity_magnitude" 
    OBSTACLE_DIRECTION = "obstacle_direction"
    GOAL_ALIGNMENT = "goal_alignment"
    DANGER_LEVEL = "danger_level"
    CLEARANCE_FRONT = "clearance_front"
    CLEARANCE_BACK = "clearance_back"
    CLEARANCE_LEFT = "clearance_left"
    CLEARANCE_RIGHT = "clearance_right"

class ActionAdvice(Enum):
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    SLOW_DOWN = "slow_down"
    SPEED_UP = "speed_up"
    TURN_TOWARDS_GOAL = "turn_towards_goal"
    AVOID_OBSTACLE = "avoid_obstacle"
    HOVER = "hover"
    NO_ADVICE = "no_advice"

@dataclass
class RuleCondition:
    """Single condition in a rule"""
    condition_type: RuleConditionType
    operator: str  # "<", ">", "==", "<=", ">="
    threshold: float
    
    def evaluate(self, obs_features: Dict[str, float]) -> bool:
        """Evaluate if this condition is satisfied"""
        if self.condition_type.value not in obs_features:
            return False
        
        value = obs_features[self.condition_type.value]
        
        if self.operator == "<":
            return value < self.threshold
        elif self.operator == "<=":
            return value <= self.threshold
        elif self.operator == ">":
            return value > self.threshold
        elif self.operator == ">=":
            return value >= self.threshold
        elif self.operator == "==":
            return abs(value - self.threshold) < 1e-6
        else:
            return False

@dataclass
class NavigationRule:
    """Single navigation rule in RDR system"""
    rule_id: int
    conditions: List[RuleCondition]
    action_advice: ActionAdvice
    confidence: float  # 0.0 to 1.0
    priority: int  # Higher number = higher priority
    created_by: str  # "human" or "system"
    created_at: str
    usage_count: int = 0
    success_count: int = 0
    
    def matches(self, obs_features: Dict[str, float]) -> bool:
        """Check if all conditions are satisfied"""
        return all(condition.evaluate(obs_features) for condition in self.conditions)
    
    def get_success_rate(self) -> float:
        """Get success rate of this rule"""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count

class RDRKnowledgeBase:
    """Ripple Down Rules Knowledge Base for UAV Navigation"""
    
    def __init__(self, knowledge_file: str = "uav_navigation_rules.json"):
        self.knowledge_file = knowledge_file
        self.rules: List[NavigationRule] = []
        self.rule_counter = 0
        self.load_knowledge_base()
        
        # Initialize with some basic expert rules
        if not self.rules:
            self._initialize_expert_rules()
    
    def _initialize_expert_rules(self):
        """Initialize with basic expert navigation rules"""
        expert_rules = [
            # Obstacle avoidance rules
            {
                "conditions": [
                    RuleCondition(RuleConditionType.LIDAR_MIN, "<", 0.3)
                ],
                "action_advice": ActionAdvice.AVOID_OBSTACLE,
                "confidence": 0.95,
                "priority": 10,
                "created_by": "expert"
            },
            
            # Goal approach rules  
            {
                "conditions": [
                    RuleCondition(RuleConditionType.DISTANCE_TO_GOAL, "<", 1.0),
                    RuleCondition(RuleConditionType.LIDAR_MIN, ">", 0.5)
                ],
                "action_advice": ActionAdvice.TURN_TOWARDS_GOAL,
                "confidence": 0.9,
                "priority": 8,
                "created_by": "expert"
            },
            
            # Speed control rules
            {
                "conditions": [
                    RuleCondition(RuleConditionType.VELOCITY_MAGNITUDE, ">", 0.8),
                    RuleCondition(RuleConditionType.LIDAR_MIN, "<", 0.5)
                ],
                "action_advice": ActionAdvice.SLOW_DOWN,
                "confidence": 0.85,
                "priority": 9,
                "created_by": "expert"
            },
            
            # Clearance-based movement
            {
                "conditions": [
                    RuleCondition(RuleConditionType.CLEARANCE_FRONT, ">", 0.7),
                    RuleCondition(RuleConditionType.GOAL_ALIGNMENT, ">", 0.5)
                ],
                "action_advice": ActionAdvice.MOVE_FORWARD,
                "confidence": 0.8,
                "priority": 6,
                "created_by": "expert"
            },
            
            # Emergency hover rule
            {
                "conditions": [
                    RuleCondition(RuleConditionType.DANGER_LEVEL, ">", 0.8)
                ],
                "action_advice": ActionAdvice.HOVER,
                "confidence": 0.95,
                "priority": 12,
                "created_by": "expert"
            }
        ]
        
        for rule_data in expert_rules:
            self.add_rule(
                rule_data["conditions"],
                rule_data["action_advice"],
                rule_data["confidence"],
                rule_data["priority"],
                rule_data["created_by"]
            )
    
    def add_rule(self, conditions: List[RuleCondition], action_advice: ActionAdvice, 
                 confidence: float, priority: int, created_by: str) -> int:
        """Add a new rule to the knowledge base"""
        rule = NavigationRule(
            rule_id=self.rule_counter,
            conditions=conditions,
            action_advice=action_advice,
            confidence=confidence,
            priority=priority,
            created_by=created_by,
            created_at=datetime.now().isoformat()
        )
        
        self.rules.append(rule)
        self.rule_counter += 1
        self.save_knowledge_base()
        return rule.rule_id
    
    def get_applicable_rules(self, obs_features: Dict[str, float]) -> List[NavigationRule]:
        """Get all rules that match current observation features"""
        applicable_rules = []
        for rule in self.rules:
            if rule.matches(obs_features):
                applicable_rules.append(rule)
        
        # Sort by priority (descending) then confidence (descending)
        applicable_rules.sort(key=lambda r: (-r.priority, -r.confidence))
        return applicable_rules
    
    def get_rule_advice(self, obs_features: Dict[str, float]) -> Optional[Tuple[ActionAdvice, float]]:
        """Get the best rule advice for current state"""
        applicable_rules = self.get_applicable_rules(obs_features)
        
        if not applicable_rules:
            return None
        
        # Return the highest priority rule's advice
        best_rule = applicable_rules[0]
        best_rule.usage_count += 1
        return (best_rule.action_advice, best_rule.confidence)
    
    def update_rule_success(self, obs_features: Dict[str, float], success: bool):
        """Update success statistics for rules that were used"""
        applicable_rules = self.get_applicable_rules(obs_features)
        for rule in applicable_rules:
            if success:
                rule.success_count += 1
    
    def save_knowledge_base(self):
        """Save knowledge base to file"""
        try:
            data = {
                "rule_counter": self.rule_counter,
                "rules": []
            }
            
            for rule in self.rules:
                rule_data = {
                    "rule_id": rule.rule_id,
                    "conditions": [
                        {
                            "condition_type": cond.condition_type.value,
                            "operator": cond.operator,
                            "threshold": cond.threshold
                        } for cond in rule.conditions
                    ],
                    "action_advice": rule.action_advice.value,
                    "confidence": rule.confidence,
                    "priority": rule.priority,
                    "created_by": rule.created_by,
                    "created_at": rule.created_at,
                    "usage_count": rule.usage_count,
                    "success_count": rule.success_count
                }
                data["rules"].append(rule_data)
            
            with open(self.knowledge_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save knowledge base: {e}")
    
    def load_knowledge_base(self):
        """Load knowledge base from file"""
        try:
            with open(self.knowledge_file, 'r') as f:
                data = json.load(f)
            
            self.rule_counter = data.get("rule_counter", 0)
            self.rules = []
            
            for rule_data in data.get("rules", []):
                conditions = []
                for cond_data in rule_data["conditions"]:
                    condition = RuleCondition(
                        condition_type=RuleConditionType(cond_data["condition_type"]),
                        operator=cond_data["operator"],
                        threshold=cond_data["threshold"]
                    )
                    conditions.append(condition)
                
                rule = NavigationRule(
                    rule_id=rule_data["rule_id"],
                    conditions=conditions,
                    action_advice=ActionAdvice(rule_data["action_advice"]),
                    confidence=rule_data["confidence"],
                    priority=rule_data["priority"],
                    created_by=rule_data["created_by"],
                    created_at=rule_data["created_at"],
                    usage_count=rule_data.get("usage_count", 0),
                    success_count=rule_data.get("success_count", 0)
                )
                self.rules.append(rule)
                
        except FileNotFoundError:
            print(f"Knowledge base file {self.knowledge_file} not found. Starting with empty knowledge base.")
        except Exception as e:
            print(f"Error loading knowledge base: {e}")

class NeuroSymbolicIntegrator:
    """Integrates symbolic RDR advice with RL agent actions"""
    
    def __init__(self, rdr_knowledge_base: RDRKnowledgeBase, 
                 integration_weight: float = 0.3):
        self.rdr_kb = rdr_knowledge_base
        self.integration_weight = integration_weight  # How much to weight symbolic advice
        self.advice_log = []
        
    def extract_observation_features(self, observation: np.ndarray) -> Dict[str, float]:
        """Extract meaningful features from RL observation vector"""
        # Observation structure: [pos(3), vel(3), goal_dist(3), lidar_readings(16), lidar_features(11)]
        # Total: 36 dimensions
        
        pos = observation[0:3]
        vel = observation[3:6] 
        goal_dist = observation[6:9]
        lidar_readings = observation[9:25]  # 16 LIDAR rays
        lidar_features = observation[25:36]  # 11 engineered features
        
        # Extract individual LIDAR features (based on _get_obs implementation)
        min_lidar = lidar_features[0]
        mean_lidar = lidar_features[1]
        obstacle_direction_x = lidar_features[2]
        obstacle_direction_y = lidar_features[3]
        danger_level = lidar_features[4]
        clearance_front = lidar_features[5]
        clearance_back = lidar_features[6]
        clearance_left = lidar_features[7] 
        clearance_right = lidar_features[8]
        goal_alignment_x = lidar_features[9]
        goal_alignment_y = lidar_features[10]
        
        features = {
            "lidar_min": float(min_lidar),
            "lidar_mean": float(mean_lidar),
            "distance_to_goal": float(np.linalg.norm(goal_dist)),
            "velocity_magnitude": float(np.linalg.norm(vel)),
            "obstacle_direction": float(np.sqrt(obstacle_direction_x**2 + obstacle_direction_y**2)),
            "goal_alignment": float(np.sqrt(goal_alignment_x**2 + goal_alignment_y**2)),
            "danger_level": float(danger_level),
            "clearance_front": float(clearance_front),
            "clearance_back": float(clearance_back),
            "clearance_left": float(clearance_left),
            "clearance_right": float(clearance_right)
        }
        
        return features
    
    def action_advice_to_vector(self, advice: ActionAdvice, confidence: float) -> np.ndarray:
        """Convert symbolic advice to action vector modification"""
        # Action space is 3D: [vx, vy, vz] normalized to [0, 1]
        advice_vector = np.zeros(3)
        
        if advice == ActionAdvice.MOVE_FORWARD:
            advice_vector = np.array([0.8, 0.0, 0.0])
        elif advice == ActionAdvice.MOVE_BACKWARD:
            advice_vector = np.array([0.2, 0.0, 0.0])
        elif advice == ActionAdvice.MOVE_LEFT:
            advice_vector = np.array([0.0, 0.8, 0.0])
        elif advice == ActionAdvice.MOVE_RIGHT:
            advice_vector = np.array([0.0, 0.2, 0.0])
        elif advice == ActionAdvice.SLOW_DOWN:
            advice_vector = np.array([0.3, 0.3, 0.0])
        elif advice == ActionAdvice.SPEED_UP:
            advice_vector = np.array([0.9, 0.9, 0.0])
        elif advice == ActionAdvice.HOVER:
            advice_vector = np.array([0.5, 0.5, 0.0])
        elif advice == ActionAdvice.TURN_TOWARDS_GOAL:
            # This requires goal direction - handled in integrate_with_rl_action
            advice_vector = np.array([0.7, 0.7, 0.0])
        elif advice == ActionAdvice.AVOID_OBSTACLE:
            # This requires obstacle direction - handled in integrate_with_rl_action  
            advice_vector = np.array([0.4, 0.4, 0.0])
        else:  # NO_ADVICE
            advice_vector = np.array([0.5, 0.5, 0.0])
        
        return advice_vector * confidence
    
    def integrate_with_rl_action(self, rl_action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """Integrate symbolic advice with RL action"""
        features = self.extract_observation_features(observation)
        rule_advice = self.rdr_kb.get_rule_advice(features)
        
        if rule_advice is None:
            # No applicable rules, return RL action
            return rl_action
        
        advice_type, confidence = rule_advice
        symbolic_action = self.action_advice_to_vector(advice_type, confidence)
        
        # Special handling for directional advice
        if advice_type == ActionAdvice.TURN_TOWARDS_GOAL:
            goal_dist = observation[6:9]
            if np.linalg.norm(goal_dist) > 0:
                goal_direction = goal_dist / np.linalg.norm(goal_dist)
                symbolic_action[0] = 0.5 + 0.3 * goal_direction[0]  # Scale to [0.2, 0.8]
                symbolic_action[1] = 0.5 + 0.3 * goal_direction[1]
        
        elif advice_type == ActionAdvice.AVOID_OBSTACLE:
            # Move perpendicular to obstacle direction
            obs_dir_x = features.get("obstacle_direction", 0)
            if abs(obs_dir_x) > 0.1:  # Significant obstacle presence
                # Move perpendicular to obstacle
                symbolic_action[0] = 0.3 if obs_dir_x > 0 else 0.7
                symbolic_action[1] = 0.7  # Prefer moving "up" when avoiding
        
        # Weighted combination of RL and symbolic actions
        integrated_action = (1 - self.integration_weight) * rl_action + self.integration_weight * symbolic_action
        
        # Ensure action stays in [0, 1] bounds
        integrated_action = np.clip(integrated_action, 0.0, 1.0)
        
        # Log the advice for analysis
        self.advice_log.append({
            "timestamp": datetime.now().isoformat(),
            "advice_type": advice_type.value,
            "confidence": confidence,
            "rl_action": rl_action.tolist(),
            "symbolic_action": symbolic_action.tolist(),
            "integrated_action": integrated_action.tolist(),
            "features": features
        })
        
        return integrated_action
    
    def update_rule_performance(self, observation: np.ndarray, success: bool):
        """Update rule performance based on outcome"""
        features = self.extract_observation_features(observation)
        self.rdr_kb.update_rule_success(features, success)
    
    def save_advice_log(self, filename: str = "neurosymbolic_advice_log.csv"):
        """Save advice log to CSV for analysis"""
        if not self.advice_log:
            return
        
        fieldnames = ["timestamp", "advice_type", "confidence", "rl_action", 
                     "symbolic_action", "integrated_action"] + list(self.advice_log[0]["features"].keys())
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in self.advice_log:
                row = {
                    "timestamp": entry["timestamp"],
                    "advice_type": entry["advice_type"],
                    "confidence": entry["confidence"],
                    "rl_action": str(entry["rl_action"]),
                    "symbolic_action": str(entry["symbolic_action"]),
                    "integrated_action": str(entry["integrated_action"])
                }
                row.update(entry["features"])
                writer.writerow(row)

# Example usage and testing
if __name__ == "__main__":
    # Initialize RDR system
    rdr_kb = RDRKnowledgeBase("uav_navigation_rules.json")
    integrator = NeuroSymbolicIntegrator(rdr_kb, integration_weight=0.3)
    
    print(f"Initialized RDR Knowledge Base with {len(rdr_kb.rules)} rules")
    
    # Test with dummy observation
    dummy_obs = np.random.rand(36)  # 36D observation space
    dummy_rl_action = np.array([0.6, 0.4, 0.0])
    
    features = integrator.extract_observation_features(dummy_obs)
    print(f"\nExtracted features: {features}")
    
    advice = rdr_kb.get_rule_advice(features)
    if advice:
        print(f"Rule advice: {advice[0].value} (confidence: {advice[1]:.2f})")
    else:
        print("No applicable rules found")
    
    integrated_action = integrator.integrate_with_rl_action(dummy_rl_action, dummy_obs)
    print(f"\nRL Action: {dummy_rl_action}")
    print(f"Integrated Action: {integrated_action}")