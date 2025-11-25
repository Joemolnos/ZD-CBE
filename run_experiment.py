import torch
import torch.nn as nn
import numpy as np
import time
import argparse
from typing import Dict, Optional, List

import os
import json

from core_agent import CoreAgentWithVQ, DreamEnvironment
from visualizer import RealTimeVisualizer, MetricsCollector
from data_loader import SyntheticMathDataLoader, ConceptualDataAnalyzer
from metrics import ConceptEmergenceMetrics
from evolutionary import EvolutionaryAgent, FitnessScheduler

class DreamPhase:
    """Phase 1: Dreaming - Self-organization without external data"""
    
    def __init__(self, model: CoreAgentWithVQ, config: Dict):
        self.model = model
        self.config = config
        self.dream_env = DreamEnvironment(model, vocab_size=config['vocab_size'])
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Initialize components
        self.visualization_interval = config.get('visualization_interval', 10)
        self.visualizer = None
        if self.visualization_interval and self.visualization_interval > 0:
            self.visualizer = RealTimeVisualizer(
                num_vq_codes=config.get('num_vq_codes', 64),
                max_sequence_length=config.get('max_sequence_length', 100)
            )
        self.metrics_collector = MetricsCollector()
        self.emergence_metrics = ConceptEmergenceMetrics(
            vocab_size=config['vocab_size'],
            vq_codebook_size=config.get('num_vq_codes', 64)
        )
        
        # Evolutionary optimization
        self.evolutionary = EvolutionaryAgent(
            model, 
            population_size=config.get('population_size', 5),
            mutation_rate=config.get('mutation_rate', 0.01)
        )
        
        # Training state
        self.step = 0
        self.current_sequence = ""
        self.fitness_scheduler = FitnessScheduler({
            'loss': 0.3,
            'perplexity': 0.2,
            'entropy': 0.2,
            'repetition': 0.1,
            'stability': 0.15,
            'complexity': 0.05
        })
    
    def dream_step(self) -> Dict:
        """Execute one dreaming step"""
        # Generate next token
        temperature = max(0.1, 1.0 - self.step / 1000)  # Decrease temperature over time
        next_token, vq_info = self.dream_env.step(temperature)
        
        # Create training target (predict next token from current sequence)
        if len(self.dream_env.sequence) > 1:
            input_seq = torch.tensor([self.dream_env.sequence[:-1]], dtype=torch.long)
            target_seq = torch.tensor([self.dream_env.sequence[1:]], dtype=torch.long)
            
            # Forward pass
            predictions, _, vq_info = self.model(input_seq)
            
            # Calculate loss
            loss = nn.CrossEntropyLoss()(predictions.view(-1, self.config['vocab_size']), 
                                       target_seq.view(-1))
            
            # Add VQ loss
            if 'loss' in vq_info:
                total_loss = loss + vq_info['loss']
            else:
                total_loss = loss
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            current_loss = loss.item()
            current_perplexity = vq_info.get('perplexity', 1.0)
            
            # Calculate emergence metrics
            entropy_reduction = self.emergence_metrics.calculate_entropy_reduction(predictions)
            vq_concentration = self.emergence_metrics.calculate_vq_usage_concentration(
                vq_info.get('encodings', np.zeros(64))
            )
            repetition_index = self.emergence_metrics.calculate_repetition_index(
                self.dream_env.get_sequence()
            )
            stability = self.emergence_metrics.calculate_conceptual_stability()
            complexity = self.emergence_metrics.calculate_emergent_complexity(
                self.dream_env.get_sequence(), 
                self.model.get_concept_usage()
            )
            
            # Update basic metrics
            self.emergence_metrics.update_basic_metrics(
                current_loss, current_perplexity,
                self.dream_env.get_sequence(),
                vq_info.get('encodings', np.zeros(64)),
                self.model.get_concept_usage()
            )
            
            # Calculate fitness
            fitness_metrics = {
                'loss': current_loss,
                'perplexity': current_perplexity,
                'entropy': entropy_reduction,
                'repetition': repetition_index,
                'stability': stability,
                'complexity': complexity
            }
            
            # Evolve population every N steps
            if self.step % self.config.get('evolution_interval', 100) == 0:
                evolved_model, best_fitness = self.evolutionary.evolve_population(fitness_metrics)
                self.model = evolved_model
            
            # Update visualization every N steps
            if (
                self.visualizer is not None
                and self.visualization_interval
                and self.visualization_interval > 0
                and self.step % self.visualization_interval == 0
            ):
                visualization_data = {
                    'sequence': self.dream_env.get_sequence(),
                    'vq_usage': vq_info.get('encodings', np.zeros(64)),
                    'loss': current_loss,
                    'perplexity': current_perplexity,
                    'stability': stability,
                    'entropy': entropy_reduction,
                    'repetition': repetition_index,
                    'fitness': best_fitness if self.step % self.config.get('evolution_interval', 100) == 0 else 0.0
                }
                
                self.visualizer.update_all(visualization_data)
        
        self.step += 1
        
        return {
            'step': self.step,
            'loss': current_loss if 'current_loss' in locals() else 0.0,
            'perplexity': current_perplexity if 'current_perplexity' in locals() else 1.0,
            'sequence_length': len(self.dream_env.sequence),
            'fitness': best_fitness if 'best_fitness' in locals() else 0.0
        }
    
    def run_dreaming(self, num_steps: int = 5000) -> Dict:
        """Run the complete dreaming phase"""
        print(f"Starting Dreaming Phase for {num_steps} steps...")
        
        start_time = time.time()
        
        for step in range(num_steps):
            metrics = self.dream_step()
            
            if step % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Step {step}/{num_steps} - Loss: {metrics['loss']:.4f} - "
                      f"Time: {elapsed:.2f}s")
            
            # Save checkpoint every 1000 steps
            if step % 1000 == 0 and step > 0:
                self.save_checkpoint(f'dream_checkpoint_step_{step}.pth')
        
        # Save final dream model
        self.save_checkpoint('dreamer_model.pth')
        
        # Generate comprehensive report
        report = self.emergence_metrics.generate_comprehensive_report()
        
        # Save emergence metrics
        self.emergence_metrics.save_metrics('dream_emergence_metrics.pkl')
        
        # Save visualization snapshot
        if self.visualizer is not None:
            self.visualizer.save_snapshot('dream_phase_final.png')
        
        print(f"Dreaming Phase completed in {time.time() - start_time:.2f} seconds")
        
        return report
    
    def save_checkpoint(self, filename: str):
        """Save dreaming phase checkpoint"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'step': self.step,
            'sequence': self.dream_env.sequence,
            'config': self.config
        }
        
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")

class AwakeningPhase:
    """Phase 2: Awakening - Fine-tuning on real tasks"""
    
    def __init__(self, dreamer_model: CoreAgentWithVQ, tabula_rasa_model: CoreAgentWithVQ, 
                 config: Dict):
        self.dreamer_model = dreamer_model
        self.tabula_rasa_model = tabula_rasa_model
        self.config = config
        
        # Initialize data loader
        self.data_loader = SyntheticMathDataLoader(
            task_type=config.get('task_type', 'addition'),
            max_digits=config.get('max_digits', 2)
        )
        
        # Optimizers
        self.dreamer_optimizer = torch.optim.Adam(
            dreamer_model.parameters(), 
            lr=config.get('awakening_lr', 0.001)
        )
        self.tabula_optimizer = torch.optim.Adam(
            tabula_rasa_model.parameters(), 
            lr=config.get('awakening_lr', 0.001)
        )
        
        # Metrics
        self.dreamer_metrics = []
        self.tabula_metrics = []
        
    def train_epoch(self, model: CoreAgentWithVQ, optimizer: torch.optim.Optimizer, 
                   batch_size: int = 32) -> Dict:
        """Train one epoch"""
        model.train()
        
        # Generate batch
        inputs, targets = self.data_loader.generate_batch(batch_size)
        
        # Forward pass
        predictions, _, _ = model(inputs)
        
        # Calculate loss
        loss = nn.CrossEntropyLoss()(predictions.view(-1, self.config['vocab_size']), 
                                   targets.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            eval_predictions, _, _ = model(inputs)
            decoded_predictions = self.data_loader.decode_predictions(eval_predictions)
            decoded_targets = self.data_loader.decode_predictions(
                torch.nn.functional.one_hot(targets, num_classes=self.config['vocab_size']).float()
            )
            accuracy = self.data_loader.evaluate_accuracy(decoded_predictions, decoded_targets)
        
        return {
            'loss': loss.item(),
            'character_accuracy': accuracy['character_accuracy'],
            'sequence_accuracy': accuracy['sequence_accuracy']
        }
    
    def run_awakening(self, num_epochs: int = 100) -> Dict:
        """Run the awakening phase comparison"""
        print(f"Starting Awakening Phase for {num_epochs} epochs...")
        
        # Initialize models
        self.dreamer_model.reset_hidden_state()
        self.tabula_rasa_model.reset_hidden_state()
        
        awakening_results = {
            'dreamer_performance': [],
            'tabula_performance': [],
            'comparison': {}
        }
        
        for epoch in range(num_epochs):
            # Train dreamer model
            dreamer_metrics = self.train_epoch(
                self.dreamer_model, self.dreamer_optimizer, self.config.get('batch_size', 32)
            )
            self.dreamer_metrics.append(dreamer_metrics)
            
            # Train tabula rasa model
            tabula_metrics = self.train_epoch(
                self.tabula_rasa_model, self.tabula_optimizer, self.config.get('batch_size', 32)
            )
            self.tabula_metrics.append(tabula_metrics)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}")
                print(f"Dreamer - Loss: {dreamer_metrics['loss']:.4f}, "
                      f"Accuracy: {dreamer_metrics['character_accuracy']:.4f}")
                print(f"Tabula Rasa - Loss: {tabula_metrics['loss']:.4f}, "
                      f"Accuracy: {tabula_metrics['character_accuracy']:.4f}")
                print("-" * 50)
        
        # Generate comparison report
        awakening_results['comparison'] = self.compare_models()
        
        # Save results
        self.save_awakening_results(awakening_results)
        
        return awakening_results
    
    def compare_models(self) -> Dict:
        """Compare the performance of dreamer vs tabula rasa models"""
        dreamer_final = self.dreamer_metrics[-1]
        tabula_final = self.tabula_metrics[-1]
        
        comparison = {
            'loss_improvement': dreamer_final['loss'] - tabula_final['loss'],
            'accuracy_advantage': dreamer_final['character_accuracy'] - tabula_final['character_accuracy'],
            'sequence_accuracy_advantage': dreamer_final['sequence_accuracy'] - tabula_final['sequence_accuracy'],
            'learning_speed': self.calculate_learning_speed(),
            'final_performance': {
                'dreamer': dreamer_final,
                'tabula_rasa': tabula_final
            }
        }
        
        return comparison
    
    def calculate_learning_speed(self) -> Dict:
        """Calculate learning speed metrics"""
        # Calculate time to reach 50% accuracy
        dreamer_50_time = self.find_accuracy_threshold(self.dreamer_metrics, 0.5)
        tabula_50_time = self.find_accuracy_threshold(self.tabula_metrics, 0.5)
        
        return {
            'dreamer_50pct_epochs': dreamer_50_time,
            'tabula_50pct_epochs': tabula_50_time,
            'speed_advantage': tabula_50_time - dreamer_50_time if dreamer_50_time != -1 else 0
        }
    
    def find_accuracy_threshold(self, metrics: list[Dict], threshold: float) -> int:
        """Find first epoch where accuracy exceeds threshold"""
        for i, metric in enumerate(metrics):
            if metric['character_accuracy'] >= threshold:
                return i
        return -1  # Never reached threshold
    
    def save_awakening_results(self, results: Dict):
        """Save awakening phase results"""
        # Save models
        torch.save(self.dreamer_model.state_dict(), 'awakened_dreamer_model.pth')
        torch.save(self.tabula_rasa_model.state_dict(), 'tabula_rasa_model.pth')
        
        # Save results
        with open('awakening_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("Awakening phase results saved!")

class ZDCBEExperiment:
    """Main experiment coordinator"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize models
        self.dreamer_model = CoreAgentWithVQ(
            vocab_size=config['vocab_size'],
            embed_dim=config.get('embed_dim', 128),
            hidden_dim=config.get('hidden_dim', 256),
            num_embeddings=config.get('num_vq_codes', 64),
            num_layers=config.get('num_layers', 2)
        )
        
        self._load_dream_checkpoint_if_available()

        self.tabula_rasa_model = CoreAgentWithVQ(
            vocab_size=config['vocab_size'],
            embed_dim=config.get('embed_dim', 128),
            hidden_dim=config.get('hidden_dim', 256),
            num_embeddings=config.get('num_vq_codes', 64),
            num_layers=config.get('num_layers', 2)
        )
        
        print(f"Dreamer model parameters: {self.dreamer_model.count_parameters():,}")
        print(f"Tabula Rasa model parameters: {self.tabula_rasa_model.count_parameters():,}")

    def run_complete_experiment(self) -> Dict:
        """Run the complete ZD-CBE experiment"""
        print("=" * 60)
        print("ZERO-DATA COGNITIVE BOOTSTRAP ENGINE EXPERIMENT")
        print("=" * 60)
        
        experiment_results = {
            'config': self.config,
            'dream_phase': None,
            'awakening_phase': None,
            'final_report': {}
        }
        
        # Phase 1: Dreaming
        print("\nPHASE 1: DREAMING")
        print("-" * 40)
        
        skip_dream_phase = self.config.get('skip_dream_phase', False)

        if skip_dream_phase:
            print("Skipping Dreaming Phase - loading existing artifacts...")
            dream_results = self._load_existing_dream_phase_results()
        else:
            dream_phase = DreamPhase(self.dreamer_model, self.config)
            dream_results = dream_phase.run_dreaming(self.config.get('dream_steps', 5000))
        
        experiment_results['dream_phase'] = dream_results
        
        # Phase 2: Awakening
        print("\nPHASE 2: AWAKENING")
        print("-" * 40)
        
        awakening_phase = AwakeningPhase(
            self.dreamer_model, 
            self.tabula_rasa_model, 
            self.config
        )
        awakening_results = awakening_phase.run_awakening(
            self.config.get('awakening_epochs', 100)
        )
        
        experiment_results['awakening_phase'] = awakening_results
        
        # Generate final report
        experiment_results['final_report'] = self.generate_final_report(experiment_results)
        
        # Save complete experiment
        self.save_experiment_results(experiment_results)
        
        return experiment_results

    def _load_dream_checkpoint_if_available(self):
        checkpoint_path = self.config.get('dream_checkpoint')
        if not checkpoint_path:
            return
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Dream checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint

        self.dreamer_model.load_state_dict(state_dict)
        print(f"Loaded dreamer model checkpoint from {checkpoint_path}")

    def _load_existing_dream_phase_results(self) -> Dict:
        metrics_path = self.config.get('dream_metrics_path')
        if not metrics_path or not os.path.exists(metrics_path):
            raise FileNotFoundError(
                "Dream metrics file is required to skip dream phase. "
                "Provide --dream_metrics pointing to the saved pickle."
            )

        metrics = ConceptEmergenceMetrics(
            vocab_size=self.config['vocab_size'],
            vq_codebook_size=self.config.get('num_vq_codes', 64)
        )
        metrics.load_metrics(metrics_path)

        print(f"Loaded dream phase metrics from {metrics_path}")
        return metrics.generate_comprehensive_report()
    
    def generate_final_report(self, results: Dict) -> Dict:
        """Generate comprehensive final report"""
        report = {
            'experiment_summary': {
                'total_dream_steps': self.config.get('dream_steps', 5000),
                'total_awakening_epochs': self.config.get('awakening_epochs', 100),
                'model_parameters': self.dreamer_model.count_parameters(),
                'experiment_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'dream_phase_summary': {
                'final_loss': results['dream_phase']['basic_metrics']['final_loss'],
                'final_perplexity': results['dream_phase']['basic_metrics']['final_perplexity'],
                'concept_emergence': results['dream_phase']['concept_emergence'],
                'evolution_metrics': results['dream_phase']['evolution_metrics']
            },
            'awakening_phase_summary': {
                'dreamer_advantage': results['awakening_phase']['comparison']['accuracy_advantage'],
                'learning_speed_advantage': results['awakening_phase']['comparison']['learning_speed']['speed_advantage'],
                'final_performance_comparison': results['awakening_phase']['comparison']['final_performance']
            },
            'key_findings': self.extract_key_findings(results)
        }
        
        return report
    
    def extract_key_findings(self, results: Dict) -> list[str]:
        """Extract key findings from experiment results"""
        findings = []
        
        # Dream phase findings
        final_loss = results['dream_phase']['basic_metrics']['final_loss']
        final_perplexity = results['dream_phase']['basic_metrics']['final_perplexity']
        
        if final_loss < 1.0:
            findings.append(f"✓ Self-organization achieved low prediction loss ({final_loss:.4f})")
        
        if final_perplexity < 10.0:
            findings.append(f"✓ VQ codes developed meaningful clusters (perplexity: {final_perplexity:.2f})")
        
        # Awakening phase findings
        accuracy_advantage = results['awakening_phase']['comparison']['accuracy_advantage']
        speed_advantage = results['awakening_phase']['comparison']['learning_speed']['speed_advantage']
        
        if accuracy_advantage > 0.1:
            findings.append(f"✓ Dreamer model showed {accuracy_advantage:.3f} accuracy advantage")
        
        if speed_advantage > 5:
            findings.append(f"✓ Dreamer model learned {speed_advantage} epochs faster")
        
        # Concept emergence findings
        repetition_index = results['dream_phase']['concept_emergence']['repetition_index']
        if repetition_index > 0.1:
            findings.append(f"✓ Emergent patterns detected (repetition index: {repetition_index:.3f})")
        
        stability = results['dream_phase']['concept_emergence']['conceptual_stability']
        if stability > 0.5:
            findings.append(f"✓ Conceptual stability achieved ({stability:.3f})")
        
        return findings
    
    def save_experiment_results(self, results: Dict):
        """Save complete experiment results"""
        # Create output directory
        os.makedirs('output', exist_ok=True)
        
        # Save final report
        with open('output/final_report.json', 'w') as f:
            json.dump(results['final_report'], f, indent=2, default=str)
        
        # Save complete results
        with open('output/complete_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save configuration
        with open('output/experiment_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"\nExperiment results saved to output/ directory")
        print(f"Key findings: {len(results['final_report']['key_findings'])}")

def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description='Zero-Data Cognitive Bootstrap Engine')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path')
    parser.add_argument('--dream_steps', type=int, default=5000,
                       help='Number of dreaming steps')
    parser.add_argument('--awakening_epochs', type=int, default=100,
                       help='Number of awakening epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for awakening phase')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--no_visualization', action='store_true',
                       help='Disable real-time visualization')
    parser.add_argument('--dream_checkpoint', type=str, default=None,
                       help='Path to dreamer model checkpoint to resume from')
    parser.add_argument('--dream_metrics', type=str, default=None,
                       help='Path to saved dream phase metrics pickle')
    parser.add_argument('--skip_dream_phase', action='store_true',
                       help='Skip dream phase and resume from checkpoint')
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'vocab_size': 28,
        'embed_dim': 128,
        'hidden_dim': 256,
        'num_vq_codes': 64,
        'num_layers': 2,
        'learning_rate': args.learning_rate,
        'dream_steps': args.dream_steps,
        'awakening_epochs': args.awakening_epochs,
        'batch_size': args.batch_size,
        'task_type': 'addition',
        'max_digits': 2,
        'population_size': 5,
        'mutation_rate': 0.01,
        'evolution_interval': 100,
        'visualization_interval': 10,
        'max_sequence_length': 100,
        'dream_checkpoint': args.dream_checkpoint,
        'dream_metrics_path': args.dream_metrics,
        'skip_dream_phase': args.skip_dream_phase
    }
    
    # Load custom config if provided
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)
    
    # Disable visualization if requested
    if args.no_visualization:
        config['visualization_interval'] = 0
    
    # Run experiment
    experiment = ZDCBEExperiment(config)
    results = experiment.run_complete_experiment()
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Key Findings ({len(results['final_report']['key_findings'])}):")
    for finding in results['final_report']['key_findings']:
        print(f"  {finding}")
    
    print(f"\nDreamer accuracy advantage: {results['final_report']['awakening_phase_summary']['dreamer_advantage']:.4f}")
    print(f"Learning speed advantage: {results['final_report']['awakening_phase_summary']['learning_speed_advantage']} epochs")

if __name__ == "__main__":
    main()