# checkpoint_manager.py

import os
import json
import torch
import shutil
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, List
import logging

@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint validation and management"""
    step: int
    timestamp: datetime
    loss: float
    validation_loss: Optional[float]
    model_hash: str
    is_merged: bool = False
    is_validated: bool = False
    validation_errors: List[str] = None
    partial_save: bool = False
    size_bytes: int = 0

class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        metrics_dir: str,
        max_checkpoints: int = 5,
        validation_frequency: int = 5,
        merge_threshold: int = 3,
        min_improvement: float = 0.01
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metrics_dir = Path(metrics_dir)
        self.max_checkpoints = max_checkpoints
        self.validation_frequency = validation_frequency
        self.merge_threshold = merge_threshold
        self.min_improvement = min_improvement
        
        # Setup logging
        self.logger = logging.getLogger('CheckpointManager')
        self.setup_logging()
        
        # Initialize directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing checkpoint metadata
        self.checkpoints_meta = self._load_checkpoints_metadata()
        
    def setup_logging(self):
        """Configure logging for checkpoint operations"""
        handler = logging.FileHandler(self.checkpoint_dir / 'checkpoint_manager.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _compute_model_hash(self, model_path: Path) -> str:
        """Compute hash of model weights for integrity checking"""
        hasher = hashlib.sha256()
        with open(model_path / "pytorch_model.bin", 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _validate_checkpoint(self, trainer, checkpoint_path: Path) -> List[str]:
        """Validate checkpoint integrity and usability"""
        errors = []
        try:
            # Basic file integrity
            required_files = ["pytorch_model.bin", "config.json", "checkpoint-state.json"]
            for file in required_files:
                if not (checkpoint_path / file).exists():
                    errors.append(f"Missing required file: {file}")

            if not errors:
                # Load checkpoint in temporary trainer
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_trainer = type(trainer)(
                        model=trainer.model.__class__(trainer.model.config),
                        args=trainer.args,
                        train_dataset=trainer.train_dataset[:10]  # Small subset for testing
                    )
                    
                    try:
                        # Test loading
                        self.load_checkpoint(tmp_trainer, checkpoint_path)
                        
                        # Test forward pass
                        batch = next(iter(tmp_trainer.get_train_dataloader()))
                        tmp_trainer.model(**batch)
                    except Exception as e:
                        errors.append(f"Checkpoint validation failed: {str(e)}")

        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return errors

    def _merge_checkpoints(self, source_paths: List[Path], target_path: Path):
        """Merge multiple partial checkpoints into a complete one"""
        self.logger.info(f"Merging checkpoints into {target_path}")
        
        # Create merged directory
        target_path.mkdir(exist_ok=True)
        
        # Load and merge model states
        merged_state = {}
        for source in source_paths:
            state_dict = torch.load(source / "pytorch_model.bin")
            merged_state.update(state_dict)
            
        # Save merged state
        torch.save(merged_state, target_path / "pytorch_model.bin")
        
        # Copy metadata from most recent checkpoint
        latest_source = max(source_paths, key=lambda p: p.stat().st_mtime)
        for file in ["config.json", "checkpoint-state.json"]:
            if (latest_source / file).exists():
                shutil.copy2(latest_source / file, target_path / file)
                
        # Update metadata
        meta = self._load_checkpoint_metadata(target_path)
        meta.is_merged = True
        meta.partial_save = False
        self._save_checkpoint_metadata(target_path, meta)
        
        # Clean up source checkpoints
        for source in source_paths:
            shutil.rmtree(source)

    def _prune_checkpoints(self):
        """Remove old checkpoints keeping only the best ones"""
        checkpoints = sorted(
            self.checkpoints_meta.items(),
            key=lambda x: (x[1].loss, -x[1].step)
        )
        
        # Always keep the latest checkpoint
        latest_checkpoint = max(
            self.checkpoints_meta.items(),
            key=lambda x: x[1].step
        )
        
        to_remove = []
        if len(checkpoints) > self.max_checkpoints:
            for path, meta in checkpoints[self.max_checkpoints:]:
                if path != latest_checkpoint[0]:
                    to_remove.append(path)
        
        for path in to_remove:
            self.logger.info(f"Pruning checkpoint: {path}")
            shutil.rmtree(path)
            del self.checkpoints_meta[path]

    def save_checkpoint(
        self,
        trainer,
        step: int,
        training_args,
        loss: float,
        validation_loss: Optional[float] = None,
        partial: bool = False
    ):
        """Save checkpoint with validation and management"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model and training state
        trainer.save_model(checkpoint_path)
        torch.save(trainer.optimizer.state_dict(), checkpoint_path / "optimizer.pt")
        if trainer.lr_scheduler:
            torch.save(trainer.lr_scheduler.state_dict(), checkpoint_path / "scheduler.pt")
            
        # Create metadata
        meta = CheckpointMetadata(
            step=step,
            timestamp=datetime.now(),
            loss=float(loss),
            validation_loss=validation_loss,
            model_hash=self._compute_model_hash(checkpoint_path),
            partial_save=partial,
            size_bytes=sum(f.stat().st_size for f in checkpoint_path.rglob('*'))
        )
        
        # Validate if needed
        if step % self.validation_frequency == 0:
            meta.validation_errors = self._validate_checkpoint(trainer, checkpoint_path)
            meta.is_validated = len(meta.validation_errors) == 0
            
        self._save_checkpoint_metadata(checkpoint_path, meta)
        self.checkpoints_meta[checkpoint_path] = meta
        
        # Consider merging partial checkpoints
        partial_checkpoints = [
            p for p, m in self.checkpoints_meta.items() 
            if m.partial_save and not m.is_merged
        ]
        if len(partial_checkpoints) >= self.merge_threshold:
            self._merge_checkpoints(partial_checkpoints, checkpoint_path)
            
        # Prune old checkpoints
        self._prune_checkpoints()
        
        return checkpoint_path

    def load_checkpoint(
        self,
        trainer,
        checkpoint_path: Optional[Path] = None,
        validate: bool = True
    ) -> Optional[int]:
        """Load checkpoint with validation"""
        if not checkpoint_path:
            return None
            
        if validate:
            errors = self._validate_checkpoint(trainer, checkpoint_path)
            if errors:
                self.logger.error(f"Checkpoint validation failed: {errors}")
                return None
                
        # Load model weights
        trainer.model.load_state_dict(
            torch.load(checkpoint_path / "pytorch_model.bin")
        )
        
        # Load optimizer state
        if (checkpoint_path / "optimizer.pt").exists():
            trainer.optimizer.load_state_dict(
                torch.load(checkpoint_path / "optimizer.pt")
            )
            
        # Load scheduler state
        if (checkpoint_path / "scheduler.pt").exists() and trainer.lr_scheduler:
            trainer.lr_scheduler.load_state_dict(
                torch.load(checkpoint_path / "scheduler.pt")
            )
            
        # Load training state
        meta = self._load_checkpoint_metadata(checkpoint_path)
        if meta:
            return meta.step
            
        return None

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get the best checkpoint based on validation loss"""
        valid_checkpoints = [
            (p, m) for p, m in self.checkpoints_meta.items()
            if m.is_validated and m.validation_loss is not None
        ]
        
        if not valid_checkpoints:
            return None
            
        return min(
            valid_checkpoints,
            key=lambda x: x[1].validation_loss
        )[0]

    def test_checkpoint(self, trainer, checkpoint_path: Path) -> bool:
        """Test checkpoint with a sample forward pass"""
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_trainer = type(trainer)(
                    model=trainer.model.__class__(trainer.model.config),
                    args=trainer.args,
                    train_dataset=trainer.train_dataset[:1]
                )
                
                self.load_checkpoint(tmp_trainer, checkpoint_path, validate=False)
                batch = next(iter(tmp_trainer.get_train_dataloader()))
                tmp_trainer.model(**batch)
                return True
        except Exception as e:
            self.logger.error(f"Checkpoint test failed: {str(e)}")
            return False

    def get_checkpoint_info(self) -> Dict:
        """Get information about all checkpoints"""
        return {
            str(path): {
                'step': meta.step,
                'timestamp': meta.timestamp.isoformat(),
                'loss': meta.loss,
                'validation_loss': meta.validation_loss,
                'is_validated': meta.is_validated,
                'is_merged': meta.is_merged,
                'size_mb': meta.size_bytes / (1024 * 1024),
                'validation_errors': meta.validation_errors
            }
            for path, meta in self.checkpoints_meta.items()
        }