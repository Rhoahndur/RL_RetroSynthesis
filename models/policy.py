"""ReactionT5 policy wrapper for RL-based retrosynthesis.

Wraps the HuggingFace ReactionT5v2-retrosynthesis model with an interface
suitable for REINFORCE training: forward pass, temperature sampling,
log-probability computation, and checkpoint management.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_MODEL_NAME = "sagawa/ReactionT5v2-retrosynthesis"


class RetroPolicy:
    """RL policy wrapping ReactionT5 for retrosynthetic prediction.

    Usage:
        policy = RetroPolicy()  # downloads model on first run
        reactants = policy.predict("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        log_p = policy.log_prob("CC(C)Cc1ccc(cc1)C(C)C(=O)O", reactants[0])
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ) -> None:
        """Load the ReactionT5 model and tokenizer.

        Args:
            model_name: HuggingFace model ID or local path.
            device: Target device ("cuda", "mps", "cpu").
                    Auto-detects if None: CUDA > MPS > CPU.
        """
        self.device = device if device is not None else self.detect_device()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def _format_input(self, product_smiles: str) -> str:
        """Format a product SMILES for the ReactionT5v2 retrosynthesis model.

        The model expects the input prefixed with 'REACTANT:' — this is the
        convention from the sagawa/ReactionT5v2-retrosynthesis model card where
        the task token indicates the desired output type.

        Args:
            product_smiles: Product molecule SMILES string.

        Returns:
            Formatted input string for the model.
        """
        return "REACTANT:" + product_smiles

    def predict(
        self,
        product_smiles: str,
        num_candidates: int = 5,
        temperature: float = 1.0,
    ) -> List[str]:
        """Generate reactant SMILES candidates via temperature-scaled sampling.

        Used during RL training for exploration.

        Args:
            product_smiles: Product molecule SMILES string.
            num_candidates: Number of candidate reactant sets to generate.
            temperature: Sampling temperature. Higher = more exploration.

        Returns:
            List of reactant SMILES strings (may contain '.' for multi-reactant).
        """
        input_text = self._format_input(product_smiles)
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=num_candidates,
                max_length=512,
            )

        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return predictions

    def predict_greedy(
        self,
        product_smiles: str,
        num_beams: int = 5,
    ) -> List[str]:
        """Generate reactant SMILES via beam search (deterministic).

        Used during MCTS simulation for exploitation.

        Args:
            product_smiles: Product molecule SMILES string.
            num_beams: Beam width for beam search.

        Returns:
            List of reactant SMILES strings ranked by beam score.
        """
        input_text = self._format_input(product_smiles)
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                do_sample=False,
                max_length=512,
            )

        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Beam search returns sequences sorted by score (best first) by default
        return predictions

    def log_prob(
        self,
        product_smiles: str,
        reactant_smiles: str,
    ) -> torch.Tensor:
        """Compute log-probability of generating a specific reactant string.

        Uses teacher forcing: feeds the reactant tokens and sums the
        per-token log-probabilities.

        IMPORTANT: Returns a tensor with requires_grad=True so that
        REINFORCE can backprop through it.

        Args:
            product_smiles: Product molecule SMILES string (input).
            reactant_smiles: Reactant SMILES string (target output).

        Returns:
            Scalar tensor: sum of log-probs over the target sequence.
        """
        # Ensure model is in train mode so loss retains grad
        self.model.train()

        input_text = self._format_input(product_smiles)
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        labels = self.tokenizer(
            reactant_smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        label_ids = labels["input_ids"]
        # Count non-padding tokens in the target for converting mean loss to sum
        num_tokens = (label_ids != self.tokenizer.pad_token_id).sum()

        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=label_ids,
        )

        # outputs.loss is mean cross-entropy over tokens; convert to total log-prob
        # loss = -mean(log_probs), so log_prob = -loss * num_tokens
        log_prob = -outputs.loss * num_tokens

        return log_prob

    def get_model(self) -> AutoModelForSeq2SeqLM:
        """Return the underlying HuggingFace model (for optimizer access).

        Returns:
            The T5 model instance.
        """
        return self.model

    def save_checkpoint(
        self,
        path: str,
        step: int,
        reward: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Save model checkpoint to disk.

        Checkpoint contains: model_state_dict, optimizer_state_dict (if provided),
        step number, best reward, and model config.

        Args:
            path: File path to save the .pt checkpoint.
            step: Current training step number.
            reward: Current best reward value.
            optimizer: Optional optimizer to save state for resume.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "step": step,
            "reward": reward,
            "model_name": self.model_name,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict:
        """Load model weights from a checkpoint file.

        Args:
            path: Path to the .pt checkpoint file.

        Returns:
            Metadata dict with keys: "step", "reward", "has_optimizer".
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        return {
            "step": checkpoint.get("step", 0),
            "reward": checkpoint.get("reward", 0.0),
            "has_optimizer": "optimizer_state_dict" in checkpoint,
        }

    @staticmethod
    def detect_device() -> str:
        """Auto-detect the best available device.

        Returns:
            "cuda" if NVIDIA GPU available, "mps" if Apple Silicon, else "cpu".
        """
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
