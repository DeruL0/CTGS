from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CTLossTerms:
    volume: torch.Tensor
    occupancy: torch.Tensor
    surface: torch.Tensor
    false_hole: torch.Tensor
    false_hole_metrics: dict

    @property
    def render(self) -> torch.Tensor:
        return self.volume
