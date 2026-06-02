from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CTLossTerms:
    volume: torch.Tensor
    occupancy: torch.Tensor
    surface: torch.Tensor

    @property
    def render(self) -> torch.Tensor:
        return self.volume
