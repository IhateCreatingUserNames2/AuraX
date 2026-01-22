# ceaf_core/monadic_base.py
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Generic, TypeVar, Callable, Any, Optional, List, Awaitable
import logging
import numpy as np
from typing import List, Optional, Any

S = TypeVar("S")  # AuraState
V = TypeVar("V")  # Valor atual no pipeline

logger = logging.getLogger("AuraV4_Monad")


@dataclass(frozen=True)
class AuraState:
    agent_id: str
    session_id: str
    identity_glyph: List[float]
    xi: float = 0.0
    memory_context: str = ""
    metadata: dict = field(default_factory=dict)

    def copy(self, update: dict = None, **kwargs):
        """
        Emula o comportamento do .copy(update=...) do Pydantic
        para que o fluxo monádico funcione sem erros.
        """
        changes = update if update else {}
        changes.update(kwargs)
        return replace(self, **changes)


@dataclass(frozen=True)
class AuraMonad(Generic[S, V]):
    state: S
    value: V | None
    xi: float = 0.0
    is_successful: bool = True
    error_info: Optional[str] = None

    async def then(self, func: Callable[[S, V], Awaitable[AuraMonad[S, Any]]]) -> AuraMonad[S, Any]:
        """Encadeia o próximo passo de forma ASSÍNCRONA."""
        if not self.is_successful:
            return self

        # Proteção contra Colapso Epistêmico (Threshold de Integridade V4)
        if self.xi > 0.85:
            return AuraMonad.failure(self.state, "Tensão Epistêmica muito alta para prosseguir.")

        try:
            # Chama a função passando o estado e o valor atual
            return await func(self.state, self.value)
        except Exception as e:
            return AuraMonad.failure(self.state, str(e))

    @staticmethod
    def start(state: S, value: V) -> AuraMonad[S, V]:
        return AuraMonad(state, value)

    @staticmethod
    def failure(state: S, error: str) -> AuraMonad[S, Any]:
        return AuraMonad(state, None, xi=1.0, is_successful=False, error_info=error)

    @staticmethod
    def success(state: S, value: Any, xi: float = 0.0) -> AuraMonad[S, Any]:
        return AuraMonad(state, value, xi=xi, is_successful=True)