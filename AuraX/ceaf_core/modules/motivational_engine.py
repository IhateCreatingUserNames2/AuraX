# ceaf_core/modules/motivational_engine.py
import time
import logging
import numpy as np
from typing import Dict, List
from collections import deque
from ceaf_core.genlang_types import MotivationalDrives, DriveState
from ceaf_core.models import DrivesConfig

logger = logging.getLogger(__name__)


class MotivationalEngine:
    """
    Motor Motivacional V3 (Síntese).
    Combina a integração com o ecossistema CEAF com mecânicas de aprendizado
    e estabilidade inspiradas na análise da Claude.
    """

    def __init__(self, config: DrivesConfig = None):
        # Garante que config seja uma instância de DrivesConfig
        self.config = config or DrivesConfig()
        self.performance_history: deque[Dict] = deque(maxlen=20)
        logger.info("MotivationalEngine (V3.1 Configurable) inicializado.")

    def update_drives(self, drives: MotivationalDrives, metrics: dict) -> MotivationalDrives:
        updated_drives = drives.copy(deep=True)

        # FASE 1: Mudança passiva (decaimento e crescimento)
        self._apply_temporal_decay(updated_drives)

        # FASE 2: Reação a eventos do turno (feedback)
        self._apply_feedback(updated_drives, metrics)

        # FASE 3: Atualiza momentum (tendências)
        self._update_momentum(updated_drives, drives)

        # FASE 4: Normalização suave (evita clipping)
        self._soft_normalize_drives(updated_drives)

        # FASE 5: Meta-aprendizado (ajusta efetividade)
        self._log_performance(metrics, updated_drives)
        if len(self.performance_history) == self.performance_history.maxlen:
            self._meta_learn(updated_drives)

        updated_drives.last_updated = time.time()
        self._log_update_summary(updated_drives)
        return updated_drives

    def _apply_temporal_decay(self, drives: MotivationalDrives):
        time_delta_hours = (time.time() - drives.last_updated) / 3600

        # CORREÇÃO: Acesso via ponto (.)
        decay_factor = self.config.passive_decay_rate * time_delta_hours

        drives.mastery.intensity += (0.5 - drives.mastery.intensity) * decay_factor
        drives.consistency.intensity += (0.5 - drives.consistency.intensity) * decay_factor

        drives.curiosity.intensity += self.config.passive_curiosity_increase * time_delta_hours
        drives.connection.intensity += self.config.passive_connection_increase * time_delta_hours

    def _apply_feedback(self, drives: MotivationalDrives, metrics: dict):
        """Aplica feedback baseado no resultado do turno."""
        cfg = self.config
        vre_rejections = metrics.get("vre_rejection_count", 0)
        final_confidence = metrics.get("final_confidence", 0.0)

        # Lógica de Falha
        if vre_rejections > 0:
            # CORREÇÃO: cfg.parametro em vez de cfg['parametro']
            boost = cfg.consistency_boost_on_failure * drives.consistency.effectiveness_score
            drives.consistency.intensity += boost
            logger.info(f"MOTIVATION (Failure): Aumentando Consistency (+{boost:.2f})")

        # Lógica de Sucesso
        if vre_rejections == 0 and final_confidence > 0.75:
            # 1. Reduz a Maestria (o agente fica "satisfeito" por ter acertado)
            satisfaction = cfg.mastery_satisfaction_on_success
            drives.mastery.intensity -= satisfaction

            # 2. Aumenta a Consistência (reforça o comportamento que deu certo)
            # Usamos getattr por segurança, mas como você adicionou no modelo, cfg.consistency_boost_on_success funcionaria.
            consistency_boost = getattr(cfg, 'consistency_boost_on_success', 0.10)

            # Multiplicamos pelo score de eficácia para manter o padrão do sistema (aprendizado)
            consistency_boost *= drives.consistency.effectiveness_score

            drives.consistency.intensity += consistency_boost

            logger.info(
                f"MOTIVATION (Success): Mastery satisfeito (-{satisfaction:.2f}), Consistency reforçado (+{consistency_boost:.2f})")

        # Lógica de Mastery (baseada na SURPRESA, vinda do LCAM)
        prediction_error = metrics.get("prediction_error_signal", {}).get("total_error", 0.0)
        if prediction_error > 0.2:
            # CORREÇÃO: No models.py padrão, 'mastery_boost_on_prediction_error' não existe explicitamente
            # no snippet fornecido anteriormente, mas estava no dicionário padrão.
            # Se não estiver no models.py, use um valor hardcoded ou adicione ao modelo.
            # Assumindo que você quer usar o valor do config ou um default:
            boost_factor = getattr(cfg, 'mastery_boost_on_prediction_error', 0.5)

            boost = boost_factor * prediction_error * drives.mastery.effectiveness_score
            drives.mastery.intensity += boost
            logger.warning(
                f"MOTIVATION (Mastery): Erro de predição ({prediction_error:.2f})! Aumentando Mastery (+{boost:.2f}) para 'entender'.")

        # Lógica de Curiosidade
        if metrics.get("topic_shifted_this_turn", False):
            # CORREÇÃO: Acesso via ponto
            drives.curiosity.intensity -= cfg.curiosity_satisfaction_on_topic_shift

        if metrics.get("relevant_memories_count", 5) < 2:
            # CORREÇÃO: Acesso via atributo ou getattr se for dinâmico
            boost_val = getattr(cfg, 'curiosity_boost_on_low_memory', 0.06)
            boost = boost_val * drives.curiosity.effectiveness_score
            drives.curiosity.intensity += boost

    def _update_momentum(self, new_drives: MotivationalDrives, old_drives: MotivationalDrives):
        """Calcula a taxa de mudança de cada drive."""
        # CORREÇÃO: getattr com default ou acesso direto se estiver no modelo
        momentum_decay = getattr(self.config, 'momentum_decay', 0.7)

        for drive_name in ["curiosity", "connection", "mastery", "consistency"]:
            new_drive: DriveState = getattr(new_drives, drive_name)
            old_drive: DriveState = getattr(old_drives, drive_name)

            change = new_drive.intensity - old_drive.intensity
            new_drive.momentum = (old_drive.momentum * momentum_decay) + change

    def _soft_normalize_drives(self, drives: MotivationalDrives):
        """Normalização suave que puxa gradualmente valores extremos de volta ao intervalo [0, 1]."""
        for drive_name in ["curiosity", "connection", "mastery", "consistency"]:
            drive: DriveState = getattr(drives, drive_name)

            if drive.intensity < 0:
                drive.intensity = 0.1 / (1 + np.exp(-drive.intensity * 5))
            elif drive.intensity > 1:
                drive.intensity = 1 - 0.1 / (1 + np.exp((drive.intensity - 1) * 5))

    def _log_performance(self, metrics: dict, final_drives: MotivationalDrives):
        """Registra o resultado do turno para meta-aprendizado."""
        is_success = metrics.get("vre_rejection_count", 0) == 0 and metrics.get("final_confidence", 0) > 0.65

        self.performance_history.append({
            "success": is_success,
            "drive_intensities": {
                "mastery": final_drives.mastery.intensity,
                "curiosity": final_drives.curiosity.intensity,
                "connection": final_drives.connection.intensity,
                "consistency": final_drives.consistency.intensity,
            }
        })

    def _meta_learn(self, current_drives: MotivationalDrives):
        """Ajusta o 'effectiveness_score' de cada drive com base no histórico de performance."""
        logger.critical("META-LEARNING: Iniciando ajuste da eficácia dos drives...")

        success_outcomes = [o for o in self.performance_history if o["success"]]
        failure_outcomes = [o for o in self.performance_history if not o["success"]]

        if not success_outcomes or not failure_outcomes:
            logger.info("META-LEARNING: Dados insuficientes para comparação (precisa de sucessos e falhas).")
            return

        # CORREÇÃO: getattr com default
        meta_learning_rate = getattr(self.config, 'meta_learning_rate', 0.05)

        for drive_name in ["curiosity", "connection", "mastery", "consistency"]:
            avg_on_success = np.mean([o["drive_intensities"][drive_name] for o in success_outcomes])
            avg_on_failure = np.mean([o["drive_intensities"][drive_name] for o in failure_outcomes])

            effectiveness_signal = avg_on_success - avg_on_failure

            drive_state: DriveState = getattr(current_drives, drive_name)
            current_effectiveness = drive_state.effectiveness_score

            adjustment = meta_learning_rate * effectiveness_signal
            new_effectiveness = current_effectiveness + adjustment

            # Garante que os pesos fiquem em um intervalo razoável (ex: 0.5 a 1.5)
            drive_state.effectiveness_score = max(0.5, min(1.5, new_effectiveness))

        logger.warning(f"META-LEARNING: Novos scores de eficácia -> "
                       f"Mastery: {current_drives.mastery.effectiveness_score:.2f}, "
                       f"Curiosity: {current_drives.curiosity.effectiveness_score:.2f}, "
                       f"Consistency: {current_drives.consistency.effectiveness_score:.2f}")

        self.performance_history.clear()

    def _log_update_summary(self, drives: MotivationalDrives):
        """Log consolidado do estado final."""
        summary = "\n" + "=" * 60 + "\nMOTIVATIONAL UPDATE (V3 Synthesis)\n" + "=" * 60
        for drive_name in ["mastery", "curiosity", "connection", "consistency"]:
            drive: DriveState = getattr(drives, drive_name)
            summary += (f"\n{drive_name.capitalize():<12}: "
                        f"{drive.intensity:.3f} "
                        f"(momentum: {drive.momentum:+.3f}, "
                        f"effectiveness: {drive.effectiveness_score:.3f})")
        summary += "\n" + "=" * 60
        logger.info(summary)