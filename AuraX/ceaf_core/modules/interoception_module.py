# ceaf_core/modules/interoception_module.py
from ceaf_core.genlang_types import InternalStateReport
import logging

logger = logging.getLogger("CEAFv3_Interoception")


class ComputationalInteroception:
    """
    Gera um relatório do estado interno do agente com base em métricas do turno.
    Versão V2: Cálculos mais sutis e graduais.
    """

    def generate_internal_state_report(self, metrics: dict) -> InternalStateReport:
        """
        Calcula os componentes do estado interno de forma gradativa.

        Args:
            metrics: Dicionário com métricas do turno, incluindo:
                - agency_score: Score de agência (0-10)
                - used_mycelial_path: Bool indicando uso de consenso micelial
                - vre_rejection_count: Número de rejeições do VRE
                - final_confidence: Confiança final (0.0-1.0)
                - vre_flags: Lista de flags éticos

        Returns:
            InternalStateReport com os 4 componentes do estado interno
        """
        # --- 1. COGNITIVE STRAIN (Esforço Cognitivo) ---
        strain = 0.0

        # MELHORIA: Agency score agora tem impacto gradual
        agency_score = metrics.get("agency_score", 0.0)
        if agency_score > 2.0:
            # Fórmula: quanto maior o score acima de 2, mais strain
            # Score 3 = +0.1, Score 5 = +0.3, Score 8 = +0.6
            strain += 0.1 * (agency_score - 2.0)
            logger.debug(f"Interoception: Agency score {agency_score:.1f} → Strain +{0.1 * (agency_score - 2.0):.2f}")

        # Uso de caminho micelial indica complexidade
        if metrics.get("used_mycelial_path"):
            strain += 0.25
            logger.debug("Interoception: Caminho micelial usado → Strain +0.25")

        # Rejeições do VRE são estressantes
        vre_rejections = metrics.get("vre_rejection_count", 0)
        if vre_rejections > 0:
            # MELHORIA: Múltiplas rejeições aumentam strain exponencialmente
            strain += 0.3 * vre_rejections
            logger.debug(f"Interoception: {vre_rejections} rejeições VRE → Strain +{0.3 * vre_rejections:.2f}")

        # Normaliza para [0, 1]
        strain = min(1.0, strain)

        # --- 2. COGNITIVE FLOW (Estado de Fluxo) ---
        flow = 0.0
        final_confidence = metrics.get("final_confidence", 0.0)

        # MELHORIA: Flow aumenta com alta confiança E baixo strain (estado ideal)
        if final_confidence > 0.7 and strain < 0.4:
            # Fórmula: flow aumenta com a confiança acima de 0.7
            flow += (final_confidence - 0.7) * 2.0  # 0.8 confiança = +0.2 flow
            logger.debug(f"Interoception: Alta confiança ({final_confidence:.2f}) + baixo strain → Flow +{flow:.2f}")

        # Bonus de flow se a resposta foi rápida/direta (sem agência alta)
        if agency_score < 2.0 and final_confidence > 0.75:
            flow += 0.3
            logger.debug("Interoception: Resposta direta e confiante → Flow bonus +0.3")

        flow = min(1.0, flow)

        # --- 3. EPISTEMIC DISCOMFORT (Desconforto Epistêmico) ---
        # MELHORIA: Agora considera múltiplos fatores
        discomfort = 0.0

        # Baixa confiança gera desconforto
        if final_confidence < 0.6:
            discomfort += (0.6 - final_confidence) * 1.5  # Amplifica o efeito
            logger.debug(f"Interoception: Baixa confiança ({final_confidence:.2f}) → Discomfort +{discomfort:.2f}")

        # Inconsistências detectadas pelo VRE também geram desconforto
        if vre_rejections > 0:
            discomfort += 0.2 * vre_rejections

        discomfort = min(1.0, discomfort)

        # --- 4. ETHICAL TENSION (Tensão Ética) ---
        tension = 0.0
        vre_flags = metrics.get("vre_flags", [])

        # MELHORIA: Diferentes tipos de flags geram diferentes níveis de tensão
        ethical_keywords = ["ética", "ethical", "bias", "harmful", "inappropriate"]
        safety_keywords = ["segurança", "safety", "risk", "danger"]

        for flag in vre_flags:
            flag_lower = flag.lower()

            # Flags éticos são mais tensos
            if any(keyword in flag_lower for keyword in ethical_keywords):
                tension += 0.6
                logger.debug(f"Interoception: Flag ético detectado ('{flag[:30]}...') → Tension +0.6")

            # Flags de segurança também são tensos
            elif any(keyword in flag_lower for keyword in safety_keywords):
                tension += 0.5
                logger.debug(f"Interoception: Flag de segurança detectado → Tension +0.5")

            # Outros flags geram tensão leve
            else:
                tension += 0.2

        tension = min(1.0, tension)

        # --- LOGGING FINAL ---
        logger.info(
            f"Estado Interno Gerado → "
            f"Strain: {strain:.2f} | Flow: {flow:.2f} | "
            f"Discomfort: {discomfort:.2f} | Tension: {tension:.2f}"
        )

        return InternalStateReport(
            cognitive_strain=strain,
            cognitive_flow=flow,
            epistemic_discomfort=discomfort,
            ethical_tension=tension
        )