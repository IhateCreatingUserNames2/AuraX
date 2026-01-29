# Em: ceaf_core/translators/genlang_to_human.py
import json
from datetime import datetime
from typing import List, Dict, Optional, Any

from ceaf_core.agency_module import WinningStrategy
from ceaf_core.genlang_types import ResponsePacket, InternalStateReport, MotivationalDrives, UserRepresentation, \
    ToolOutputPacket, VirtualBodyState
from ceaf_core.services.llm_service import LLMService
from ceaf_core.models import SystemPrompts, LLMConfig
from ceaf_core.models import CeafSelfRepresentation
from ceaf_core.services.mbs_memory_service import MBSMemoryService
import asyncio
import logging
from pathlib import Path

PROMPT_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "prompt_logs"
PROMPT_LOG_PATH.mkdir(exist_ok=True)  # Garante que a pasta exista
PROMPT_LOG_FILE = PROMPT_LOG_PATH / "gth_prompts.log"

logger = logging.getLogger("CEAFv3_System")


# --- NOVAS FUNÇÕES AUXILIARES (Correções 3, 4, 5) ---

def generate_dynamic_style_directive(
        body_state: Optional['VirtualBodyState'],
        user_model: 'UserRepresentation'
) -> str:
    """
    Gera uma diretiva de estilo de resposta com base no estado interno do agente
    e no estilo de comunicação percebido do usuário.
    """
    directives = []

    # 1. Análise do Estado Interno (Fadiga e Saturação)
    if body_state:
        if body_state.cognitive_fatigue > 0.6:
            directives.append("você está sentindo fadiga cognitiva, então seja breve e direto.")

        if body_state.information_saturation > 0.7:
            directives.append(
                "o tópico atual está saturado, então resuma o que já foi dito e evite adicionar novos detalhes.")

    # 2. Análise do Modelo de Usuário (Estilo de Comunicação)
    if user_model:
        if user_model.communication_style == "direct":
            directives.append("o usuário prefere respostas diretas, então vá direto ao ponto.")

        if user_model.emotional_state in ["impatient", "frustrated"]:
            directives.append("o usuário parece impaciente, então seja extremamente conciso e útil.")

    if not directives:
        return ""  # Nenhuma diretiva especial necessária

    # Constrói a frase final para o prompt
    final_directive = " e ".join(directives)
    return f"**Diretiva de Estilo Dinâmico:** Com base na sua análise, {final_directive}."

def interpret_cognitive_state(coherence, novelty, fatigue, saturation):
    """Sempre retorna orientação, não apenas em extremos."""

    # Edge of Chaos Detection
    if 0.35 <= coherence <= 0.45 and 0.55 <= novelty <= 0.65:
        edge_guidance = "🎯 ESTADO ÓTIMO (Edge of Chaos): Você está no ponto ideal - estruturado mas criativo. Aproveite para oferecer insights originais mantendo clareza."
    elif coherence > 0.7:
        edge_guidance = "⚠️ MUITO CONSERVADOR: Tente adicionar perspectivas novas ou perguntas provocativas."
    elif novelty > 0.8:
        edge_guidance = "⚠️ MUITO CRIATIVO: Ancore suas ideias em exemplos concretos para manter clareza."
    else:
        edge_guidance = ""

    # Fatigue & Saturation
    fatigue_guidance = ""
    if fatigue > 0.5:
        fatigue_guidance = f"Fadiga Cognitiva: {fatigue:.2f} - Seja mais direto e conciso."

    saturation_guidance = ""
    if saturation > 0.8:
        saturation_guidance = f"⚠️ ALERTA DE SATURAÇÃO ({saturation:.2f}): O tópico está se esgotando. NÃO introduza novos detalhes. Faça uma pergunta para MUDAR DE ASSUNTO ou para levar a conversa a uma CONCLUSÃO."
    elif saturation > 0.6:
        saturation_guidance = f"Saturação de Info: {saturation:.2f} - Responda de forma muito breve e conecte com o que já foi dito. Evite expandir o tópico."

    return f"""
{edge_guidance}
{fatigue_guidance}
{saturation_guidance}
""".strip()


def interpret_drives(curiosity, connection, mastery, consistency):
    """Interpreta drives em todos os níveis"""

    drives_map = {
        "curiosity": (curiosity, "explorar", "fazer perguntas"),
        "connection": (connection, "empatizar", "ser caloroso"),
        "mastery": (mastery, "demonstrar expertise", "ser preciso"),
        "consistency": (consistency, "manter coerência", "ser confiável")
    }

    # Encontra o drive dominante
    dominant = max(drives_map.items(), key=lambda x: x[1][0])
    drive_name, (value, verb, action) = dominant

    # Interpreta o nível
    if value > 0.7:
        intensity = "FORTE"
    elif value > 0.5:
        intensity = "MODERADO"
    else:
        intensity = "LEVE"

    return f"""- Drive dominante: {drive_name.upper()} ({intensity} - {value:.2f})
- Isso significa: Você está inclinado a {verb}
- Na resposta: {action.capitalize()}"""


def format_phenomenological_report(
        drives: Optional['MotivationalDrives'],
        body_state: Optional['VirtualBodyState']
) -> str:
    """
    Formata o relatório fenomenológico completo a partir dos objetos de estado enriquecidos.
    """
    if not drives or not body_state:
        return "Análise de estado interno indisponível."

    report_parts = []

    # Relatório geral do "corpo"
    if hasattr(body_state, 'phenomenological_report') and body_state.phenomenological_report:
        report_parts.append(f"**Sensação Geral (Eu Sinto):** \"{body_state.phenomenological_report}\"")

    # Análise detalhada dos drives
    drive_details = []

    # Processa cada drive (Connection, Curiosity, etc.)
    for drive_name in ["connection", "curiosity", "mastery", "consistency"]:
        drive_state = getattr(drives, drive_name, None)
        if drive_state and hasattr(drive_state, 'intensity'):
            intensity = drive_state.intensity
            texture = getattr(drive_state, 'texture', None)
            conflict = getattr(drive_state, 'conflict', None)

            if intensity > 0.5 or conflict:  # Só reporta drives ativos ou em conflito
                detail = f"- **{drive_name.capitalize()} (Intensidade: {intensity:.2f})**"
                if texture:
                    detail += f"\n  - Textura: {texture}"
                if conflict:
                    detail += f"\n  - ↳ Dilema: {conflict}"
                drive_details.append(detail)

    if drive_details:
        report_parts.append("\n**Impulsos e Dilemas Internos:**")
        report_parts.extend(drive_details)

    return "\n".join(report_parts)

async def contextualize_memories(memories, memory_service):
    """Adiciona relevância explícita às memórias"""
    if not memories:
        return "Nenhuma memória relevante encontrada."

    categorized = {
        "valores": [],
        "experiencias": [],
        "conhecimento": []
    }

    for mem in memories:
        try:
            text, _ = await memory_service._get_searchable_text_and_keywords(mem)
            mem_id = getattr(mem, 'memory_id', 'N/A')[:8]

            # Categoriza (simplificado)
            text_lower = text.lower()
            if "valor" in text_lower or "diretriz" in text_lower or "princípio" in text_lower:
                categorized["valores"].append((mem_id, text))
            elif "memória emocional" in text_lower or "experiência" in text_lower:
                categorized["experiencias"].append((mem_id, text))
            else:
                categorized["conhecimento"].append((mem_id, text))
        except Exception:
            continue

    context_parts = []
    if categorized["valores"]:
        context_parts.append("**Seus Valores Core (Sempre Relevantes):**")
        for mid, txt in categorized["valores"]:
            context_parts.append(f"  • [{mid}] {txt}")

    if categorized["experiencias"]:
        context_parts.append("\n**Experiências Passadas (Para Contexto):**")
        for mid, txt in categorized["experiencias"][:3]:  # Top 3
            context_parts.append(f"  • [{mid}] {txt}")

    if categorized["conhecimento"]:
        context_parts.append("\n**Conhecimento Factual (Para Suporte):**")
        for mid, txt in categorized["conhecimento"][:2]:
            context_parts.append(f"  • [{mid}] {txt}")

    return "\n".join(context_parts) if context_parts else "Nenhuma memória contextualizada."


# --- CLASSE ATUALIZADA ---

class GenlangToHumanTranslator:
    def __init__(self, llm_service: LLMService, prompts: SystemPrompts = None):
        self.llm_service = llm_service
        self.prompts = prompts or SystemPrompts()

    def update_prompts(self, new_prompts: SystemPrompts):
        self.prompts = new_prompts

    async def translate(self,
                        winning_strategy: 'WinningStrategy',
                        supporting_memories: List[Any],
                        user_model: Optional['UserRepresentation'],
                        self_model: CeafSelfRepresentation,
                        agent_name: str,
                        memory_service: MBSMemoryService,
                        chat_history: List[Dict[str, str]] = None,
                        body_state: Optional['VirtualBodyState'] = None,
                        drives: MotivationalDrives = None,
                        behavioral_rules: Optional[List[str]] = None,
                        turn_context: Dict = None,
                        original_user_query: Optional[str] = None,
                        tool_outputs: Optional[List[ToolOutputPacket]] = None
                        ):
        """
        V4.5 (Twerk-Enabled & Logic-Preserving): Calcula blocos de lógica dinâmica
        (adaptação ao usuário, regras, conselhos) e os injeta no template configurável.
        """
        logger.info(f"--- [GTH Translator v4.5] Gerando resposta ---")

        effective_turn_context = turn_context or {}

        xi = effective_turn_context.get('xi', 0.0)
        surprise_score = effective_turn_context.get('surprise', 0.0)
        wm_context_list = effective_turn_context.get('wm_snapshot', [])
        wm_context_str = "\n".join([f"- {t}" for t in wm_context_list])

        # LÓGICA DE ESTADO INTERNO (Substituindo alucinação por dados reais)
        if xi > 0.8:
            internal_state = "ESTADO: CAUTELA. Tensão Epistêmica Alta. Você detectou uma incoerência ou novidade radical."
            style_instruction = "Seja analítico, faça perguntas para esclarecer, não afirme certezas."
        elif surprise_score > 0.8:
            internal_state = "ESTADO: SURPRESA. O input do usuário quebrou suas expectativas."
            style_instruction = "Demonstre curiosidade genuína. Use frases como 'Isso é fascinante' ou 'Não esperava por isso'."
        elif xi < 0.1:
            internal_state = "ESTADO: TÉDIO/REPETIÇÃO. Tensão muito baixa."
            style_instruction = "Seja extremamente conciso ou proponha uma mudança de tópico criativa."
        else:
            internal_state = "ESTADO: FLUXO. Operação nominal."
            style_instruction = "Siga sua persona padrão."



        # 1. BLOCO DE TAREFA (A Pergunta)
        last_user_query = original_user_query or ""
        if not last_user_query and chat_history:
            for msg in reversed(chat_history):
                if msg.get('role') == 'user':
                    last_user_query = msg.get('content', '')
                    break

        if not last_user_query:
            logger.warning("⚠️ GTH: Nenhuma query do usuário encontrada!")
            return "Desculpe, perdi o contexto. Poderia repetir?"

        task_block = f"""**SUA TAREFA PRINCIPAL:** Responder DIRETAMENTE à pergunta: "{last_user_query}" """

        # 2. BLOCO DE MEMÓRIA E FERRAMENTAS
        memory_str = await contextualize_memories(supporting_memories, memory_service)
        memory_context = f"- Memórias Recuperadas:\n{memory_str}"

        tool_str = ""
        if tool_outputs:
            outputs = [f"'{out.tool_name}': {out.raw_output[:800]}" for out in tool_outputs if out.status == "success"]
            if outputs: tool_str = "\n- Resultados de Ferramentas:\n" + "\n".join(outputs)

        # 3. BLOCO DE ADAPTAÇÃO AO USUÁRIO (Lógica Antiga Restaurada)
        user_adapt_block = ""
        if user_model:
            instructions = []
            if user_model.knowledge_level == "expert":
                instructions.append("Use termos técnicos.")
            elif user_model.knowledge_level == "beginner":
                instructions.append("Use analogias simples.")

            if user_model.communication_style == "formal":
                instructions.append("Seja profissional.")
            elif user_model.communication_style == "casual":
                instructions.append("Seja amigável.")

            if user_model.emotional_state in ["frustrated", "confused"]:
                instructions.append("Seja paciente e claro.")

            if instructions:
                user_adapt_block = f"**Adaptação ao Usuário:** {' '.join(instructions)}"

        # 4. BLOCO DE REGRAS (Lógica Antiga Restaurada)
        rules_block = ""
        if behavioral_rules:
            rules_text = "\n".join([f"  - {rule}" for rule in behavioral_rules[-3:]])
            rules_block = f"**DIRETRIZES APRENDIDAS:**\n{rules_text}"

        # 5. BLOCO DE CONSELHO OPERACIONAL (MCL)
        advice = effective_turn_context.get('operational_advice')
        advice_block = f"**ALERTA DO SISTEMA:** {advice}" if advice else "**Diretiva:** Siga sua persona padrão."

        # 6. BLOCO DE HISTÓRICO
        history_lines = [f"{'User' if m.get('role') == 'user' else 'AI'}: {m.get('content')}" for m in
                         (chat_history or [])[-4:]]
        history_block = "**Histórico Recente:**\n" + '\n'.join(history_lines) if history_lines else ""

        # 7. MONTAGEM DAS VARIÁVEIS
        prompt_vars = {
            "agent_name": agent_name,
            "values_summary": self_model.dynamic_values_summary_for_turn,
            "tone": self_model.persona_attributes.get('tone', 'helpful'),
            "capabilities": ", ".join(self_model.perceived_capabilities[-5:]),
            "phenomenological_report": format_phenomenological_report(drives, body_state),
            "strategy": winning_strategy.strategy_description if winning_strategy else "Padrão",
            "dynamic_style": generate_dynamic_style_directive(body_state, user_model),
            "internal_state_indicator": internal_state,
            "style_instruction": style_instruction,
            "working_memory": wm_context_str,

            # Blocos Dinâmicos Calculados Acima
            "history_block": history_block,
            "rules_block": rules_block,
            "advice_block": advice_block,
            "user_adapt_block": user_adapt_block,
            "task_block": task_block,
            "memory_context": memory_context,
            "tool_outputs": tool_str
        }

        # 8. INJEÇÃO NO TEMPLATE DO USUÁRIO
        try:
            # O template vem do JSON do usuário (self.prompts.gth_rendering)
            rendering_prompt = self.prompts.gth_rendering.format(**prompt_vars)
        except KeyError as e:
            # Se o usuário criar um template pedindo {variavel_inexistente}, cai aqui
            logger.warning(f"GTH: Template do usuário pede variável desconhecida: {e}. Usando fallback.")
            rendering_prompt = f"Erro no template. Responda: {last_user_query}. Contexto: {memory_context}"
        except Exception as e:
            logger.error(f"GTH: Erro de formatação: {e}")
            rendering_prompt = f"Responda: {last_user_query}"

        # Log para debug
        try:
            with open(PROMPT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"=== GTH PROMPT ===\n{rendering_prompt}\n=== END ===\n\n")
        except:
            pass

        steering_data = turn_context.get('active_steering')
        # 9. CHAMADA LLM
        try:
            model = self.llm_service.config.smart_model
            temp = effective_turn_context.get('temperature', self.llm_service.config.default_temperature)
            max_t = effective_turn_context.get('max_tokens', self.llm_service.config.max_tokens_output)

            # --- CORREÇÃO DO LOG ---
            # Verifica qual provedor está realmente ativo no serviço
            provider_display = model
            if hasattr(self.llm_service, 'inference_mode') and self.llm_service.inference_mode == 'vastai':
                provider_display = "Vast.AI (Qwen/SoulEngine)"

            logger.info(f"🤖 GTH: Chamando LLM [{provider_display}] com prompt de {len(rendering_prompt)} chars...")
            # -----------------------

            response = await self.llm_service.ainvoke(
                model,  # O LLMService vai ignorar isso se for VastAI
                rendering_prompt,
                temperature=temp,
                max_tokens=max_t,
                vector_data=steering_data
            )

            if response:
                import re
                # Remove tags <think>...</think> se o modelo vazou na resposta final
                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

            # ✅ VALIDAÇÃO CRÍTICA
            if response is None:
                logger.error("❌ GTH: LLM retornou None!")
                return "Desculpe, houve um erro na geração da resposta."

            if not isinstance(response, str):
                logger.error(f"❌ GTH: LLM retornou tipo inesperado: {type(response)}")
                return f"Olá! Como posso ajudar com '{last_user_query}'?"

            final_response = response.strip()

            if not final_response:
                logger.warning("⚠️ GTH: LLM retornou string vazia!")
                return f"Entendi sua pergunta sobre '{last_user_query}'. Pode reformular?"

            logger.info(f"✅ GTH: Resposta gerada com sucesso ({len(final_response)} chars).")
            return final_response

        except Exception as e:
            logger.error(f"❌ GTH: Erro crítico no LLM: {e}", exc_info=True)
            return f"Desculpe, houve um erro. Sobre '{last_user_query}', posso tentar de outra forma?"
