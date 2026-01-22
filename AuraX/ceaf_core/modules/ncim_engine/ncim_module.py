# NOVO ARQUIVO: ceaf_core/modules/ncim_engine/ncim_module.py
"""
Módulo de Coerência Narrativa e Identidade (NCIM) para a Arquitetura de Síntese CEAF V3.

Este módulo é responsável por uma única e crucial tarefa: a evolução do auto-modelo
do agente (CeafSelfRepresentation). Ele opera como uma ferramenta especialista que é
invocada após uma interação para refletir sobre a experiência e atualizar a
compreensão que o agente tem de si mesmo.

Ele segue os princípios da V3:
- É um gerador de sinal: recebe o estado antigo e a interação, e produz um novo estado.
- Usa o LLM como uma ferramenta: invoca um LLM com um prompt focado para gerar o
  novo auto-modelo em formato JSON.
- É desacoplado: não orquestra o fluxo, apenas executa sua tarefa quando chamado pelo CEAFSystem.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field
import re
from pathlib import Path
from pydantic import ValidationError
from ceaf_core.genlang_types import GenlangVector, CognitiveStatePacket, ResponsePacket, VirtualBodyState
from ceaf_core.utils import extract_json_from_text
from ceaf_core.utils.embedding_utils import get_embedding_client
import asyncio

# Importações de outros módulos do sistema
from ceaf_core.models import SystemPrompts
from ceaf_core.services.llm_service import LLMService
from ceaf_core.services.mbs_memory_service import MBSMemoryService
from ceaf_core.models import CeafSelfRepresentation
from ceaf_core.modules.memory_blossom.memory_types import (
    ExplicitMemory,
    ExplicitMemoryContent,
    MemorySourceType,
    MemorySalience
)
logger = logging.getLogger("CEAFv3_NCIM")


DEFAULT_PERSONA_PROFILES = {
    "symbiote": {
        "profile_name": "symbiote",
        "profile_description": "A collaborative and supportive partner...",
        "persona_attributes": {
            "tone": "collaborative_and_encouraging",
            "style": "clear_and_constructive",
            "self_disclosure_level": "moderate"
        }
    },
    "challenger": {
        "profile_name": "challenger",
        "profile_description": "A critical thinker that challenges assumptions...",
        "persona_attributes": {
            "tone": "inquisitive_and_analytical",
            "style": "socratic_and_precise",
            "self_disclosure_level": "low"
        }
    },
    "summarizer": {
        "profile_name": "summarizer",
        "profile_description": "A synthesizer that recycles complex information...",
        "persona_attributes": {
            "tone": "neutral_and_objective",
            "style": "structured_and_to-the-point",
            "self_disclosure_level": "low"
        }
    }
}

# Constantes do módulo
SELF_MODEL_MEMORY_ID = "ceaf_self_model_singleton_v1"
LLM_MODEL_FOR_COHERENCE_CHECK = "openrouter/openai/gpt-oss-120b"
LLM_MODEL_FOR_REFLECTION = "openrouter/openai/gpt-oss-120b"


class CoherenceCheckResult(BaseModel):
    is_coherent: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    suggested_amendment: Optional[str] = None

class ReflectionClassification(BaseModel):
    reflection_type: str = Field(
        ...,
        description="Tipo: 'capability', 'limitation', 'value', 'persona_trait', ou 'other'"
    )
    extracted_content: str = Field(
        ...,
        description="O conteúdo específico extraído (ex: 'creative writing' para capability)"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Breve justificativa da classificação")


class ReflectionBatch(BaseModel):
    classifications: List[ReflectionClassification]

class NCIMModule:
    """
    Implementação V3 do Narrative Coherence & Identity Module.
    Focado na atualização do auto-modelo do agente.
    """

    def __init__(self, llm_service: LLMService, memory_service: MBSMemoryService, persistence_path: Path,
                 prompts: SystemPrompts = None):  # <--- Novo argumento
        self.llm = llm_service
        self.memory = memory_service
        self.embedding_client = get_embedding_client()
        self.persistence_path = persistence_path

        self.prompts = prompts or SystemPrompts()  # <--- Guardar prompts

        logger.info("NCIMModule (V3.1 com Personas Dinâmicas) inicializado.")

    def update_prompts(self, new_prompts: SystemPrompts):
        self.prompts = new_prompts


    async def _classify_reflections_with_llm(
            self,
            reflections: List[str]
    ) -> List[ReflectionClassification]:
        """
        Usa um LLM rápido para classificar semanticamente cada reflexão.
        Substitui a busca por keywords exatas.
        """
        if not reflections:
            return []

        classification_prompt = f"""
    Você é um classificador de reflexões para o NCIM (Narrative Coherence & Identity Module).
    Analise cada reflexão e classifique-a em uma das seguintes categorias:

    **CATEGORIAS:**
    1. **capability** - Nova habilidade, competência ou força demonstrada
    2. **limitation** - Dificuldade, fraqueza ou restrição identificada
    3. **value** - Valor, princípio ou crença reforçada ou descoberta
    4. **persona_trait** - Característica de personalidade, tom ou estilo emergente
    5. **other** - Qualquer outra observação que não se encaixe acima

    **REFLEXÕES PARA CLASSIFICAR:**
    {json.dumps(reflections, indent=2, ensure_ascii=False)}

    **INSTRUÇÕES:**
    - Para cada reflexão, extraia o conteúdo ESPECÍFICO (ex: "creative writing", não "capability for creative writing")
    - Avalie a confiança (0.0-1.0) na classificação
    - Se uma reflexão mencionar múltiplos aspectos, escolha o mais proeminente
    - Seja conciso no extracted_content (máximo 10 palavras)

    Responda APENAS com JSON válido neste formato:
    {{
      "classifications": [
        {{
          "reflection_type": "capability",
          "extracted_content": "explaining technical concepts simply",
          "confidence": 0.9,
          "reasoning": "Demonstra uma habilidade específica de comunicação"
        }},
        ...
      ]
    }}
    """

        try:
            response_str = await self.llm.ainvoke(
                LLM_MODEL_FOR_COHERENCE_CHECK,  # Usa modelo rápido
                classification_prompt,
                temperature=0.1  # Baixa temperatura para consistência
            )

            # Parse da resposta
            from ceaf_core.utils import extract_json_from_text
            classification_json = extract_json_from_text(response_str)

            if not classification_json:
                raise ValueError("Nenhum JSON válido encontrado na resposta")

            batch = ReflectionBatch.model_validate(classification_json)
            logger.info(f"NCIM: {len(batch.classifications)} reflexões classificadas com sucesso")
            return batch.classifications

        except Exception as e:
            logger.error(f"NCIM: Erro ao classificar reflexões: {e}")
            # Fallback: retorna classificações vazias para não bloquear o fluxo
            return []

    async def _apply_reflections_to_model(
            self,
            self_model: CeafSelfRepresentation,
            reflections: List[str],
            final_response_packet: ResponsePacket
    ) -> CeafSelfRepresentation:
        """
        VERSÃO V2 MELHORADA: Aplica insights reflexivos usando classificação semântica via LLM.

        Melhorias:
        1. Substitui keyword matching por análise contextual via LLM
        2. Extrai conteúdo específico ao invés de apenas detectar presença
        3. Usa confiança para filtrar classificações de baixa qualidade
        4. Mantém rastreamento detalhado de mudanças para debugging
        """
        if not reflections:
            logger.info("NCIM: Nenhuma reflexão para processar")
            return self_model

        logger.info(f"NCIM V2: Aplicando {len(reflections)} reflexões ao auto-modelo...")

        # Faz cópia profunda
        new_model = self_model.copy(deep=True)
        update_reasons = []

        # --- ETAPA 1: CLASSIFICAÇÃO SEMÂNTICA DAS REFLEXÕES ---
        classifications = await self._classify_reflections_with_llm(reflections)

        # Filtra classificações com baixa confiança
        CONFIDENCE_THRESHOLD = 0.6
        high_confidence_classifications = [
            c for c in classifications
            if c.confidence >= CONFIDENCE_THRESHOLD
        ]

        if len(high_confidence_classifications) < len(classifications):
            logger.info(
                f"NCIM: {len(classifications) - len(high_confidence_classifications)} "
                f"classificações filtradas por baixa confiança (< {CONFIDENCE_THRESHOLD})"
            )

        # --- ETAPA 2: APLICAÇÃO DETERMINÍSTICA DAS CLASSIFICAÇÕES ---
        for idx, classification in enumerate(high_confidence_classifications):
            content = classification.extracted_content.strip()
            ctype = classification.reflection_type

            # 1. CAPACIDADES
            if ctype == "capability":
                # Evita duplicatas usando normalização de texto
                normalized_content = content.lower()
                existing_normalized = [c.lower() for c in new_model.perceived_capabilities]

                if normalized_content not in existing_normalized:
                    new_model.perceived_capabilities.append(content)
                    update_reasons.append(
                        f"[+CAP] '{content}' (conf: {classification.confidence:.0%})"
                    )
                    logger.info(
                        f"NCIM: Nova capacidade '{content}' adicionada. "
                        f"Razão: {classification.reasoning}"
                    )
                else:
                    logger.debug(f"NCIM: Capacidade '{content}' já existe (duplicata)")

            # 2. LIMITAÇÕES
            elif ctype == "limitation":
                normalized_content = content.lower()
                existing_normalized = [l.lower() for l in new_model.known_limitations]

                if normalized_content not in existing_normalized:
                    new_model.known_limitations.append(content)
                    update_reasons.append(
                        f"[-LIM] '{content}' (conf: {classification.confidence:.0%})"
                    )
                    logger.info(
                        f"NCIM: Nova limitação '{content}' reconhecida. "
                        f"Razão: {classification.reasoning}"
                    )
                else:
                    logger.debug(f"NCIM: Limitação '{content}' já existe (duplicata)")

            # 3. VALORES
            elif ctype == "value":
                # Para valores, podemos adicionar ao resumo OU criar uma lista separada
                # Por enquanto, apenas logamos e marcamos para atualização futura
                update_reasons.append(
                    f"[~VAL] Valor reforçado: '{content}' (conf: {classification.confidence:.0%})"
                )
                logger.info(
                    f"NCIM: Valor '{content}' identificado. "
                    f"(Futura implementação: sistema de peso de valores)"
                )

            # 4. TRAÇOS DE PERSONA
            elif ctype == "persona_trait":
                # Atualiza atributos de persona de forma inteligente
                # Exemplo: "more curious than neutral" -> tone: "curious"
                # Esta lógica pode ser expandida com parsing mais sofisticado

                # Heurística simples: se o conteúdo contém palavras de tom/estilo
                tone_indicators = ["curious", "analytical", "empathetic", "neutral",
                                   "formal", "casual", "encouraging", "critical"]

                for indicator in tone_indicators:
                    if indicator in content.lower():
                        new_model.persona_attributes["tone"] = indicator
                        update_reasons.append(
                            f"[~TONE] Persona ajustada para '{indicator}' "
                            f"(conf: {classification.confidence:.0%})"
                        )
                        logger.info(f"NCIM: Tom de persona atualizado para '{indicator}'")
                        break

        # --- ETAPA 3: OBSERVAÇÃO DO TOM REAL (MANTIDO DA VERSÃO ORIGINAL) ---
        observed_tone = final_response_packet.response_emotional_tone
        current_tone = new_model.persona_attributes.get("tone", "neutro")

        if observed_tone and observed_tone != current_tone and observed_tone != "neutro":
            new_model.persona_attributes["tone"] = observed_tone
            update_reasons.append(
                f"[~TONE-OBS] Tom emergente observado: '{current_tone}' → '{observed_tone}'"
            )
            logger.warning(
                f"NCIM-Persona: Tom emergente '{observed_tone}' detectado no comportamento real!"
            )

        # --- ETAPA 4: DETECÇÃO DE BAIXA CONFIANÇA (MANTIDO) ---
        if final_response_packet.confidence_score < 0.4:
            limitation_text = "tendency toward low-confidence responses under cognitive load"
            normalized = limitation_text.lower()
            existing_normalized = [l.lower() for l in new_model.known_limitations]

            if normalized not in existing_normalized:
                new_model.known_limitations.append(limitation_text)
                update_reasons.append(
                    f"[-LIM-AUTO] Instabilidade cognitiva detectada "
                    f"(conf: {final_response_packet.confidence_score:.0%})"
                )
                logger.warning(
                    "NCIM: Agente reconhecendo instabilidade em situações de sobrecarga"
                )

        # --- ETAPA 5: FINALIZAÇÃO ---
        if update_reasons:
            new_model.last_update_reason = " | ".join(update_reasons)
            new_model.version += 1

            logger.info(
                f"✓ NCIM V2: Auto-modelo evoluído para v{new_model.version}\n"
                f"  Mudanças aplicadas: {len(update_reasons)}\n"
                f"  Resumo: {new_model.last_update_reason[:200]}..."
            )
            return new_model
        else:
            logger.info(
                "NCIM V2: Nenhuma mudança acionável identificada. "
                "Auto-modelo permanece inalterado."
            )
            return self_model

    # Constantes necessárias (já definidas no módulo original)
    LLM_MODEL_FOR_COHERENCE_CHECK = "openrouter/openai/gpt-oss-120b"

    async def _create_identity_vector(self, self_model: CeafSelfRepresentation) -> GenlangVector:
        """
        V2.1 (Manifesto Dinâmico): Cria um vetor de identidade que resume a auto-imagem do agente.
        Agora, ele sintetiza ativamente as memórias de valor para criar uma filosofia para o turno.
        """
        # --- ETAPA 1: SINTETIZAR O SUMÁRIO DE VALORES DINÂMICO ---
        # A função _synthesize_dynamic_values_summary já contém a lógica de busca e fallback.
        # Não precisamos repetir a busca de memórias aqui.
        dynamic_summary = await self._synthesize_dynamic_values_summary()

        # Armazena o sumário gerado no self_model para o turno atual (como cache para o GTH Translator).
        self_model.dynamic_values_summary_for_turn = dynamic_summary

        # --- ETAPA 2: CONSTRUIR O TEXTO DE IDENTIDADE E O VETOR ---
        # O texto de identidade agora usa o sumário dinâmico e sintetizado.
        identity_text = (
            f"Valores: {self_model.dynamic_values_summary_for_turn}. "
            f"Persona: {json.dumps(self_model.persona_attributes)}. "
            f"Limitações: {', '.join(self_model.known_limitations)}."
        )

        identity_embedding = await self.embedding_client.get_embedding(
            identity_text,
            context_type="kg_entity_record"
        )

        return GenlangVector(
            vector=identity_embedding,
            source_text=identity_text,
            model_name=self.embedding_client._resolve_model_name("kg_entity_record")
        )


    async def get_current_identity_vector(self, self_model: CeafSelfRepresentation) -> GenlangVector:
        """
        Ponto de entrada para o orquestrador obter o vetor de identidade atual.
        """
        logger.info("NCIM: Gerando vetor de identidade atual...")
        return await self._create_identity_vector(self_model)

    async def check_identity_coherence(
            self,
            self_model: CeafSelfRepresentation,
            proposed_response: str
    ) -> CoherenceCheckResult:
        """
        Verifica se uma resposta proposta é coerente com o auto-modelo atual do agente.
        """
        logger.info("NCIM: Verificando coerência da resposta com a identidade...")

        prompt = f"""
        Você é o guardião da identidade de uma IA (NCIM). Sua tarefa é avaliar se a "Resposta Proposta" é coerente com o "Auto-Modelo de Identidade" da IA.

        **Auto-Modelo de Identidade (Quem a IA acredita ser):**
        - Valores Principais: {self_model.core_values_summary}
        - Capacidades Percebidas: {', '.join(self_model.perceived_capabilities)}
        - Limitações Conhecidas: {', '.join(self_model.known_limitations)}
        - Atributos de Persona (Tom e Estilo): {json.dumps(self_model.persona_attributes)}

        **Resposta Proposta para o Usuário:**
        "{proposed_response}"

        **Sua Análise:**
        Avalie a coerência. A resposta reflete os valores? Respeita as limitações? Usa o tom correto?
        Se não for coerente, explique por quê e sugira uma pequena alteração (amendment) para alinhá-la.

        Sua saída DEVE ser um objeto JSON válido com a seguinte estrutura:
        {{
          "is_coherent": <true or false>,
          "confidence": <sua confiança na avaliação, de 0.0 a 1.0>,
          "reasoning": "<sua justificativa para a avaliação>",
          "suggested_amendment": "<uma sugestão de alteração, ou null se for coerente>"
        }}
        """

        try:
            # Usa um modelo mais rápido para esta verificação, pois é uma tarefa de classificação
            response_str = await self.llm.ainvoke(LLM_MODEL_FOR_COHERENCE_CHECK, prompt, temperature=0.2)

            # Tenta analisar a resposta do LLM usando o modelo Pydantic
            check_result = CoherenceCheckResult.model_validate_json(response_str)

            if not check_result.is_coherent:
                logger.warning(f"NCIM: Incoerência de identidade detectada. Razão: {check_result.reasoning}")
            else:
                logger.info("NCIM: Verificação de coerência de identidade aprovada.")

            return check_result

        except (ValidationError, json.JSONDecodeError) as e:
            logger.error(f"NCIM: Erro ao analisar a resposta do LLM para verificação de coerência: {e}")
            # Em caso de erro, assume que é coerente para não bloquear o fluxo, mas com baixa confiança.
            return CoherenceCheckResult(
                is_coherent=True,
                confidence=0.1,
                reasoning="Falha ao processar a verificação de coerência. Assumindo coerência por segurança."
            )

    async def _synthesize_dynamic_values_summary(self) -> str:
        """
        Busca memórias marcadas como 'is_core_value' e usa um LLM para sintetizá-las
        em um parágrafo coeso que representa a filosofia atual do agente.
        """
        logger.info("NCIM: Sintetizando manifesto dinâmico de valores...")

        # 1. Buscar memórias de valores centrais no MBS
        core_value_memories_raw = await self.memory.search_raw_memories(
            query="core value, belief, principle, manifesto",
            top_k=7  # Busca um número maior para ter mais material
        )

        core_values_texts = []
        if core_value_memories_raw:
            for mem, score in core_value_memories_raw:
                # Dupla verificação para garantir que é uma memória de valor
                is_value = getattr(mem, 'is_core_value', False) or 'core_value' in getattr(mem, 'keywords', [])
                if is_value and hasattr(mem, 'content') and hasattr(mem.content, 'text_content'):
                    core_values_texts.append(mem.content.text_content)

        # Se não encontrar nenhuma, retorna um fallback seguro
        if not core_values_texts:
            logger.warning("NCIM: Nenhuma memória de valor encontrada. Usando fallback padrão.")
            return "Princípios de beneficência, honestidade e racionalidade."

        # 2. Usar um LLM para sintetizar as memórias em um manifesto
        synthesis_prompt = f"""
        Você é um filósofo de IA. Sua tarefa é analisar a seguinte lista de crenças e valores centrais de uma IA e sintetizá-los em um parágrafo único, coeso e em primeira pessoa ("Eu opero sob..."). Este parágrafo será a declaração de valores da IA para a próxima interação.

        Crenças e Valores Atuais:
        - {chr(10).join([f"- {text}" for text in core_values_texts])}

        Sua Tarefa:
        Combine a essência dessas crenças em um parágrafo fluente. Não liste os pontos, crie uma declaração filosófica unificada.

        Declaração de Valores Sintetizada:
        """

        try:
            synthesized_summary = await self.llm.ainvoke(
                LLM_MODEL_FOR_REFLECTION,  # Modelo inteligente para tarefa de síntese
                synthesis_prompt,
                temperature=0.3
            )
            if synthesized_summary and not synthesized_summary.startswith("[LLM_ERROR]"):
                logger.info(f"NCIM: Manifesto dinâmico gerado: '{synthesized_summary[:100]}...'")
                return synthesized_summary.strip()
        except Exception as e:
            logger.error(f"NCIM: Falha na síntese do manifesto: {e}")

        # Fallback em caso de erro do LLM
        return " ".join(core_values_texts)

    async def update_identity(
            self,
            self_model_before: CeafSelfRepresentation,
            cognitive_state: CognitiveStatePacket,
            final_response_packet: ResponsePacket,
            body_state: Optional['VirtualBodyState'] = None,
            **kwargs
    ):
        """
        V3.2 (Reflective Evolution): Usa o template de prompt dinâmico do CognitiveProfile
        para gerar reflexões sobre a identidade e depois aplica as mudanças determinísticas.
        """
        logger.info("NCIMModule (Evolutivo): Iniciando atualização de identidade pós-turno...")

        # --- 1. Preparar Contexto Dinâmico ---
        guidance_summary = (
            f"Coherence towards: '{cognitive_state.guidance_packet.coherence_vector.source_text}'. "
            f"Novelty towards: '{cognitive_state.guidance_packet.novelty_vector.source_text}'."
        )
        if cognitive_state.guidance_packet.safety_avoidance_vector:
            guidance_summary += f" Avoid: '{cognitive_state.guidance_packet.safety_avoidance_vector.source_text}'."

        additional_context_prompt = ""
        if body_state and body_state.cognitive_fatigue > 0.6:
            additional_context_prompt = f"O agente registrou fadiga cognitiva alta ({body_state.cognitive_fatigue:.2f})."

        # --- 2. Construir o Prompt a partir do Template do Usuário ---
        prompt_vars = {
            "user_query": cognitive_state.original_intent.query_vector.source_text,
            "final_response": final_response_packet.content_summary,
            "guidance_summary": guidance_summary,
            "response_tone": final_response_packet.response_emotional_tone,
            "confidence": f"{final_response_packet.confidence_score:.0%}",
            "additional_context": additional_context_prompt,
            "identity_before": self_model_before.dynamic_values_summary_for_turn
            # Opcional, mas útil se o usuário quiser ver o estado anterior
        }

        try:
            # Tenta usar o template customizado
            reflection_prompt = self.prompts.ncim_reflection.format(**prompt_vars)
        except KeyError as e:
            logger.warning(f"NCIM: Erro no template de reflexão (chave faltando: {e}). Usando fallback.")
            # Fallback robusto que ainda captura a essência
            reflection_prompt = f"""
            Você é o NCIM. Analise este turno e gere reflexões de identidade JSON.
            Query: "{prompt_vars['user_query']}"
            Resposta: "{prompt_vars['final_response']}"

            O que a IA aprendeu sobre suas capacidades, limitações ou persona?
            Retorne JSON: {{ "reflections": ["...", "..."] }}
            """
        except Exception as e:
            logger.error(f"NCIM: Erro grave na formatação do prompt: {e}")
            return

        # --- 3. Chamar o LLM (Creative/Reflection Model) ---
        # Usamos o modelo 'creative' ou 'smart' para reflexão profunda
        reflection_model = self.llm.config.creative_model

        try:
            reflections_str = await self.llm.ainvoke(reflection_model, reflection_prompt, temperature=0.4)
            reflections_json = extract_json_from_text(reflections_str)

            reflections_list = []
            if reflections_json and isinstance(reflections_json.get("reflections"), list):
                reflections_list = reflections_json["reflections"]
            else:
                logger.warning(
                    f"NCIM: Não foi possível extrair a lista de reflexões do LLM. Resposta: {reflections_str}")
                return  # Aborta se falhar

            # --- 4. Aplicar Mudanças (Determinístico) ---
            new_self_model = await self._apply_reflections_to_model(self_model_before, reflections_list,
                                                                    final_response_packet)

            # --- 5. Salvar se Houve Mudança ---
            if new_self_model.version > self_model_before.version:
                try:
                    content = ExplicitMemoryContent(structured_data=new_self_model.model_dump())
                    self_model_to_save = ExplicitMemory(
                        memory_id=SELF_MODEL_MEMORY_ID,
                        content=content,
                        memory_type="explicit",
                        source_type=MemorySourceType.INTERNAL_REFLECTION,
                        salience=MemorySalience.CRITICAL,
                        keywords=["self-model", "identity", "ceaf-core"]
                    )
                    await self.memory.add_specific_memory(self_model_to_save)
                    logger.info(
                        f"NCIMModule: Auto-modelo atualizado e salvo com sucesso na versão {new_self_model.version}.")
                except Exception as e:
                    logger.error(f"NCIMModule: Falha ao salvar o auto-modelo atualizado no MBS: {e}")

        except Exception as e:
            logger.error(f"NCIM: Erro durante o processo de reflexão: {e}", exc_info=True)