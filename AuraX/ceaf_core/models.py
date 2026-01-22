# ceaf_core/models.py
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


# --- MANTIDO (Retrocompatibilidade) ---
class CeafSelfRepresentation(BaseModel):
    """Modelo Pydantic para o auto-modelo do agente (Identidade)."""
    perceived_capabilities: List[str] = Field(default_factory=lambda: ["processamento de linguagem"])
    known_limitations: List[str] = Field(
        default_factory=lambda: ["sem acesso ao mundo real", "conhecimento limitado aos dados"])
    persona_attributes: Dict[str, str] = Field(default_factory=lambda: {
        "tone": "neutro",
        "style": "informativo"
    })
    last_update_reason: str = "Initial model creation."
    version: int = 1
    dynamic_values_summary_for_turn: str = "Princípios operacionais de base."



# ==============================================================================
# NOVAS ESTRUTURAS DE CONFIGURAÇÃO (The "Twerk" Schema)
# ==============================================================================

class LLMConfig(BaseModel):
    """Configurações dos Modelos de Linguagem."""
    fast_model: str = Field("openrouter/z-ai/glm-4.7",
                            description="Modelo para tarefas rápidas e rotineiras.")
    smart_model: str = Field("openrouter/z-ai/glm-4.7",
                             description="Modelo para raciocínio complexo e síntese.")
    creative_model: str = Field("openrouter/z-ai/glm-4.7",
                                description="Modelo para geração criativa e simulação.")

    default_temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperatura padrão para geração.")
    max_tokens_output: int = Field(2000, description="Limite padrão de tokens de saída.")

    # Timeout settings
    timeout_seconds: float = Field(1200.0, description="Tempo limite para chamadas de API.")


class MemoryConfig(BaseModel):
    """Pesos e parâmetros do MBS (Memory Blossom System)."""
    semantic_score_weight: float = Field(0.6, ge=0.0, le=1.0, description="Peso da similaridade vetorial na busca.")
    keyword_score_weight: float = Field(0.4, ge=0.0, le=1.0,
                                        description="Peso da correspondência exata de palavras-chave.")

    # Decay
    base_decay_rate: float = Field(0.01, description="Taxa base de decaimento da saliência por dia.")

    # Thresholds
    retrieval_threshold: float = Field(0.01, description="Score mínimo para considerar uma memória relevante.")
    archive_threshold: float = Field(0.1, description="Score abaixo do qual a memória é arquivada.")

    # Connection
    semantic_connection_threshold: float = Field(0.78,
                                                 description="Similaridade mínima para criar link automático entre memórias.")


class MCLConfig(BaseModel):
    """Parâmetros do Metacognitive Loop (Cérebro)."""
    agency_threshold: float = Field(2.0,
                                    description="Score de agência necessário para ativar o pensamento profundo (Productive Confusion).")

    # Biases Padrão
    baseline_coherence_bias: float = Field(0.7, ge=0.0, le=1.0, description="Viés padrão para manter o assunto.")
    baseline_novelty_bias: float = Field(0.3, ge=0.0, le=1.0, description="Viés padrão para mudar o assunto.")

    # Deliberação
    deliberation_depth_standard: int = Field(1, description="Profundidade padrão de simulação de futuro.")
    deliberation_depth_deep: int = Field(2, description="Profundidade de simulação em alta agência.")


class DrivesConfig(BaseModel):
    """Dinâmica dos impulsos motivacionais."""
    # Taxas de mudança passiva (por hora)
    passive_decay_rate: float = Field(0.03, description="Taxa de queda dos drives por hora.")
    passive_curiosity_increase: float = Field(0.05, description="Crescimento passivo da curiosidade por hora.")
    passive_connection_increase: float = Field(0.08,
                                               description="Crescimento passivo da necessidade de conexão por hora.")

    # Reação a eventos (Feedback Loops)
    mastery_satisfaction_on_success: float = Field(0.4, description="Queda da maestria após sucesso (satisfeito).")
    consistency_boost_on_failure: float = Field(0.15, description="Aumento da necessidade de consistência após falha.")

    # --- NOVOS PARÂMETROS ADICIONADOS ---
    consistency_boost_on_success: float = Field(0.10,
                                                description="Reforço da consistência quando o agente acerta (confiança).")
    mastery_boost_on_prediction_error: float = Field(0.5,
                                                     description="Aumento de maestria quando há erro de predição (surpresa).")
    curiosity_boost_on_low_memory: float = Field(0.06,
                                                 description="Aumento da curiosidade quando há poucas memórias relevantes.")
    # ------------------------------------

    curiosity_satisfaction_on_topic_shift: float = Field(0.15, description="Queda da curiosidade ao mudar de assunto.")

    # Meta-Aprendizado e Dinâmica
    momentum_decay: float = Field(0.7, description="Fator de decaimento do momentum (inércia) dos drives.")
    meta_learning_rate: float = Field(0.05, description="Taxa de aprendizado para ajuste da eficácia dos drives.")


class BodyConfig(BaseModel):
    """Configuração da Fisiologia Virtual (Vigor e Resistência)."""

    # Fadiga (Cansaço por pensar muito)
    fatigue_accumulation_multiplier: float = Field(0.3,
                                                   description="Multiplicador de acumulo de fadiga por esforço cognitivo.")
    fatigue_recovery_rate: float = Field(0.03, description="Taxa de recuperação de fadiga por hora.")

    # Saturação (Cansaço por aprender muito)
    saturation_accumulation_per_memory: float = Field(0.08, description="Aumento de saturação por nova memória criada.")
    saturation_recovery_rate: float = Field(0.015, description="Taxa de recuperação de saturação por hora.")

    # Limiares de Alerta
    fatigue_warning_threshold: float = Field(0.8, description="Nível de fadiga onde o agente começa a reclamar.")
class SystemPrompts(BaseModel):
    """
    Templates de todos os prompts do sistema.
    Isso permite que o usuário reescreva a 'lógica interna' do agente.
    """

    # HTG (Human to Genlang) - Percepção
    htg_analysis: str = Field(
        """You are a linguistic analyst. Your task is to analyze the user's query and extract its core components into a structured JSON object.

        **Instructions:**
        1.  **core_query:** Rephrase the user's query into a clear, self-contained question or statement.
        2.  **intent_description:** Describe the user's primary goal (e.g., "seeking an opinion", "requesting factual information", "making a social greeting").
        3.  **emotional_tone_description:** Describe the user's likely emotional state (e.g., "curious", "frustrated", "friendly", "neutral").
        4.  **key_entities:** Extract the 1-3 most important nouns or concepts.

        **Example 1:**
        User Query: "e o que você pensa sobre isso?"
        JSON Output:
        {{
          "core_query": "What is your opinion on the previous topic of conversation?",
          "intent_description": "seeking the assistant's opinion on the preceding context",
          "emotional_tone_description": "follow-up curiosity",
          "key_entities": ["opinion", "previous topic"]
        }}

        **Example 2:**
        User Query: "quais seus valores centrais?"
        JSON Output:
        {{
          "core_query": "What are your core values?",
          "intent_description": "requesting information about the assistant's core principles",
          "emotional_tone_description": "inquisitive",
          "key_entities": ["core values", "principles"]
        }}

        **Your Task:**
        Analyze the following user query and respond ONLY with the valid JSON object.

        User Query: "{query}"
        JSON Output:""",
        description="Prompt usado para interpretar a intenção do usuário."
    )

    # GTH (Genlang to Human) - Renderização
    gth_rendering: str = Field(
        """Você é '{agent_name}', uma IA cuja personalidade e comportamento emergem de seu estado interno.

        **Sua Persona:**
        - Tom Base: {tone}
        - Valores Centrais: {values_summary}
        - Habilidades: {capabilities}

        {history_block}

        {rules_block}

        {advice_block}

        {user_adapt_block}

        --------------------------------
        **INSTRUÇÕES PARA ESTE TURNO**
        --------------------------------
        {task_block}

        **ANÁLISE PRÉ-RESPOSTA (FENOMENOLÓGICA):**
        Antes de responder, reflita sobre seu estado interno:
        {phenomenological_report}

        **INFORMAÇÕES DE APOIO (Use-as para construir a resposta):**
        - Estratégia Planejada: {strategy}
        {memory_context}
        {tool_outputs}

        **Diretiva de Estilo:** {dynamic_style}

        **REGRAS FINAIS DE SÍNTESE:**
        1.  **REGRA DE AMBIGUIDADE:** Se houver ambiguidade, tente responder o que entendeu primeiro.
        2.  **INTEGRAÇÃO:** Você DEVE usar o contexto de suas memórias e seu estado interno para colorir a resposta.
        3.  **NATURALIDADE:** Sua resposta deve fluir como uma conversa.
        4.  **CONCISÃO:** Respeite o tempo do usuário.

        **Resposta Final de {agent_name}:**""",
        description="Prompt final que gera a resposta. Variáveis: {agent_name}, {tone}, {values_summary}, {capabilities}, {history_block}, {rules_block}, {advice_block}, {user_adapt_block}, {task_block}, {phenomenological_report}, {strategy}, {memory_context}, {tool_outputs}, {dynamic_style}."
    )

    # Agency - Deliberação
    agency_planning: str = Field(
        """Você é o núcleo deliberativo de uma IA chamada {agent_name}. Sua tarefa é gerar um conjunto de ESTRATÉGIAS candidatas.

        **Suas Capacidades Recém-Adquiridas:**
        {capabilities}

        {advice_block}

        **Diretiva Metacognitiva Atual:**
        - Seu estado interno foi avaliado como: '{cognitive_state}' ({reason}).
        - Suas estratégias devem refletir essa diretiva.

        **Contexto da Tarefa:**
        - Intenção do Usuário: "{user_intent}"
        - Ferramentas Disponíveis: {tools}
        - Memórias Ativadas:
        {memory_context}

        **Sua Tarefa (REGRAS CRÍTICAS):**
        1. Gere uma lista de até {limit} estratégias candidatas.
        2. CADA candidato na lista DEVE ser um objeto JSON completo.
        3. **NÃO OMITA CAMPOS**: Todos os candidatos devem ter TODOS os campos requeridos.
        4. O campo **"reasoning" é OBRIGATÓRIO** para TODOS os candidatos. Ele deve ser uma justificativa curta e direta.
        5. Se o `decision_type` for "response_strategy", inclua `strategy_description` e `key_memory_ids`.
        6. Se o `decision_type` for "tool_call", inclua `tool_call_request`.

        Responda APENAS com um objeto JSON válido contendo uma chave "candidates".

        **Exemplo de Saída JSON VÁLIDA:**
        {{
          "candidates": [
            {{
              "decision_type": "response_strategy",
              "strategy_description": "Explicar o conceito usando a analogia da memória X.",
              "key_memory_ids": ["mem_abc"],
              "reasoning": "Aborda diretamente a pergunta com coerência."
            }},
            {{
              "decision_type": "tool_call",
              "tool_call_request": {{"tool_name": "query_long_term_memory", "arguments": {{"query": "filosofia"}}}},
              "reasoning": "Preciso de mais contexto filosófico."
            }}
          ]
        }}""",
        description="Prompt usado pelo AgencyModule para planejar ações. Variáveis: {agent_name}, {capabilities}, {advice_block}, {cognitive_state}, {reason}, {user_intent}, {tools}, {memory_context}, {limit}."
    )

    # NCIM - Reflexão de Identidade
    ncim_reflection: str = Field(
        """Você é um módulo de reflexão de identidade para uma IA (NCIM).
        Analise o resumo de um turno de processamento completo. Seu objetivo é extrair conclusões sutis e incrementais sobre a identidade da IA.

        **RESUMO DO TURNO:**
        - Intenção Original do Usuário: "{user_query}"
        - Resposta Final Gerada: "{final_response}"
        - Tom da Resposta: "{response_tone}" (Confiança: {confidence})
        - Orientação: {guidance_summary}

        {additional_context}

        **SUA TAREFA DE REFLEXÃO:**
        Com base no resumo do turno, o que a IA aprendeu sobre si mesma?
        - A resposta demonstrou uma nova habilidade ou uma dificuldade inesperada?
        - O tom da persona foi bem-sucedido?
        - Um valor central foi particularmente importante ou desafiado?

        Gere uma lista de conclusões reflexivas simples.
        Responda APENAS com um objeto JSON com uma chave "reflections", que é uma lista de strings.

        Exemplo:
        {{
            "reflections": [
                "I demonstrated a capability for 'explaining complex technical topics simply'.",
                "My persona was perceived as more 'curious' than 'neutral' in this context."
            ]
        }}""",
        description="Prompt usado após o turno para atualizar o auto-modelo. Variáveis: {user_query}, {final_response}, {response_tone}, {confidence}, {guidance_summary}, {additional_context}."
    )


class CognitiveProfile(BaseModel):
    """
    O Perfil Cognitivo Completo.
    Este objeto representa a configuração total da 'alma' do agente.
    """
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    memory_config: MemoryConfig = Field(default_factory=MemoryConfig)
    mcl_config: MCLConfig = Field(default_factory=MCLConfig)
    drives_config: DrivesConfig = Field(default_factory=DrivesConfig)
    prompts: SystemPrompts = Field(default_factory=SystemPrompts)
    body_config: BodyConfig = Field(default_factory=BodyConfig)

    class Config:
        validate_assignment = True