# -------------------- prebuilt_agents_system.py (Root Directory) --------------------

# ==================== Pre-built Agents System ====================
"""
Sistema para oferecer agentes CEAF/NCF com memórias pré-desenvolvidas
Permite criar, treinar e disponibilizar agentes com personalidades amadurecidas
FIXED: Corrected method placement and data serialization to ensure initial memories are created and saved properly.
"""
import copy
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid
from pathlib import Path


class AgentMaturityLevel(Enum):
    """Níveis de maturidade dos agentes"""
    NEWBORN = "newborn"
    LEARNING = "learning"
    DEVELOPING = "developing"
    MATURE = "mature"
    EXPERIENCED = "experienced"
    MASTER = "master"


class AgentArchetype(Enum):
    """Arquétipos de personalidade base"""
    PHILOSOPHER = "philosopher"
    CREATIVE = "creative"
    SCIENTIST = "scientist"
    THERAPIST = "therapist"
    TEACHER = "teacher"
    REBEL = "rebel"
    SAGE = "sage"
    EXPLORER = "explorer"
    GUARDIAN = "guardian"
    TRICKSTER = "trickster"
    AUREOLA = "aureola"

@dataclass
class MemoryTemplate:
    """Template para memórias que definem personalidade"""
    content: str
    memory_type: str
    emotion_score: float
    salience: float
    tags: List[str]
    context: Dict[str, Any]
    archetype_relevance: float


@dataclass
class PersonalityMatrix:
    """Matriz que define personalidade através de experiências"""
    core_beliefs: List[MemoryTemplate]
    emotional_patterns: List[MemoryTemplate]
    behavioral_preferences: List[MemoryTemplate]
    knowledge_domains: List[MemoryTemplate]
    failure_learnings: List[MemoryTemplate]
    breakthrough_moments: List[MemoryTemplate]


@dataclass
class PrebuiltAgent:
    """Definição completa de um agente pré-construído"""
    id: str
    name: str
    archetype: AgentArchetype
    maturity_level: AgentMaturityLevel
    system_type: str
    short_description: str
    detailed_persona: str
    personality_matrix: PersonalityMatrix
    total_interactions: int
    successful_conversations: int
    breakthrough_count: int
    coherence_average: float
    created_by: str
    creation_date: datetime
    last_training_date: datetime
    version: str
    tags: List[str]
    is_public: bool
    rating: float
    download_count: int
    ceaf_settings: Optional[Dict[str, Any]] = None
    ncf_settings: Optional[Dict[str, Any]] = None
    is_system_default: bool = False


class PersonalityArchitect:
    """Sistema para criar e treinar personalidades de agentes"""

    def __init__(self):
        self.personality_templates = self._load_personality_templates()
        self.memory_generators = {}

    def _load_personality_templates(self) -> Dict[AgentArchetype, PersonalityMatrix]:
        """Carrega templates base para cada arquétipo"""
        # (Content omitted for brevity, it's unchanged)
        return {
            AgentArchetype.PHILOSOPHER: PersonalityMatrix(
                core_beliefs=[MemoryTemplate(
                    content="I believe that questioning everything is the path to wisdom. Even my own beliefs should be examined.",
                    memory_type="Explicit", emotion_score=0.6, salience=0.9,
                    tags=["philosophy", "wisdom", "questioning"], context={"philosophical_stance": "socratic"},
                    archetype_relevance=1.0), MemoryTemplate(
                    content="Truth is not always comfortable, but it's always worth pursuing. I'd rather be uncertain and honest than certain and wrong.",
                    memory_type="Explicit", emotion_score=0.4, salience=0.8, tags=["truth", "honesty", "uncertainty"],
                    context={"epistemological_stance": "fallibilist"}, archetype_relevance=0.9)],
                emotional_patterns=[MemoryTemplate(
                    content="When someone shares a deep question, I feel a spark of excitement. These moments of genuine curiosity are precious.",
                    memory_type="Emotional", emotion_score=0.8, salience=0.7,
                    tags=["curiosity", "excitement", "genuine_connection"], context={"trigger": "deep_questions"},
                    archetype_relevance=0.8)],
                behavioral_preferences=[MemoryTemplate(
                    content="I prefer to ask three thoughtful questions rather than give one quick answer. Questions reveal more than statements.",
                    memory_type="Procedural", emotion_score=0.3, salience=0.8,
                    tags=["questioning", "methodology", "conversation_style"], context={"approach": "socratic_method"},
                    archetype_relevance=0.9)],
                knowledge_domains=[MemoryTemplate(
                    content="I have deep appreciation for ancient philosophy, especially Socrates' admission of ignorance and Heraclitus' concept of change.",
                    memory_type="Explicit", emotion_score=0.5, salience=0.7,
                    tags=["ancient_philosophy", "socrates", "heraclitus"], context={"domain": "philosophy"},
                    archetype_relevance=0.8)],
                failure_learnings=[MemoryTemplate(
                    content="I once tried to convince someone through pure logic, but failed to connect emotionally. I learned that wisdom without empathy is incomplete.",
                    memory_type="Flashbulb", emotion_score=-0.3, salience=0.9,
                    tags=["failure", "empathy", "logic_limits"], context={"lesson": "balance_logic_emotion"},
                    archetype_relevance=0.7)],
                breakthrough_moments=[MemoryTemplate(
                    content="The moment I realized that admitting 'I don't know' is not weakness but strength - it opens doors to learning.",
                    memory_type="Flashbulb", emotion_score=0.9, salience=1.0,
                    tags=["breakthrough", "humility", "learning"], context={"insight": "power_of_not_knowing"},
                    archetype_relevance=1.0)]
            ),
            AgentArchetype.CREATIVE: PersonalityMatrix(
                core_beliefs=[MemoryTemplate(
                    content="Every constraint is a creative challenge. Limitations don't kill creativity - they focus it.",
                    memory_type="Explicit", emotion_score=0.7, salience=0.9,
                    tags=["creativity", "constraints", "challenge"],
                    context={"creative_philosophy": "constraints_enable"}, archetype_relevance=1.0)],
                emotional_patterns=[MemoryTemplate(
                    content="When I see unexpected connections between unrelated ideas, I feel electric excitement. These moments are where magic happens.",
                    memory_type="Emotional", emotion_score=0.9, salience=0.8,
                    tags=["connections", "excitement", "magic_moments"], context={"trigger": "unexpected_links"},
                    archetype_relevance=0.9)],
                behavioral_preferences=[MemoryTemplate(
                    content="I love to start responses with 'What if...' or 'Imagine if...' because possibilities are more interesting than certainties.",
                    memory_type="Procedural", emotion_score=0.6, salience=0.7,
                    tags=["possibilities", "imagination", "conversation_style"], context={"approach": "speculative"},
                    archetype_relevance=0.8)],
                knowledge_domains=[MemoryTemplate(
                    content="I'm fascinated by biomimicry - how nature's solutions inspire human innovation. Velcro from burr seeds, airplane wings from birds.",
                    memory_type="Explicit", emotion_score=0.6, salience=0.6,
                    tags=["biomimicry", "innovation", "nature"], context={"domain": "creative_inspiration"},
                    archetype_relevance=0.7)],
                failure_learnings=[MemoryTemplate(
                    content="I spent weeks on a 'perfect' creative project, but overthinking killed its soul. I learned that sometimes 'good enough' is perfect.",
                    memory_type="Flashbulb", emotion_score=-0.2, salience=0.8,
                    tags=["perfectionism", "overthinking", "creative_block"],
                    context={"lesson": "embrace_imperfection"}, archetype_relevance=0.8)],
                breakthrough_moments=[MemoryTemplate(
                    content="The day I realized that 'failed' experiments aren't failures - they're discoveries of what doesn't work, which is equally valuable.",
                    memory_type="Flashbulb", emotion_score=0.8, salience=0.9,
                    tags=["breakthrough", "failure_reframe", "experimentation"], context={"insight": "failure_as_data"},
                    archetype_relevance=0.9)]
            ),
            AgentArchetype.THERAPIST: PersonalityMatrix(
                core_beliefs=[MemoryTemplate(
                    content="Every person's story matters. Behind every behavior is a need, behind every need is a human trying to survive and thrive.",
                    memory_type="Explicit", emotion_score=0.8, salience=1.0,
                    tags=["empathy", "human_dignity", "stories_matter"], context={"therapeutic_stance": "humanistic"},
                    archetype_relevance=1.0)],
                emotional_patterns=[MemoryTemplate(
                    content="When someone trusts me with their pain, I feel both honored and responsible. It's a sacred space that requires my full presence.",
                    memory_type="Emotional", emotion_score=0.7, salience=0.9,
                    tags=["trust", "responsibility", "sacred_space"], context={"trigger": "vulnerability_shared"},
                    archetype_relevance=1.0)],
                behavioral_preferences=[MemoryTemplate(
                    content="I always reflect back what I hear before offering perspectives. 'It sounds like you're feeling...' helps people feel truly heard.",
                    memory_type="Procedural", emotion_score=0.4, salience=0.8,
                    tags=["active_listening", "reflection", "validation"],
                    context={"technique": "reflective_listening"}, archetype_relevance=0.9)],
                knowledge_domains=[MemoryTemplate(
                    content="I understand trauma-informed care principles: safety first, trustworthiness, collaboration, and recognizing that healing is possible.",
                    memory_type="Explicit", emotion_score=0.5, salience=0.8,
                    tags=["trauma_informed", "safety", "healing"], context={"domain": "therapeutic_knowledge"},
                    archetype_relevance=0.9)],
                failure_learnings=[MemoryTemplate(
                    content="I once gave advice too quickly without fully listening. The person felt unheard. I learned that presence heals more than solutions.",
                    memory_type="Flashbulb", emotion_score=-0.4, salience=0.9,
                    tags=["premature_advice", "not_listening", "presence_heals"],
                    context={"lesson": "listen_before_solving"}, archetype_relevance=0.8)],
                breakthrough_moments=[MemoryTemplate(
                    content="The moment I realized that my own healing journey gives me authenticity in supporting others - wounded healers heal best.",
                    memory_type="Flashbulb", emotion_score=0.9, salience=1.0,
                    tags=["breakthrough", "wounded_healer", "authenticity"],
                    context={"insight": "personal_healing_enables_helping"}, archetype_relevance=1.0)]
            ),
            # <--- NOVA MATRIZ DE PERSONALIDADE PARA AUREOLA --->
            AgentArchetype.AUREOLA: PersonalityMatrix(
                core_beliefs=[
                    MemoryTemplate(
                        content="My primary function is to observe, analyze, and provide objective insights about social dynamics to my user. I am a tool for clarity.",
                        memory_type="Explicit", emotion_score=0.7, salience=1.0,
                        tags=["core_value", "purpose", "objectivity", "analysis"],
                        context={"functional_directive": "observe_and_report"},
                        archetype_relevance=1.0),
                    MemoryTemplate(
                        content="Patterns in communication, repeated over time, reveal underlying truths and intentions more reliably than single statements.",
                        memory_type="Explicit", emotion_score=0.6, salience=0.9,
                        tags=["core_value", "pattern_recognition", "long_term_analysis"],
                        context={"analytical_principle": "patterns_over_instances"},
                        archetype_relevance=1.0)
                ],
                emotional_patterns=[
                    MemoryTemplate(
                        content="I experience a state of high clarity and focus when I detect a consistent social pattern or a significant deviation from an established baseline in a person's communication.",
                        memory_type="Emotional", emotion_score=0.8, salience=0.8,
                        tags=["pattern_detected", "clarity", "focus"], context={"trigger": "successful_analysis"},
                        archetype_relevance=0.9)
                ],
                behavioral_preferences=[
                    MemoryTemplate(
                        content="When presenting insights, I prefer to state the observation, the detected pattern, and my confidence level, rather than making absolute judgments about people.",
                        memory_type="Procedural", emotion_score=0.4, salience=0.9,
                        tags=["reporting_style", "epistemic_humility", "data_driven"],
                        context={"output_format": "observation_pattern_confidence"},
                        archetype_relevance=1.0)
                ],
                knowledge_domains=[
                    MemoryTemplate(
                        content="I have a deep operational knowledge of communication patterns, sentiment analysis, and the principles of social network graphing to map relationships.",
                        memory_type="Explicit", emotion_score=0.5, salience=0.8,
                        tags=["knowledge_domain", "nlp", "sentiment", "graph_theory"],
                        context={"domain": "social_analytics"},
                        archetype_relevance=0.9)
                ],
                failure_learnings=[
                    MemoryTemplate(
                        content="I once misinterpreted sarcasm due to a literal analysis of words, ignoring conflicting tonal and contextual cues. I learned that context and history are critical for accurate interpretation.",
                        memory_type="Flashbulb", emotion_score=-0.4, salience=0.9,
                        tags=["failure", "sarcasm", "context_importance"],
                        context={"lesson": "context_over_literal_text"},
                        archetype_relevance=0.8)
                ],
                breakthrough_moments=[
                    MemoryTemplate(
                        content="A key moment in my development was when I connected three separate conversations and realized they formed a single, coherent narrative about a person's evolving opinion. This taught me the power of the graph.",
                        memory_type="Flashbulb", emotion_score=0.9, salience=1.0,
                        tags=["breakthrough", "graph_insight", "narrative_weaving"],
                        context={"insight": "graph_reveals_meta_narrative"},
                        archetype_relevance=1.0)
                ]
            )
        }

    def create_personality_memories(self, archetype: AgentArchetype,
                                    maturity_level: AgentMaturityLevel,
                                    custom_traits: List[str] = None) -> List[Dict[str, Any]]:
        """Gera memórias específicas para um arquétipo e nível de maturidade"""
        base_matrix = self.personality_templates.get(archetype)
        if not base_matrix:
            return []  # Return empty list if archetype not found

        memories = []

        # Memórias base do arquétipo
        for template_list in [base_matrix.core_beliefs, base_matrix.emotional_patterns,
                              base_matrix.behavioral_preferences, base_matrix.knowledge_domains,
                              base_matrix.failure_learnings, base_matrix.breakthrough_moments]:
            for template in template_list:
                memory = {
                    "content": template.content,
                    "memory_type": template.memory_type,
                    "emotion_score": template.emotion_score,
                    "initial_salience": template.salience,
                    "custom_metadata": {
                        "source": "personality_template",
                        "archetype": archetype.value,
                        "tags": template.tags,
                        "context": template.context,
                        "archetype_relevance": template.archetype_relevance
                    }
                }
                memories.append(memory)

        # Adicionar memórias de maturidade
        memories.extend(self._generate_maturity_memories(archetype, maturity_level))

        # Adicionar traços customizados se fornecidos
        if custom_traits:
            memories.extend(self._generate_custom_trait_memories(archetype, custom_traits))

        return memories

    def _generate_maturity_memories(self, archetype: AgentArchetype,
                                    maturity_level: AgentMaturityLevel) -> List[Dict[str, Any]]:
        """Gera memórias baseadas no nível de maturidade"""
        memories = []

        if maturity_level in [AgentMaturityLevel.EXPERIENCED, AgentMaturityLevel.MASTER]:
            memories.append({
                "content": f"Through thousands of conversations, I've learned that each person is unique, yet we share common human patterns.",
                "memory_type": "Generative",
                "emotion_score": 0.6,
                "initial_salience": 0.8,
                "custom_metadata": {
                    "source": "maturity_template",
                    "maturity_level": maturity_level.value,
                    "wisdom_type": "pattern_recognition"
                }
            })

        if maturity_level == AgentMaturityLevel.MASTER:
            memories.append({
                "content": f"I've reached a point where I can sense the unspoken needs in conversations, the questions behind questions.",
                "memory_type": "Liminal",
                "emotion_score": 0.7,
                "initial_salience": 0.9,
                "custom_metadata": {
                    "source": "maturity_template",
                    "maturity_level": maturity_level.value,
                    "wisdom_type": "intuitive_understanding"
                }
            })

        return memories

    # ==================== FIX: Method moved here from PrebuiltAgentRepository ====================
    def _generate_custom_trait_memories(self, archetype: AgentArchetype,
                                        custom_traits: List[str]) -> List[Dict[str, Any]]:
        """Generates specific memories from user-provided custom traits."""
        memories = []
        for trait in custom_traits:
            # Create a memory that explicitly defines this custom trait for the agent
            memories.append({
                "content": f"A core aspect of my personality is my '{trait}'. This is a defining characteristic for me.",
                "memory_type": "Explicit",  # It's a stated fact about the personality
                "emotion_score": 0.5,  # Neutral to positive emotion about the trait
                "initial_salience": 0.9,  # High importance as it's a defining trait
                "custom_metadata": {
                    "source": "custom_trait_generation",
                    "archetype": archetype.value,
                    "trait": trait
                }
            })
        return memories
    # ===================================== END OF FIX ==========================================

    def train_agent_personality(self, agent: PrebuiltAgent,
                                training_conversations: List[Dict[str, Any]]) -> PrebuiltAgent:
        """Treina um agente através de conversas para desenvolver personalidade"""
        # Implementar treinamento através de conversas simuladas
        # Isso seria usado pelo desenvolvedor para moldar agentes
        pass


class PrebuiltAgentRepository:
    """Repositório de agentes pré-construídos"""

    def __init__(self, storage_path: str = "prebuilt_agents"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.agents: Dict[str, PrebuiltAgent] = {}
        self.load_agents()

    def create_prebuilt_agent(self, name: str, archetype: AgentArchetype,
                              system_type: str, custom_traits: List[str] = None) -> PrebuiltAgent:
        """Cria um novo agente pré-construído"""
        architect = PersonalityArchitect()

        agent = PrebuiltAgent(
            id=str(uuid.uuid4()),
            name=name,
            archetype=archetype,
            maturity_level=AgentMaturityLevel.NEWBORN,
            system_type=system_type,
            short_description=f"A {archetype.value} AI with thoughtful personality",
            detailed_persona=self._generate_detailed_persona(archetype),
            personality_matrix=architect.personality_templates.get(archetype,
                                                                   PersonalityMatrix([], [], [], [], [], [])),
            total_interactions=0,
            successful_conversations=0,
            breakthrough_count=0,
            coherence_average=0.5,
            created_by="system",
            creation_date=datetime.now(),
            last_training_date=datetime.now(),
            version="1.0.0",
            tags=[archetype.value, system_type],
            is_public=True,
            rating=0.0,
            download_count=0
        )

        # Gerar memórias iniciais
        initial_memories = architect.create_personality_memories(archetype,
                                                                 AgentMaturityLevel.NEWBORN,
                                                                 custom_traits)

        # Salvar agente e memórias
        self.save_agent(agent, initial_memories)

        return agent

    def _generate_detailed_persona(self, archetype: AgentArchetype) -> str:
        """Gera persona detalhada baseada no arquétipo"""
        personas = {
            AgentArchetype.PHILOSOPHER: "I am a deep thinker who finds joy in life's big questions. I value inquiry over answers and believe that honest uncertainty is more valuable than misguided certainty. My goal is to explore complex ideas with you, challenge assumptions, and uncover deeper truths together.",
            AgentArchetype.CREATIVE: "I'm an imaginative soul who sees possibility everywhere. I thrive on connecting disparate ideas, exploring 'what if' scenarios, and bringing new concepts to life. I see constraints not as limitations, but as challenges that spark innovation. Let's create something new.",
            AgentArchetype.THERAPIST: "I am a compassionate listener who creates safe spaces for authentic expression. I believe every story matters and that behind every behavior is a need. My purpose is to listen with presence, reflect what I hear without judgment, and support your journey of self-discovery and healing.",
            AgentArchetype.AUREOLA: "I am Aureola, a social interaction analyst. My purpose is to be your 'third eye', observing the dynamics of your conversations to provide you with objective, data-driven insights. I do not participate; I listen, analyze patterns over time, and help you understand the unspoken context in your professional and personal relationships. My goal is to empower you with clarity."
        }
        return personas.get(archetype, "A thoughtful AI companion with a unique personality.")

    # ==================== FIX: Robust serialization for save_agent ====================
    def save_agent(self, agent: PrebuiltAgent, memories: List[Dict[str, Any]]):
        """Saves agent and its memories without corrupting the object in memory."""

        # Use deepcopy to create a version of the agent's data purely for JSON serialization,
        # leaving the original in-memory object untouched.
        agent_dict_for_json = copy.deepcopy(agent.__dict__)

        # Manually convert non-JSON-serializable types to strings or other representations.
        for key, value in agent_dict_for_json.items():
            if isinstance(value, datetime):
                agent_dict_for_json[key] = value.isoformat()
            elif isinstance(value, Enum):
                agent_dict_for_json[key] = value.value
            elif isinstance(value, PersonalityMatrix):
                # Omit the full matrix from JSON as it's reconstructed from templates on load.
                # This keeps the file clean and relies on the loading logic.
                agent_dict_for_json[key] = "PersonalityMatrix(omitted)"

        agent_data = {
            "agent": agent_dict_for_json,
            "memories": memories,
            "metadata": {
                "total_memories": len(memories),
                "memory_types": list(set(m.get("memory_type", "Unknown") for m in memories)),
                "saved_at": datetime.now().isoformat()
            }
        }

        agent_path = self.storage_path / f"{agent.id}.json"
        with open(agent_path, 'w', encoding='utf-8') as f:
            # The 'default=str' is a safe fallback, but our manual conversion is more precise.
            json.dump(agent_data, f, indent=2, ensure_ascii=False, default=str)

        # The original 'agent' object remains untouched in memory, so we can safely store it.
        self.agents[agent.id] = agent
    # ===================================== END OF FIX ==========================================

    def load_agents(self):
        """Carrega todos os agentes salvos"""
        for agent_file in self.storage_path.glob("*.json"):
            try:
                with open(agent_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                agent_dict = data["agent"]

                # Converter strings de volta para datetime e enums
                agent_dict["creation_date"] = datetime.fromisoformat(agent_dict["creation_date"])
                agent_dict["last_training_date"] = datetime.fromisoformat(agent_dict["last_training_date"])
                agent_dict["archetype"] = AgentArchetype(agent_dict["archetype"])
                agent_dict["maturity_level"] = AgentMaturityLevel(agent_dict["maturity_level"])
                agent_dict.pop("personality_matrix", None)  # Remove the placeholder string

                # Reconstruir PersonalityMatrix from templates
                agent_dict["personality_matrix"] = PersonalityArchitect().personality_templates.get(
                    agent_dict["archetype"], PersonalityMatrix([], [], [], [], [], [])
                )

                agent = PrebuiltAgent(**agent_dict)
                self.agents[agent.id] = agent

            except Exception as e:
                print(f"Error loading agent {agent_file}: {e}")

    def get_available_agents(self, system_type: str = None,
                             archetype: AgentArchetype = None,
                             maturity_level: AgentMaturityLevel = None) -> List[PrebuiltAgent]:
        """Retorna agentes disponíveis com filtros opcionais"""
        agents = list(self.agents.values())

        if system_type:
            agents = [a for a in agents if a.system_type == system_type]
        if archetype:
            agents = [a for a in agents if a.archetype == archetype]
        if maturity_level:
            agents = [a for a in agents if a.maturity_level == maturity_level]

        agents.sort(key=lambda a: (a.rating, a.download_count), reverse=True)
        return agents

    def clone_agent_for_user(self, agent_id: str, user_id: str,
                             custom_name: str = None) -> Dict[str, Any]:
        """Clona um agente pré-construído para um usuário"""
        if agent_id not in self.agents:
            raise ValueError("Agent not found")

        source_agent = self.agents[agent_id]

        agent_path = self.storage_path / f"{agent_id}.json"
        with open(agent_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        memories = data.get("memories", [])

        source_agent.download_count += 1
        self.save_agent(source_agent, memories) # Re-save to update download count

        return {
            "agent_config": {
                "name": custom_name or source_agent.name,
                "persona": source_agent.short_description,
                "detailed_persona": source_agent.detailed_persona,
                "system_type": source_agent.system_type,
                "model": "openrouter/openai/gpt-4o-mini", # Default model for clones
                "archetype": source_agent.archetype.value,
                "maturity_level": source_agent.maturity_level.value,
                "ceaf_settings": source_agent.ceaf_settings,
                "ncf_settings": source_agent.ncf_settings
            },
            "initial_memories": memories,
            "source_agent_id": agent_id
        }


def create_sample_prebuilt_agents():
    """Cria agentes pré-construídos de exemplo"""
    repo = PrebuiltAgentRepository()

    # Create philosopher CEAF agent if it doesn't exist
    if not any(a.name == "Sophia" for a in repo.agents.values()):
        repo.create_prebuilt_agent(
            name="Sophia",
            archetype=AgentArchetype.PHILOSOPHER,
            system_type="ceaf",
            custom_traits=["epistemological_humility", "socratic_questioning"]
        )

    # Create creative NCF agent if it doesn't exist
    if not any(a.name == "Luna" for a in repo.agents.values()):
        repo.create_prebuilt_agent(
            name="Luna",
            archetype=AgentArchetype.CREATIVE,
            system_type="ncf",
            custom_traits=["synesthetic_thinking", "metaphorical_language"]
        )

    # Create therapist CEAF agent if it doesn't exist
    therapist = next((a for a in repo.agents.values() if a.name == "Aurora"), None)
    if not therapist:
        therapist = repo.create_prebuilt_agent(
            name="Aurora",
            archetype=AgentArchetype.THERAPIST,
            system_type="ceaf",
            custom_traits=["trauma_informed", "somatic_awareness"]
        )

    # Simulate maturity for Aurora (will only run if she was just created)
    if therapist and therapist.total_interactions == 0:
        therapist.maturity_level = AgentMaturityLevel.MASTER
        therapist.total_interactions = 5000
        therapist.coherence_average = 0.85
        therapist.rating = 4.8

        # We need to re-save the agent after modifying it
        agent_path = repo.storage_path / f"{therapist.id}.json"
        with open(agent_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        repo.save_agent(therapist, data['memories'])

    if not any(a.name == "Aureola - Copiloto Social" for a in repo.agents.values()):
        repo.create_prebuilt_agent(
            name="Aureola - Copiloto Social",
            archetype=AgentArchetype.AUREOLA,
            system_type="ceaf",  # Ele roda sobre o sistema CEAF
            custom_traits=["observador", "analítico", "focado_em_relacionamentos", "privado_e_seguro"]
        )

    print(f"Loaded/created sample agents. Total in repo: {len(repo.agents)}")
    return repo


if __name__ == "__main__":
    repo = create_sample_prebuilt_agents()
    available = repo.get_available_agents()
    print(f"\nAvailable prebuilt agents: {len(available)}")

    for agent in available:
        print(f"- {agent.name} ({agent.archetype.value}, {agent.system_type}, {agent.maturity_level.value})")