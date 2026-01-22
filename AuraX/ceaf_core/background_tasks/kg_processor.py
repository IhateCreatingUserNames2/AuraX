# ceaf_core/background_tasks/kg_processor.py
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_core.core_schema import ValidationInfo

from ceaf_core.services.llm_service import LLMService
from ceaf_core.services.mbs_memory_service import MBSMemoryService
from ceaf_core.modules.memory_blossom.memory_types import (
    ExplicitMemory, KGEntityRecord, KGRelationRecord, MemorySourceType,
    MemorySalience, KGEntityType
)
from ceaf_core.utils.common_utils import extract_json_from_text

logger = logging.getLogger("KGProcessor")


# +++ START OF FIX: Make KGEntity model more robust to LLM output +++
class KGEntity(BaseModel):
    id_str: str
    label: str
    type: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)

    @field_validator('id_str', mode='before')
    @classmethod
    def normalize_id_field(cls, v, info: 'ValidationInfo'):
        """
        Aceita nomes de chave incorretos comuns para o ID da entidade ('_str', 'id', 'entity_id')
        e os normaliza para 'id_str' antes da validação.
        """
        # Se 'id_str' já foi fornecido corretamente, use-o.
        if v:
            return v

        # O objeto 'info.data' contém o dicionário de entrada bruto completo.
        raw_data = info.data

        # Verifica variações comuns e retorna a primeira encontrada.
        if '_str' in raw_data:
            return raw_data['_str']
        if 'id' in raw_data:
            return raw_data['id']
        if 'entity_id' in raw_data:
            return raw_data['entity_id']

        # Se nenhum for encontrado, retorna o valor original (None) e deixa a
        # validação padrão do Pydantic levantar o erro "Field required".
        return v

    @field_validator('label', mode='before')
    @classmethod
    def accept_name_for_label(cls, v, info: 'ValidationInfo'):
        """
        Aceita 'name' como uma alternativa para o campo 'label'.
        """
        if v:
            return v
        if 'name' in info.data:
            return info.data['name']
        return v




class KGRelation(BaseModel):
    source_id_str: str
    target_id_str: str
    label: str
    context: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class KGSynthesisOutput(BaseModel):
    extracted_entities: List[KGEntity] = Field(default_factory=list)
    extracted_relations: List[KGRelation] = Field(default_factory=list)


class KGProcessor:
    def __init__(self, llm_service: LLMService, memory_service: MBSMemoryService):
        self.llm = llm_service
        self.mbs = memory_service
        allowed_entity_types = [e.value for e in KGEntityType]

        self.synthesis_prompt_template = f"""
                You are a Knowledge Graph Synthesizer. Your function is to process text from an AI's memories
                and extract structured knowledge as entities and relationships.

                **CRITICAL RULES:**
                1.  **Entity `type` field:** The `type` field for each entity MUST be one of the following exact values: {json.dumps(allowed_entity_types)}.
                2.  **Relation `id_str` fields:** Every object in the `extracted_relations` list MUST have BOTH a `source_id_str` and a `target_id_str`.
                3.  **Output Format:** Your response MUST BE a single, valid JSON object with top-level keys "extracted_entities" and "extracted_relations".

                **Memory Text to Process:**
                ---
                {{memory_text}}
                ---

                **Correct JSON Output Schema (using 'label' for entities):**
                {{{{
                  "extracted_entities": [
                    {{{{
                      "id_str": "...", "label": "Entity Label", "type": "...", ...
                    }}}}
                  ],
                  "extracted_relations": [
                    {{{{ "source_id_str": "...", "target_id_str": "...", "label": "...", ... }}}}
                  ]
                }}}}

                **Your JSON Output:**
                """

        self.aureola_synthesis_prompt_template = f"""
                        You are a Social Dynamics Analyst. Your function is to process a conversation transcript
                        and extract a detailed social knowledge graph, focusing on the relationships and interactions between people.

                        **CRITICAL RULES:**
                        1.  **Entity `type` field:** MUST be one of {json.dumps(allowed_entity_types)}. Use 'person' for speakers (e.g., "speaker_1", "user").
                        2.  **Relation `id_str` fields:** Every relation MUST have `source_id_str` and `target_id_str`.
                        3.  **Focus on Social Dynamics:** Your main goal is to map the social interactions.

                        **Conversation Transcript to Process:**
                        ---
                        {{memory_text}}
                        ---

                        **Analysis Task:**
                        1.  Identify each speaker (e.g., "[speaker_1]") as a 'person' entity.
                        2.  Extract key 'concept' or 'issue' entities discussed.
                        3.  Create relationships that describe the social dynamic. Use descriptive labels.
                        4.  **Pay CRITICAL attention to relationship indicators.** If one speaker refers to another with a relational term, create a specific relationship.
                            - **Kinship:** "pai", "mãe", "filho", "irmão" -> Create relations like `IS_FATHER_OF`, `IS_SON_OF`.
                            - **Professional:** "chefe", "colega", "cliente" -> Create relations like `IS_BOSS_OF`, `IS_COLLEAGUE_OF`.
                            - **Social:** "amigo", "vizinho".
                            
                        **Examples of GOOD Relationship Labels:**
                        - `agreed_with`
                        - `disagreed_with`
                        - `asked_question_to`
                        - `offered_support_to`
                        - `expressed_frustration_about` (source: person, target: concept)
                        - `shared_personal_story_about` (source: person, target: concept)
                        - `challenged_idea_of` (source: person, target: person)
                        - `agreed_with`
                        - `is_father_of`  # Exemplo de relação de parentesco
                        - `is_boss_of`   # Exemplo de relação profissional
                        - `asked_question_to`
                        - `expressed_frustration_about` (source: person, target: concept)

                        **Your JSON Output:**
                        """


    async def _repair_json_with_llm(self, broken_json_str: str, error_message: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to repair a broken JSON string using an LLM call.
        """
        logger.warning(f"KGProcessor: Attempting to repair broken JSON with LLM. Error: {error_message}")
        prompt = f"""
        The following text was intended to be a valid JSON object, but it failed validation with this error:
        Error: "{error_message}"

        Broken JSON Text:
        ```json
        {broken_json_str}
        ```

        Your task is to fix the JSON text so it becomes valid according to the schema.
        - Correct any syntax errors (missing commas, brackets, quotes).
        - Ensure all required fields are present. For entities, use the field name "label" instead of "name". For relations, ensure 'source_id_str' and 'target_id_str' are present.
        - Do NOT change the content of the fields, only the structure and field names.

        Respond ONLY with the corrected, valid JSON object.
        """
        try:
            repaired_str = await self.llm.ainvoke(
                self.llm.config.fast_model,
                prompt,
                temperature=0.0
            )
            repaired_json = extract_json_from_text(repaired_str)
            if isinstance(repaired_json, dict):
                logger.info("KGProcessor: Successfully repaired JSON with LLM.")
                return repaired_json
            else:
                logger.error(
                    f"KGProcessor: LLM-based JSON repair did not return a valid dictionary. Response: {repaired_str}")
        except Exception as e:
            logger.error(f"KGProcessor: An exception occurred during the LLM-based JSON repair process: {e}")

        return None

    # ... (process_memories_to_kg method is unchanged as the fix is in the Pydantic model) ...
    async def process_memories_to_kg(self, memories: List[ExplicitMemory]) -> Tuple[int, int]:
        # This function does not need to be changed. The Pydantic model fix handles the logic.
        if not memories:
            return 0, 0

        total_entities = 0
        total_relations = 0

        for memory in memories:
            text_content, _ = await self.mbs._get_searchable_text_and_keywords(memory)
            if not text_content or len(text_content.split()) < 5:
                continue

            prompt = self.synthesis_prompt_template.format(memory_text=text_content)

            try:
                response_str = await self.llm.ainvoke(
                    self.llm.config.smart_model,
                    prompt,
                    temperature=0.0
                )
                json_output = extract_json_from_text(response_str)

                if not json_output:
                    logger.warning(f"KGProcessor: No valid JSON extracted for memory {memory.memory_id}. Skipping.")
                    continue

                if isinstance(json_output, list):
                    logger.warning(
                        f"KGProcessor: LLM returned a list instead of a dict for memory {memory.memory_id}. Attempting to fix.")
                    if json_output and isinstance(json_output[0], dict) and ('id_str' in json_output[0]):
                        json_output = {"extracted_entities": json_output, "extracted_relations": []}
                    else:
                        logger.error(
                            f"KGProcessor: Could not safely repair list-based JSON output for memory {memory.memory_id}. Skipping.")
                        continue

                synthesis_result = None
                try:
                    synthesis_result = KGSynthesisOutput.model_validate(json_output)
                except ValidationError as e:
                    logger.error(
                        f"KGProcessor: Pydantic validation failed for memory {memory.memory_id}. Attempting LLM repair. Details: {e}")

                    repaired_json = await self._repair_json_with_llm(json.dumps(json_output), str(e))
                    if repaired_json:
                        try:
                            synthesis_result = KGSynthesisOutput.model_validate(repaired_json)
                            logger.info(f"KGProcessor: LLM repair successful for memory {memory.memory_id}.")
                        except ValidationError as e2:
                            logger.error(
                                f"KGProcessor: Repaired JSON still failed validation for memory {memory.memory_id}: {e2}")
                            continue
                    else:
                        logger.error(
                            f"KGProcessor: LLM repair did not return valid JSON for memory {memory.memory_id}. Skipping.")
                        continue

                if not synthesis_result:
                    logger.error(
                        f"KGProcessor: Could not obtain a valid synthesis result for memory {memory.memory_id} after all steps. Skipping.")
                    continue

                # Commit Entities
                for entity_data in synthesis_result.extracted_entities:
                    entity_type_enum = KGEntityType.OTHER
                    try:
                        entity_type_str = getattr(entity_data, 'type', 'OTHER') or 'OTHER'
                        entity_type_enum = KGEntityType[entity_type_str.upper()]
                    except KeyError:
                        logger.warning(f"Unknown KG entity type '{entity_data.type}' from LLM. Defaulting to OTHER.")

                    entity_record = KGEntityRecord(
                        entity_id_str=entity_data.id_str,
                        label=entity_data.label,
                        entity_type=entity_type_enum,
                        description=entity_data.description,
                        attributes=entity_data.attributes,
                        aliases=entity_data.aliases,
                        source_type=MemorySourceType.INTERNAL_REFLECTION,
                        salience=MemorySalience.MEDIUM,
                        metadata={"source_memory_id": memory.memory_id}
                    )
                    await self.mbs.add_specific_memory(entity_record)
                    total_entities += 1

                # Commit Relations
                for relation_data in synthesis_result.extracted_relations:
                    relation_record = KGRelationRecord(
                        source_entity_id_str=relation_data.source_id_str,
                        target_entity_id_str=relation_data.target_id_str,
                        relation_label=relation_data.label,
                        description=relation_data.context,
                        attributes=relation_data.attributes,
                        source_type=MemorySourceType.INTERNAL_REFLECTION,
                        salience=MemorySalience.MEDIUM,
                        metadata={"source_memory_id": memory.memory_id}
                    )
                    await self.mbs.add_specific_memory(relation_record)
                    total_relations += 1

            except Exception as e:
                logger.error(f"KGProcessor: Unhandled exception while processing memory {memory.memory_id}: {e}",
                             exc_info=True)
                continue

        return total_entities, total_relations

    async def process_aureola_transcription_to_kg(self, memories: List[ExplicitMemory]) -> Tuple[int, int]:
        """
        Processa memórias de transcrição da Aureola com um prompt focado em análise social.
        """
        if not memories:
            return 0, 0

        total_entities = 0
        total_relations = 0

        for memory in memories:
            text_content, _ = await self.mbs._get_searchable_text_and_keywords(memory)
            if not text_content or len(text_content.split()) < 3:
                continue

            # Usa o novo prompt especializado
            prompt = self.aureola_synthesis_prompt_template.format(memory_text=text_content)

            try:
                # O resto da lógica é idêntico ao process_memories_to_kg
                # Isso é bom, pois reutilizamos a lógica de parsing e salvamento
                response_str = await self.llm.ainvoke(
                    self.llm.config.smart_model,
                    prompt,
                    temperature=0.0
                )
                json_output = extract_json_from_text(response_str)

                if not json_output:
                    continue

                synthesis_result = None
                try:
                    synthesis_result = KGSynthesisOutput.model_validate(json_output)
                except ValidationError as e:
                    repaired_json = await self._repair_json_with_llm(json.dumps(json_output), str(e))
                    if repaired_json:
                        try:
                            synthesis_result = KGSynthesisOutput.model_validate(repaired_json)
                        except ValidationError as e2:
                            logger.error(
                                f"KGProcessor (Aureola): Repaired JSON still failed validation for memory {memory.memory_id}: {e2}")
                            continue

                if not synthesis_result:
                    continue

                # Commit Entities
                for entity_data in synthesis_result.extracted_entities:
                    entity_type_enum = KGEntityType.OTHER
                    try:
                        entity_type_enum = KGEntityType[entity_data.type.upper()]
                    except KeyError:
                        logger.warning(f"Unknown KG entity type '{entity_data.type}' from LLM. Defaulting to OTHER.")

                    entity_record = KGEntityRecord(
                        entity_id_str=entity_data.id_str,
                        label=entity_data.label,
                        entity_type=entity_type_enum,
                        description=entity_data.description,
                        attributes=entity_data.attributes,
                        aliases=entity_data.aliases,
                        source_type=MemorySourceType.INTERNAL_REFLECTION,
                        salience=MemorySalience.MEDIUM,
                        metadata={"source_memory_id": memory.memory_id}
                    )
                    await self.mbs.add_specific_memory(entity_record)
                    total_entities += 1

                # Commit Relations
                for relation_data in synthesis_result.extracted_relations:
                    relation_record = KGRelationRecord(
                        source_entity_id_str=relation_data.source_id_str,
                        target_entity_id_str=relation_data.target_id_str,
                        relation_label=relation_data.label,
                        description=relation_data.context,
                        attributes=relation_data.attributes,
                        source_type=MemorySourceType.INTERNAL_REFLECTION,
                        salience=MemorySalience.MEDIUM,
                        metadata={"source_memory_id": memory.memory_id}
                    )
                    await self.mbs.add_specific_memory(relation_record)
                    total_relations += 1

            except Exception as e:
                logger.error(
                    f"KGProcessor (Aureola): Unhandled exception while processing memory {memory.memory_id}: {e}",
                    exc_info=True)
                continue

        return total_entities, total_relations