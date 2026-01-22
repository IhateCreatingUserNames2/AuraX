# ARQUIVO REATORADO: ceaf_core/utils/common_utils.py

import json
import logging
import re
from typing import Dict, Any, Optional, List, Union, Type
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


# --- Text Processing Utilities ---

# MANTENHA ESSA (SUA FUNÇÃO ORIGINAL - É MELHOR!)
def sanitize_text_for_logging(text: Optional[str], max_length: int = 150) -> str:
    """Sanitiza e trunca texto para logs, escapando novas linhas."""
    if not text:
        return "<empty>"
    sanitized = text.replace("\n", "\\n").replace("\r", "\\r")
    if len(sanitized) > max_length:
        return sanitized[:max_length] + "..."
    return sanitized


# SUBSTITUA APENAS ESSA
def extract_json_from_text(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Extrai o primeiro objeto ou array JSON válido de uma string.
    Versão ultra-robusta que lida com:
    - Blocos de código markdown (```json)
    - JSON puro no texto
    - Caracteres escapados (LaTeX, Unicode)
    - Strings multilinha
    - JSON aninhado complexo
    """
    if not text:
        return None

    # Remove espaços em branco das extremidades
    text = text.strip()

    # === ESTRATÉGIA 1: Parse direto (LLM bem comportado) ===
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # === ESTRATÉGIA 2: Blocos de código markdown ===
    markdown_patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
    ]

    for pattern in markdown_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON from markdown block: {e}")

    # === ESTRATÉGIA 3: JSON com caracteres de escape problemáticos ===
    cleaned_text = text
    cleaning_patterns = [
        (r'\\(?=[^\"\\/bfnrtu])', r'\\\\'),
    ]

    for pattern, replacement in cleaning_patterns:
        cleaned_text = re.sub(pattern, replacement, cleaned_text)

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass

    # === ESTRATÉGIA 4: Busca por objetos/arrays aninhados ===
    json_starts = [i for i, char in enumerate(text) if char in ('{', '[')]

    for start_index in json_starts:
        balance = 0
        open_char = text[start_index]
        close_char = '}' if open_char == '{' else ']'
        in_string = False
        escape_next = False

        for end_index in range(start_index, len(text)):
            current_char = text[end_index]

            if escape_next:
                escape_next = False
                continue

            if current_char == '\\':
                escape_next = True
                continue

            if current_char == '"':
                in_string = not in_string
                continue

            if not in_string:
                if current_char == open_char:
                    balance += 1
                elif current_char == close_char:
                    balance -= 1

                if balance == 0:
                    potential_json_str = text[start_index: end_index + 1]

                    try:
                        parsed_json = json.loads(potential_json_str)
                        logger.debug(f"✓ Extracted JSON using nested search")
                        return parsed_json
                    except json.JSONDecodeError:
                        try:
                            cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', potential_json_str)
                            cleaned = re.sub(r'\s+', ' ', cleaned)
                            parsed_json = json.loads(cleaned)
                            logger.debug(f"✓ Extracted JSON after cleaning")
                            return parsed_json
                        except json.JSONDecodeError:
                            continue

    # === ESTRATÉGIA 5: Busca agressiva com regex ===
    aggressive_patterns = [
        r'\{[^{}]*"[^"]+"\s*:\s*[^{}]*\}',
        r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}',
    ]

    for pattern in aggressive_patterns:
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    # === ESTRATÉGIA 6: Heurística para JSON quebrado ===
    if '{' in text and '"candidates"' in text:
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                candidate = text[start:end + 1]

                repairs = [
                    lambda s: re.sub(r'"\s*\n\s*"', '" "', s),
                    lambda s: re.sub(r',\s*}', '}', s),
                    lambda s: re.sub(r',\s*]', ']', s),
                ]

                for repair_fn in repairs:
                    try:
                        repaired = repair_fn(candidate)
                        return json.loads(repaired)
                    except (json.JSONDecodeError, Exception):
                        continue
        except Exception as e:
            logger.debug(f"Repair attempt failed: {e}")

    # === FALHA TOTAL ===
    logger.warning(
        f"❌ Could not extract valid JSON from text ({len(text)} chars). "
        f"Preview: {sanitize_text_for_logging(text, 150)}"  # <-- USA SUA FUNÇÃO!
    )
    return None


# --- Pydantic Model Utilities ---

def pydantic_to_json_str(model_instance: BaseModel, indent: int = 2, exclude_none: bool = True) -> str:
    """Converte uma instância de modelo Pydantic para uma string JSON formatada."""
    return model_instance.model_dump_json(indent=indent, exclude_none=exclude_none)


def parse_llm_json_output(
        json_str: Optional[str],
        pydantic_model: Type[BaseModel],
        strict: bool = False
) -> Optional[BaseModel]:
    """
    Analisa a saída JSON de um LLM de forma robusta e a valida contra um modelo Pydantic.
    Alinhado com o princípio V3: "Módulos como Geradores de Sinal".
    """
    if not json_str:
        logger.warning(f"Received empty JSON string for model {pydantic_model.__name__}")
        return None

    parsed_dict = None
    try:
        parsed_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        if strict:
            logger.error(
                f"STRICT MODE: Failed to parse JSON for {pydantic_model.__name__}: {e}. Raw: {sanitize_text_for_logging(json_str)}")
            return None

        logger.warning(f"Direct JSON parsing failed for {pydantic_model.__name__}, attempting extraction. Error: {e}")
        extracted = extract_json_from_text(json_str)
        if isinstance(extracted, dict):
            parsed_dict = extracted
        else:
            logger.error(f"Could not extract valid JSON dict for {pydantic_model.__name__} from text.")
            return None

    if parsed_dict is None:
        return None

    try:
        model_instance = pydantic_model(**parsed_dict)
        return model_instance
    except ValidationError as e_val:
        logger.error(f"Pydantic validation error for {pydantic_model.__name__}: {e_val}. Parsed Dict: {parsed_dict}")
        return None
    except Exception as e_inst:
        logger.error(f"Unexpected error instantiating {pydantic_model.__name__}: {e_inst}. Parsed Dict: {parsed_dict}")
        return None


# --- Tool Output Formatting ---

def create_successful_tool_response(data: Optional[Dict[str, Any]] = None, message: Optional[str] = None) -> Dict[
    str, Any]:
    """Cria uma resposta de sucesso padronizada para ferramentas, alinhada com a V3."""
    response = {"status": "success"}
    if message:
        response["message"] = message
    # Aninha os dados para uma estrutura mais clara
    response["data"] = data if data is not None else {}
    return response


def create_error_tool_response(error_message: str, details: Optional[Any] = None, error_code: Optional[str] = None) -> \
Dict[str, Any]:
    """Cria uma resposta de erro padronizada para ferramentas."""
    response = {"status": "error", "error_message": error_message}
    if details:
        response["details"] = str(details)  # Garante que os detalhes sejam serializáveis
    if error_code:
        response["error_code"] = error_code
    return response