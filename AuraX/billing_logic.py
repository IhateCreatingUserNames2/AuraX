# billing_logic.py
import logging
from database.models import User, CreditTransaction
import asyncio
logger = logging.getLogger(__name__)

# --- FONTE ÚNICA DA VERDADE PARA CUSTOS ---

# Tabela de custos por 1 MILHÃO de tokens (em USD)
# Fonte: OpenRouter.ai (verifique os preços para atualizações)
MODEL_API_COSTS_USD = {
    # Grátis
    "openrouter/deepseek/deepseek-r1-0528:free": (0.0, 0.0),
    "openrouter/horizon-beta": (0.0, 0.0),

    # Valor
    "openrouter/openai/gpt-4o-mini": (0.15, 0.60),
    "openrouter/google/gemini-2.5-flash": (0.30, 2.50),

    # Avançado
    "openrouter/openai/gpt-4o": (1.5, 6.0),
    "openrouter/anthropic/claude-3.5-sonnet": (3.0, 15.0),

    # Padrão para modelos não listados
    "default": (0.5, 1.5)
}

# Tabela de preços para o usuário final (em créditos por 1 MILHÃO de tokens)
# Esta é a tabela que define quanto o usuário paga, incluindo sua margem de lucro.
MODEL_USER_COSTS_CREDITS = {
    # Grátis
    "openrouter/deepseek/deepseek-r1-0528:free": 100,  # Custo nominal para evitar abuso
    "openrouter/horizon-beta": 100,

    # Valor
    "openrouter/openai/gpt-4o-mini": 1500,
    "openrouter/google/gemini-2.5-flash": 2000,

    # Avançado
    "openrouter/openai/gpt-4o": 15000,
    "openrouter/anthropic/claude-3.5-sonnet": 25000,

    # Padrão
    "default": 5000
}


# --- FUNÇÕES DE LÓGICA DE NEGÓCIO ---

def calculate_credit_cost_from_tokens(model_name: str, input_tokens: int, output_tokens: int) -> int:
    """
    Calcula o custo final em créditos para o usuário com base nos tokens.
    """
    # Usa o custo de usuário, não o custo da API
    cost_per_million_tokens = MODEL_USER_COSTS_CREDITS.get(model_name, MODEL_USER_COSTS_CREDITS["default"])

    # Simplificando: usamos um custo único por milhão de tokens para o usuário, em vez de entrada/saída separados
    total_tokens = input_tokens + output_tokens
    cost = (total_tokens / 100_000_000) * cost_per_million_tokens

    # Garante um custo mínimo de 1 crédito para qualquer interação
    return max(1, int(cost))


async def check_and_debit_credits(
        db_session,
        user_id: str,
        agent_id: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int
) -> bool:
    """
    Verifica o saldo do usuário, debita o custo e registra a transação de forma não-bloqueante.
    Retorna True se bem-sucedido, False se o usuário não tiver créditos suficientes.
    """
    cost = calculate_credit_cost_from_tokens(model_name, input_tokens, output_tokens)

    # --- PADRÃO NÃO-BLOQUEANTE PARA OPERAÇÕES DE DB SÍNCRONAS ---
    # A lógica de banco de dados (que é síncrona/bloqueante) é encapsulada
    # em uma função interna para ser executada em um thread separado.
    def db_operations() -> bool:
        """Contém todas as interações síncronas com o banco de dados."""
        user = db_session.query(User).filter(User.id == user_id).first()
        if not user:
            logger.error(f"Usuário {user_id} não encontrado para débito de créditos.")
            return False

        if user.credits < cost:
            logger.warning(f"Usuário {user_id} com créditos insuficientes. Saldo: {user.credits}, Custo: {cost}")
            return False

        try:
            # Esta seção é uma transação atômica
            user.credits -= cost
            transaction = CreditTransaction(
                user_id=user_id,
                agent_id=agent_id,
                amount=-cost,
                model_used=model_name,
                description=f"Chat com agente via CEAF V3 ({input_tokens} in, {output_tokens} out)"
            )
            db_session.add(transaction)
            db_session.commit()
            logger.info(f"Debitado {cost} créditos do usuário {user_id}. Novo saldo: {user.credits}")
            return True
        except Exception as e:
            db_session.rollback()
            logger.error(f"Erro ao debitar créditos do usuário {user_id}: {e}", exc_info=True)
            return False

    # Executa a função de banco de dados síncrona em um thread separado,
    # liberando o event loop principal para lidar com outras requisições.
    success = await asyncio.to_thread(db_operations)

    return success