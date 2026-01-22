
AuraX is an architecture for AI agents that combines deterministic processing (structured code) with language models (LLMs).
Unlike common chatbots that only respond to the last prompt, this system has a "cognitive loop" managed via queues, vector long-term memory, and an "internal state" system (simulating fatigue, curiosity, and epistemic tension).
What does it do?
Geometric Perception: Before responding, the system mathematically calculates (via vectors) whether the user's message is new or repetitive.
Hybrid Memory: Uses Qdrant to store memories and retrieve them by semantic similarity.
Resilient Orchestration: Uses Temporal.io to manage the flow of thought. If the server crashes in the middle of a thought process, it resumes from where it left off.

Vector Steering (Optional): If running locally with a GPU, it can inject vectors directly into the model layers (via soul_engine.py) to alter the AI ​​behavior without using prompts.

Architecture
The system consists of 4 main parts that need to run simultaneously:
Infrastructure (Docker): Database (Postgres), Cache (Redis), Vector Memory (Qdrant), and Task Queue (Temporal).
API (FastAPI): Receives requests from the Frontend and manages users.
Worker (Temporal Worker): The "brain" that processes heavy tasks in the background.
Soul Engine (Optional): Local inference server for LLM models (e.g., Qwen) with vector injection.

Prerequisites
Python 3.10+
Docker Desktop (Required to run the infrastructure)
Git
Installation
1. Clone the repository
code
Bash
git clone https://github.com/your-username/auraceaf-v4.git
cd auraceaf-v4
2. Create the virtual environment and install dependencies
code
Bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
3. Configure environment variables
Create a .env file in the project root. Here's an example below:

code
Ini
# .env
DATABASE_URL=postgresql+asyncpg://ceaf_user:ceaf_pass@localhost:5433/ceaf_db
REDIS_URL=redis://localhost:6379/0
QDRANT_URL=http://localhost:6333
TEMPORAL_HOST=localhost:7233

# API Keys (If using cloud)
OPENROUTER_API_KEY=your-key-here

# Inference Configuration (local or cloud)
# Use 'openrouter' for external API or 'vastai' to run with the local soul_engine
INFERENCE_MODE=openrouter
VASTAI_ENDPOINT=http://localhost:1111
How to Run
You will need 3 (or 4) open terminals.

Step 1: Set up the Infrastructure (Docker)
The system depends on Postgres, Redis, Qdrant, and Temporal. Do not attempt to run Python without them.

code.Bash: `docker compose up -d postgres redis qdrant temporal temporal-ui`
Wait approximately 30 seconds for Temporal to initialize the database.

Step 2: Run the API (Terminal 1)
This is the server that serves the website and receives messages.

code.Bash: `python main_app.py`
Access the frontend at: http://localhost:8000
Step 3: Run the Worker (Terminal 2)
This script processes the AI's reasoning. Without it, the chat will be stuck on "Thinking...".

Step 4 (Optional): Run the Soul Engine (Terminal 3)
Only if you have an NVIDIA GPU and want to run the model locally with Vector Steering.

code Bash
python worker.py
