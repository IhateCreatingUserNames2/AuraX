# AuraX

AuraX is a neuro-symbolic architecture for AI agents that addresses limitations in theory of mind and temporal reasoning found in standard language models. The system implements geometric state representation and persistent memory through vector databases, enabling coherent perspective-taking and continuous temporal dynamics.

## Architecture Overview

The architecture separates knowledge states using geometric constraints in vector space, preventing information leakage between agent perspective and external context. This structural approach solves the perspective-taking failures documented in comparative cognition studies (chimpanzee baseline tasks).

### Core Components

**Geometric State Engine**: Calculates epistemic tension using vector distances in embedding space. State transitions are modeled on manifolds rather than Euclidean space, allowing continuous evolution of internal parameters (curiosity, fatigue, coherence metrics).

**Memory System**: Qdrant vector database implements retrieval-based knowledge access. The agent's knowledge is constrained to retrieved context, preventing omniscient behavior that violates theory of mind requirements.

**Temporal Dynamics**: Redis-based exponential decay functions replace discrete time steps. Memory strength and activation energy degrade continuously, implementing liquid time-constant networks without requiring recurrent architectures.

**Workflow Orchestration**: Temporal.io manages cognitive loops with checkpoint recovery. Processing failures resume from last committed state rather than restarting.

**Vector Steering (Optional)**: Direct layer-wise vector injection for model behavior modification during inference. Requires GPU deployment with supported model formats.

## System Requirements

- Python 3.10+
- Docker Desktop
- 8GB RAM minimum (16GB recommended)
- NVIDIA GPU (optional, for soul_engine Vector Steering) or Vast.AI 

## Installation

```bash
git clone https://github.com/your-username/aurax.git
cd aurax

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create `.env` file in project root:

```ini
# Database connections
DATABASE_URL=postgresql+asyncpg://ceaf_user:ceaf_pass@localhost:5433/ceaf_db
REDIS_URL=redis://localhost:6379/0
QDRANT_URL=http://localhost:6333
TEMPORAL_HOST=localhost:7233

# Inference mode: 'openrouter' or 'vastai'
INFERENCE_MODE=openrouter
OPENROUTER_API_KEY=your-key-here
VASTAI_ENDPOINT=http://localhost:1111
```

## Deployment

### 1. Infrastructure Services

```bash
docker compose up -d postgres redis qdrant temporal temporal-ui
```

Wait 30 seconds for Temporal initialization before proceeding.

### 2. API Server

Terminal 1:
```bash
python main_app.py
```

Frontend accessible at `http://localhost:8000`

### 3. Worker Process

Terminal 2:
```bash
python worker.py
```

Required for cognitive loop processing. Without active worker, requests will timeout.

### 4. Soul Engine (Optional)

Terminal 3 (GPU deployment only):
```bash
python soul_engine.py
```

Enables local inference with layer-wise vector manipulation. Set `INFERENCE_MODE=vastai` in `.env`.

## Architecture Details

**Simulated Ignorance**: Memory-based scoping prevents context contamination. Agent responses are constrained to retrieved vectors, enforcing knowledge boundaries required for perspective-taking tasks.

**Geometric Tension Calculation**: PCA-reduced state vectors measure deviation from homeostatic equilibrium. Tension magnitude determines response urgency and retrieval strategy.

**Continuous Time Evolution**: State parameters decay exponentially between interactions via Redis TTL and scheduled updates. Implements liquid network dynamics without RNN overhead.

**Checkpoint Recovery**: Temporal.io workflow persistence enables mid-process recovery. Cognitive loops resume from last committed decision point after crashes.

## References

This implementation builds on findings from, but, some i reached by myself, so they are here to validate AuraX. 

- Theory of mind evaluation in LLMs vs. primate baselines (arXiv:2601.12410)
- Riemannian geometry for spatio-temporal graph networks (arXiv:2601.14115)

More References: 

* **2512.01797** - H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs
* **2512.07092** - The Geometry of Persona: Disentangling Personality from Reasoning in Large Language Models
* **2505.10779** - Qualia Optimization
* **2506.12224** - Mapping Neural Theories of Consciousness onto the Common Model of Cognition
* **1905.13049** - Neural Consciousness Flow
* **2308.08708** - Consciousness in Artificial Intelligence: Insights from the Science of Consciousness
* **2309.10063** - Survey of Consciousness Theory from Computational Perspective
* **2502.17420** - The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence
* **2410.02536** - Intelligence at the Edge of Chaos
* **2512.24880** - mHC: Manifold-Constrained Hyper-Connections
* **2512.19466** - Epistemological Fault Lines Between Human and Artificial Intelligence
* **2512.24601** - Recursive Language Models
* **2512.20605** - Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning
* **2512.22431** - Monadic Context Engineering
* **2512.22199** - Bidirectional RAG: Safe Self-Improving Retrieval-Augmented Generation Through Multi-Stage Validation
* **2512.22568** - Lessons from Neuroscience for AI: How integrating Actions, Compositional Structure and Episodic Memory could enable Safe, Interpretable and Human-Like AI
* **2512.23412** - MindWatcher: Toward Smarter Multimodal Tool-Integrated Reasoning
* **2507.16003** - Learning without training: The implicit dynamics of in-context learning
* **2512.19135** - Understanding Chain-of-Thought in Large Language Models via Topological Data Analysis
* **2310.01405** - Representation Engineering: A Top-Down Approach to AI Transparency
* **2512.04469** - Mathematical Framing for Different Agent Strategies
* **2511.20639** - Latent Collaboration in Multi-Agent Systems
* **2511.16043** - Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning
* **2510.26745** - Deep sequence models tend to memorize geometrically; it is unclear why



## Frontend

it has a basic FrontEND. 
Check /frontend folder 
<img width="1917" height="918" alt="image" src="https://github.com/user-attachments/assets/7bd7f8c4-d6a8-49b3-bce6-232a813a833a" />


## License

MIT


## Considerations

It has bugs, 
It was not totally tested because i do not have good hardware nor budget. 
The requeriments file may be missing libs, 


