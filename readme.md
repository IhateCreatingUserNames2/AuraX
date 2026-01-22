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
- NVIDIA GPU (optional, for soul_engine)

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

This implementation builds on findings from:

- Theory of mind evaluation in LLMs vs. primate baselines (arXiv:2601.12410)
- Riemannian geometry for spatio-temporal graph networks (arXiv:2601.14115)

More References: 
https://arxiv.org/abs/2512.01797
https://www.arxiv.org/abs/2512.07092
https://arxiv.org/abs/2505.10779
https://www.arxiv.org/abs/2506.12224
https://arxiv.org/abs/1905.13049
https://arxiv.org/abs/2308.08708
https://arxiv.org/abs/2309.10063
https://arxiv.org/abs/2502.17420
https://arxiv.org/abs/2410.02536
2512.24880v1( optional ) 
2512.19466v1
2512.24601v1
https://arxiv.org/abs/2512.20605v2
2512.22431v1
https://arxiv.org/abs/2512.22199v1
2512.22568v1
2512.23412v1
2507.16003v3
2512.19135v1
2310.01405v4
2310.01405v4
2512.01797v2
2512.04469v1
https://arxiv.org/abs/2511.20639v1
https://arxiv.org/abs/2511.16043v1
https://arxiv.org/abs/2510.26745v1


Additional citations available in `/docs/references.md`

## License

[Specify license here]

## Contact

[Specify contact information]
