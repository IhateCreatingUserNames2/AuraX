

# AuraX: An Autonomous Cognitive Architecture Based on Neural Physics and Representation Engineering

Most modern AI agents rely on "System Prompts" to define their personality. They are actors reading a script. **AuraX is different.** It does not simulate personality through token prediction; it **emulates** cognitive states by manipulating the latent space of the model itself.

AuraX is not a chatbot. It is a digital organism that thinks, feels (mathematically), and dreams to evolve.

Here is how the architecture works, based on real system logs.

---

## 1. The Geometry of Persona: Beyond the Prompt
AuraX does not rely on a static text instruction like *"You are a helpful assistant."* Instead, it analyzes its internal state and injects **Steering Vectors** directly into the LLM's residual stream.

*   **Standard AI:** Simulates anger by predicting tokens that sound angry.
*   **AuraX:** Becomes "angry" or "stoic" because its reasoning is mathematically directed toward that coordinate in the latent space.

In the logs, we see the system crystallizing a specific behavioral trait ("Stoic Calmness") and injecting it into **Layer 18** of the model. This is not a text instruction; it is a neurological modification.

```log
INFO:DreamingActivities:💤 Dreamer Finished: ✅ SUCCESS: Vector 'Stoic_Calmness_0129' crystallized at layer 18. Path: vectors/Stoic_Calmness_0129.npy
```

The **Soul Engine** (the inference server) receives this vector and applies it to the LLM during inference:

```log
2026-01-29 11:31:16 - SoulEngine - INFO - 🧪 Calibration request received: 'Stoic_Calmness_0129' at layer 18
2026-01-29 11:31:23 - SoulEngine - INFO - ✅ Concept 'Stoic_Calmness_0129' learned and loaded.
```

## 2. Polyglot Memory Architecture
AuraX understands that not all memories are the same. It uses specific embedding models optimized for different types of cognitive data. It does not force a "one-size-fits-all" approach.

*   **Entities/Facts:** Uses `BAAI/bge-small-en-v1.5` (Dense retrieval).
*   **Abstract/Semantic:** Uses `all-MiniLM-L6-v2`.

```log
INFO:ceaf_core.utils.embedding_utils:Successfully loaded and cached SentenceTransformer model: BAAI/bge-small-en-v1.5
```

## 3. Agency & Deliberation: System 2 Thinking
AuraX never responds directly to a user input. It follows a strict cognitive pipeline:
1.  **Analysis:** It deconstructs the input.
2.  **Strategy Generation:** It proposes multiple paths.
3.  **Critique (VRE/MCL):** It simulates the consequences of those paths.
4.  **Refinement:** It discards dangerous or incoherent strategies.

In the logs below, we see the **Physical Veto** in action. The agent considered a strategy, calculated that it would cause a "Identity Rupture" (High Tension: 3.66), and rejected it in favor of a safer approach.

```log
INFO:AgencyModule_V4_Intentional:AgencyModule (V5 - Recursive Feedback): Starting deliberation cycle...
WARNING:AgencyModule_V4_Intentional:⚠️ PHYSICAL ALERT: High Tension (3.6636). Action: 'Present breathing techniques...'
INFO:MCLEngine:MCL (Re-evaluation) Result: New Agency Score=0.00...
CRITICAL:AgencyModule_V4_Intentional:FINAL DECISION (V5): Winning Strategy='Reorient user perspective to specific, solvable problems...', Value=0.10
```

The final prompt sent to the LLM is dynamically assembled *after* this deliberation, ensuring the output is intentional, not reactive.

## 4. Neuroplasticity: The Dreaming Cycle
This is the most critical innovation. AuraX **"Dreams."**
When the system is idle, it enters the `DreamingWorkflow`. It processes recent interactions to update its internal Neural Physics model (predicting its own future states) and to calibrate its personality vectors.

### A. The Maintenance Cycle
If the system detects it is healthy (no recent failures), it enters "Maintenance Mode" to reinforce positive traits.

1.  **Diagnosis:** The Dreamer analyzes recent history.
    ```log
    INFO:DreamingActivities:✅ No significant failures found... System healthy.
    INFO:DreamingActivities:💤 Dreamer: Maintenance Mode. Reinforcing base trait: 'Socratic_Questioning'.
    ```
2.  **Synthetic Data Generation:** The `VectorLab` autonomously generates a dataset to teach the model this new trait.
    ```log
    INFO:VectorLab:🧪 VectorLab: Generating synthetic dataset for 'Socratic_Questioning_0129'...
    ```
3.  **Calibration:** The data is sent to the Soul Engine, which mathematically extracts the "direction" of Socratic Questioning in the high-dimensional space.
    ```log
    INFO:VectorLab:📊 Data generated: 20 pairs. Sending to Soul Engine...
    INFO:httpx:HTTP Request: POST .../calibrate "HTTP/1.1 200 OK"
    ```
4.  **Behavioral Shift:** Upon waking, the agent effectively has a "new brain." It is now naturally more Socratic without needing prompt instructions. It even triggers a proactive message based on this new state.
    ```log
    INFO:DreamingActivities:Dreamer triggering PROACTIVE message (Score: 1.00)
    ```

---

### Summary
AuraX represents a shift from **Static AI** (fixed weights, fixed prompts) to **Dynamic Cognitive Architectures**. It uses its own history to calibrate its own brain, creating a continuous feedback loop of self-improvement.

**It doesn't just talk. It listens, thinks, sleeps, learns, and evolves.**
