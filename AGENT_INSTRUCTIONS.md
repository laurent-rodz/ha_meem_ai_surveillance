# HA-MEEM AI Surveillance: Agent Context & Specification

> **Primary Goal:** Reliable Authorized vs. Unknown classification at the factory entry gate.
> **Strict Enforcement:** All structural, architectural, or foundational changes require explicit human authorization. 

## 1. Project Objective
Build a production-grade, real-time face recognition (FR) system tailored for an active factory environment using entry-zone CCTV.
* **Proof of Concept (PoC):** 1–2 cameras
* **Production Target:** 20 cameras

---

## 2. Agent Permissions & Boundaries

### The "Green Zone" (Permitted Actions)
* **Implementation:** Create single-purpose modules/components (max 200 LOC).
* **Refactoring:** Localized cleanup within a single file to improve readability or performance.
* **Bug Fixes:** Patching logic errors in existing functions.
* **Testing:** Must write or update unit tests for every change made.

### The "Red Zone" (Prohibited Actions)
* **Implicit Deny:** If an action is not explicitly listed in the "Green Zone," it is prohibited.
* **Structural:** Do not modify folder hierarchy, rename directories, or move files between subsystems.
* **Dependencies:** Do not add, remove, or version-bump entries in `package.json`, `requirements.txt`, or equivalent.
* **Logic Constraints:** No training/fine-tuning code, no ClearML/WandB telemetry integration, and no new external API or network-bound logic.
* **Frameworks:** Do not introduce new UI libraries, state management patterns, or database schemas.

### Change Protocol
* **Requesting Changes:** If a task requires a "Red Zone" action, the agent must stop and present a "Change Request" summary for human approval.

---

## 3. Operational Constraints

* **Subject Behavior:** Workers will not intentionally pause or look at the camera.
* **Camera Positioning:** Tilted at 15°–25°.
* **Image Quality:** Real-world blur, motion, and partial occlusions are expected.
* **Resolution Gate:** Faces must be **≥80px in width** to qualify for recognition; reject faces below this threshold.
* **Quality Gate:** Reject excessively blurry frames.
* **Inference Rule:** Single-frame recognition is strictly forbidden. **Multi-frame fusion is mandatory.**

---

## 4. Technical Stack

| Category | Technology |
| :--- | :--- |
| **OS** | Windows |
| **Framework** | ONNX Runtime (GPU) + TensorRT |
| **Optimization** | TensorRT 10.12 (FP16) |
| **CUDA** | Version 12.6 |
| **Experiment Tracking** | ClearML *(Note: Agent must not write this integration per Red Zone rules)* |
| **Version Control** | GitHub |
| **IDE** | Google Antigravity (Agentic AI IDE) |

---

## 5. System Design Principles

### Architecture & Structure
* **Modular Architecture:** Design small, single-responsibility classes. No monolithic scripts.
* **Strict Separation of Concerns:** Use `core/` for pure Computer Vision logic, `apps/` for runtime pipelines/wrappers, and `experiments/` for research/training.
* **Config-Driven:** No hardcoded thresholds or paths; all parameters must be loaded via configuration files.
* **Stability First:** Production stability takes strict precedence over academic novelty.
* **State Mutation:** Strictly avoid hidden or global state mutations.

### Technical Anti-Patterns (Do NOT Use)
* No GAN-based frontalization.
* No super-resolution applied to face recognition crops.
* No heavy transformer architectures (unless strictly justified by measured failure of lighter models).

---

## 6. Recognition Pipeline (PoC) & Integration

* **Face Detection:** SCRFD
* **Recognition Model:** AdaFace (pretrained baseline, 512-dimensional normalized embedding output)
* **Matching Metric:** Cosine similarity
* **Decision Logic:** Track-level aggregated embedding (Multi-frame fusion)
* **Decision Threshold:** Configurable via settings
* **Agnostic Core:** The CV `core/` must remain completely framework-agnostic.
* **Strict Integration Rule:** **Do NOT** introduce API, networking, or database dependencies into the `core/` directory.

---

## 7. Code Rules & Quality

* **Typing:** Enforce strict Python type hints across all methods.
* **Configuration:** Use config files for *all* thresholds, variables, and paths.
* **Paths & Weights:** Never hardcode model paths; never commit `.onnx`, `.engine`, or `.pt` weights to version control.
* **Quality:** Write highly testable components.

---

## 8. Definition of Done (PoC)

| Metric / Feature | Target Requirement |
| :--- | :--- |
| **Multi-frame Fusion** | Stable, tested, and integrated |
| **False Acceptance Rate (FAR)** | `< 2%` |
| **False Rejection Rate (FRR)** | `< 5%` |
| **Speed** | Sustained real-time performance |
| **Observability** | Clean, structured logs generated per entry event |