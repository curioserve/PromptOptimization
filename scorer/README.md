# EXPO Scorer (Prompt Utility MLP)

A tiny, single-hidden-layer MLP used to score prompts for bandit-style prompt optimization.

- **EXPO**: scores (Task Description ⊕ Meta-Instruction) embeddings  
  Input dim: 6144 (2 × 3072), Hidden: 1536, Output: scalar
- **EXPO-ES**: scores exemplar embeddings  
  Input dim: 3072, Hidden: 512, Output: scalar

