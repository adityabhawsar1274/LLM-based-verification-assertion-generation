# LLM-Based Verification Assertion Generation

This project provides a robust, end-to-end pipeline for automatically generating and formally verifying SystemVerilog Assertions (SVA) from natural language specifications using Large Language Models (LLMs) and SymbiYosys (sby).

## Overview

The verification pipeline works in two main phases:

1. **Specification Tree Building (`main1.py`)**: Parses natural language specifications, chunks them, embeds them, and builds a hierarchical similarity tree serialized as a JSON file. This structure allows efficient retrieval of relevant spec details based on verification queries.
2. **Assertion Generation & Verification (`main2.py`)**: Uses a natural language query, retrieves the most relevant specification chunks, and interfaces with an LLM (e.g., Llama 3) to generate SystemVerilog Assertions. The generated assertions are passed to a pre-flight checker (to catch forbidden operators/missing DUT instantiations) and then formally verified using Yosys/SymbiYosys. The process iterates, using formal counterexamples to automatically refine the assertions until they pass.

## Requirements

- **Python 3.8+**
- **SymbiYosys (`sby`)**: Installed and available in your `PATH`.
- **Yosys**: Used under the hood by `sby` for formal verification and Cone of Influence (COI) analysis.
- **Python Packages**: 
  - `requests`
  - `argparse`
  (Other common standard libraries are used: `json`, `re`, `subprocess`, `sys`, `os`, `tempfile`)

## Setup & Configuration

1. **API Key Setup**:
   The LLM generation depends on the Groq API. Open `config.py` and replace `<YOUR_GROQ_API_KEY>` with your actual Groq API key:
   ```python
   API_KEY = "your-actual-api-key-here"
   ```
   *(Note: Ensure you do not commit your real API key. The `.gitignore` is set up to ignore `API_list.txt` if you prefer to load keys externally.)*

2. **Adjust Models & Parameters**:
   In `config.py`, you can configure the LLM models (e.g., `GROQ_MODEL_LARGE`), maximum verification iterations (`MAX_ITERATIONS`), and SymbiYosys bounded model checking depth (`BMC_DEPTH`).

## Usage

### Phase 1: Building the Spec Tree
Use `main1.py` to parse your natural language specification and build the JSON tree.

```bash
python3 main1.py --spec-file <path_to_spec.txt> --output-tree spec_tree.json
```
If no `--spec-file` is provided, it uses a built-in sample specification for a divider module.

### Phase 2: Generating & Verifying Assertions
Use `main2.py` to run the LLM generation and formal verification loop against your RTL design.

```bash
python3 main2.py test_rtl.v --query "Generate an assertion to ensure that when valid_in is asserted and den equals zero, exc_flag is set exactly 2 clock cycles later" --tree-file spec_tree.json --output final_result.json
```

**What happens under the hood:**
- The pipeline parses the RTL to identify input/output ports.
- It prompts the LLM to generate a Yosys-compatible assertion.
- A hard pre-flight check validates the generated assertion syntax.
- SymbiYosys (`sby`) is run.
- If it `PASS`es, the pipeline exits successfully.
- If it `FAIL`s, the pipeline parses the VCD counterexample, analyzes signal coverage/COI, and feeds the gap analysis back to the LLM for a subsequent refinement iteration.

## Repository Structure

- `main1.py` / `main2.py`: Entry points for building the spec tree and running the verification pipeline.
- `spec_tree.py`: Logic for building the hierarchical similarity tree.
- `tree_retriever.py`: Retriever logic to find relevant chunks from the spec tree using cosine similarity.
- `assertion_generator.py`: LLM prompt construction and response parsing.
- `llm_client.py`: Wrapper for making requests to the Groq API.
- `formal_runner.py`: Orchestrates SymbiYosys generation, execution, and output parsing.
- `rtl_utils.py`: Extracts port data and runs Yosys cone-of-influence analysis.
- `result_builder.py`: Formats the results, errors, and counterexamples into structured dictionaries.
- `config.py`: Central configuration for the pipeline (API keys, model choices, etc.).
