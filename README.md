# FHIR Benchmarking

This repository contains the code for benchmarking the FHIR generation capabilities of various LLMs and PhenoML lang2FHIR.

## Usage

This benchmarking script can be used to benchmark the FHIR generation capabilities of various LLMs and PhenoML lang2FHIR. Test cases are defined in the `tests.json` file. Each test case contains a description of the test, the expected FHIR resource type, and expected codes (as relevant). The benchmarking script utilizes the publically available [FHIR Validator API](https://validator.fhir.org/) to validate the generated FHIR resources against US Core IG. Test cases are designed to evaluate accuracy of FHIR resource generation, codes generated, and overall latency of the API call. 

### Prerequisites

- Python 3.11+
- pip
- pip install -r requirements.txt
- API keys and credentials for the LLMs and PhenoML lang2FHIR (sign up for lang2FHIR API access [here](https://developer.pheno.ml))

### Running the script

```bash
python3 lang2fhir_benchmarking.py
```

## Results

Benchmarking results are saved in the `benchmark_results.json` file. The `benchmark_analysis.ipynb` file contains the code used to analyze the results with plots illustrating success rates and latency by API. Anthropic's Claude 3.5 Sonnet v2, OpenAI's GPT-4o-mini, and Google's Gemini 2.0 Flash were evaluated with PhenoML lang2FHIR API to provide a comprehensive comparison from a latency and performance perspective.

In the current analysis, we see that while all APIs generate valid FHIR as evaluated by the public FHIR Validator, PhenoML lang2FHIR API outperforms direct usage of major commercial AI APIs on code matching (100% success rate on the 32 test cases compared to approx 30-60% success rate for commercial APIs) and outperforms Anthropic and OpenAI on latency.  

While direct usage of LLM APIs can generate accurate codes for some test cases, for less common codes, code hallucination is probable. Lang2FHIR currently utilizes Gemini 2.0 Flash as an LLM within the overall system and we are now extending lang2FHIR to support Private LLM usage (via Ollama); enabling completely private FHIR generation and language powered healthcare workflows. 

![Benchmarking Results](output.png)
Generation date: 2025-03-11 from `benchmark_analysis.ipynb` 

## Future Work

- Include more test cases and more comprehensive analysis 
- Benchmark lang2FHIR performance with private LLM usage
- Evaluate lang2FHIR performance as part of an agentic workflow against clinical tasks from benchmarks such as [MedAgentBench](https://github.com/MedAgentBench/MedAgentBench) and [MedHELM](https://crfm.stanford.edu/helm/medhelm/latest/) and FHIR tasks from eval frameworks such as [Flexpa's LLM FHIR Eval](https://github.com/flexpa/llm-fhir-eval)
- Include more LLM models and APIs in benchmarking

##  Additional Information

FHIR® is a registered trademark of HL7.



