# FHIR Benchmarking

This repository contains the code for benchmarking the FHIR generation capabilities of various LLMs and PhenoML lang2FHIR.

## Usage

The benchmarking script can be used to benchmark the FHIR generation capabilities of various LLMs and PhenoML lang2FHIR. Test cases are defined in the `tests.json` file. Each test case contains a description of the test, the expected FHIR resource type, and expected codes (as relevant). The benchmarking script utilizes the publically available [FHIR Validator API] (https://validator.fhir.org/) to validate the generated FHIR resources against US Core IG. Test cases are designed to evaluate accuracy of FHIR resource generation, codes generated, and overall latency of the API call. 

### Prerequisites

- Python 3.11+
- pip
- pip install -r requirements.txt
- API keys and credentials for the LLMs and PhenoML lang2FHIR

### Running the script

```bash
python3 lang2fhir_benchmarking.py
```

## Results

Benchmarking results are saved in the `benchmark_results.json` file. The `benchmark_analysis.ipynb` file contains the code used to analyze the results with plots illustrating success rates and latency by API. In the current analysis, we see that while all APIs generate valid FHIR as evaluated by the public FHIR Validator, PhenoML lang2FHIR model outperforms direct usage of major commercial AI APIs on code matching (100% success rate on the 18 test cases compared to approx 40-60% success rate for commercial APIs) and outperforms Anthropic and OpenAI on latency. While commmercial AI APIs can generate accurate codes for some test cases, for less common codes, code hallucination is more probable. Lang2FHIR currently utilizes Gemini as an LLM and we are now extending it to support Private LLM usage to enable completely private FHIR generation. 

## Future Work

- Include more test cases and more comprehensive analysis 
- Benchmark lang2FHIR performance with private LLM usage
- Evaluate lang2FHIR performance as part of an agentic workflow against clinical tasks from benchmarks such as [MedAgentBench](https://github.com/MedAgentBench/MedAgentBench) and [MedHELM](https://crfm.stanford.edu/helm/medhelm/latest/)
- Include more LLM models and APIs in benchmarking



