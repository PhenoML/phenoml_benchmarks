import json
import requests
import os
from dotenv import load_dotenv
import base64
from openai import OpenAI
from anthropic import Anthropic
import time
from google import genai

# Load environment variables from secrets.env
load_dotenv('secrets.env')

# Constants
VALIDATOR_URL = "https://validator.fhir.org/validate"
PHENOML_API_URL = "https://experiment.pheno.ml" # If you're on Experiment plan use this otherwise use your own PhenoML instance url
LLM_APIs = ["lang2FHIR", "OpenAI", "Anthropic", "Gemini"]

## Helper functions to call APIs

# PhenoML lang2FHIR API
def get_phenoml_token():
    credentials = base64.b64encode(
        f"{os.getenv('PHENOML_USERNAME')}:{os.getenv('PHENOML_PASSWORD')}"
        .encode()
    ).decode()
    
    response = requests.post(
        f"{PHENOML_API_URL}/auth/token",
        headers={
            'Accept': 'application/json',
            'Authorization': f'Basic {credentials}'
        }
    )
    response.raise_for_status()
    return response.json()['token']

def call_phenoml_api(resource, input_text):
    """Prepares payload and calls PhenoML lang2FHIRAPI to generate FHIR resource."""
    start_time = time.time()
    token = get_phenoml_token()
    payload = {
        "version": "R4",
        "resource": resource,
        "text": input_text
    }
    try:
        response = requests.post(
            f"{PHENOML_API_URL}/lang2fhir/create",
            headers={
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            json=payload
        )
        response.raise_for_status()
        result = response.json()
    finally:
        end_time = time.time()
    
    return result, (end_time - start_time)

# OpenAI API
def call_openai_api(prompt, input_text):
    """Calls OpenAI API to generate FHIR resource."""
    start_time = time.time()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    json_prompt = f"{prompt}\n\nPlease respond with a valid FHIR resource in JSON format.\n\n{input_text}"
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": json_prompt}
            ]
        )
        response_text = completion.choices[0].message.content
        result = json.loads(response_text)
    except (json.JSONDecodeError, Exception) as e:
        print(f"Failed to parse JSON from OpenAI response: {e}")
        result = {"resourceType": "Unknown"}
    finally:
        end_time = time.time()
    
    return result, (end_time - start_time)

# Anthropic API
def call_anthropic_api(prompt, input_text):
    """Calls Anthropic API to generate FHIR resource."""
    start_time = time.time()
    client = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
    json_prompt = f"{prompt}\n\nPlease respond with a valid FHIR resource in JSON format.\n\n{input_text}"
    try:
        completion = client.messages.create(
            max_tokens=4096,
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": json_prompt}]
        )
        response_text = completion.content[0].text
        result = json.loads(response_text)
    except (json.JSONDecodeError, AttributeError, IndexError) as e:
        print(f"Failed to parse JSON from Claude response: {e}")
        result = {"resourceType": "Unknown"}
    finally:
        end_time = time.time()
    
    return result, (end_time - start_time)

# Gemini API
def call_gemini_api(prompt, input_text):
    """Calls Gemini API to generate FHIR resource."""
    start_time = time.time()
    full_text = prompt + "\n\n" + input_text
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=full_text,
            config={
                'response_mime_type': 'application/json',
            },
        )
        response_text = response.text
        result = json.loads(response_text)
    except (KeyError, json.JSONDecodeError, IndexError) as e:
        print(f"Failed to parse Gemini response: {e}")
        result = {"resourceType": "Unknown"}
    finally:
        end_time = time.time()
    
    return result, (end_time - start_time)

# Generate FHIR resource using the specified API
def generate_fhir(resource, prompt, input_text, api):
    """Generates a FHIR resource using the specified API."""
    api_functions = {
        "lang2FHIR": call_phenoml_api,
        "Gemini": call_gemini_api,
        "OpenAI": call_openai_api,
        "Anthropic": call_anthropic_api
    }
    DEFAULT_RESPONSE = {"resourceType": "Unknown"}

    try:
        if api not in api_functions:
            raise ValueError("Unsupported API")
        api_func = api_functions[api]
        args = (resource, input_text) if api == "lang2FHIR" else (prompt, input_text)
        response, latency = api_func(*args)

        # Normalize response to dictionary
        if isinstance(response, list):
            response = response[0] if response and isinstance(response[0], dict) else DEFAULT_RESPONSE
        elif not isinstance(response, dict):
            print(f"Warning: Unexpected response type from {api}: {type(response)}")
            response = DEFAULT_RESPONSE

        # Save the generated resource to a file
        output_path = os.path.join(OUTPUT_DIR, api, f"{test['test_name']}.json")
        with open(output_path, 'w') as f:
            json.dump(response, f, indent=2)

        return response, latency

    except Exception as e:
        print(f"Error generating FHIR resource with {api}: {str(e)}")
        return DEFAULT_RESPONSE, 0

# Validate FHIR Resource and extract codes to validate against acceptable codes
def extract_codes_from_resource(resource):
    """Extracts codes from 'code' field and any field ending with 'CodeableConcept' in the resource."""
    codes = set()
    
    if not isinstance(resource, dict):
        print(f"Warning: Resource is not a dictionary: {type(resource)}")
        return codes
    
    # Check for 'code' field
    if "code" in resource and isinstance(resource["code"], dict):
        codings = resource["code"].get("coding", [])
        codes.update(coding["code"] for coding in codings if isinstance(coding, dict) and "code" in coding)
    
    # Check for any field ending with 'CodeableConcept'
    for field, value in resource.items():
        if field.endswith("CodeableConcept") and isinstance(value, dict):
            codings = value.get("coding", [])
            codes.update(coding["code"] for coding in codings if isinstance(coding, dict) and "code" in coding)
    
    return codes

def validate_fhir(resource):
    """Validates a FHIR resource using the public FHIR validator service."""
    if not isinstance(resource, dict):
        print(f"Warning: Cannot validate non-dictionary resource: {type(resource)}")
        return {
            "issue": [{
                "severity": "error",
                "code": "processing",
                "diagnostics": f"Invalid resource format: {type(resource)}"
            }]
        }
    
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    try:
        # Prepare the validation payload
        validation_request = {
            "cliContext": {
                "sv": "4.0.1",
                "ig": [
                    "hl7.fhir.us.core#4.0.1"
                ],
                "locale": "en"
            },
            "filesToValidate": [
                {
                    "fileName": "resource_to_validate.json",
                    "fileContent": json.dumps(resource),
                    "fileType": "json"
                }
            ],
            #TODO: Add proper sessionId
            "sessionId": "validation-session"
        }
        
        response = requests.post(
            VALIDATOR_URL,
            headers=headers,
            json=validation_request,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"Validation request failed: {str(e)}")
        return {
            "issue": [{
                "severity": "error",
                "code": "processing",
                "diagnostics": f"Validation service error: {str(e)}"
            }]
        }

def is_fhir_valid(validation_result):
    """Checks if FHIR resource validation has only information/warning level issues."""
    if "issue" not in validation_result:
        return True  # No issues = valid

    # Filter issues by severity- only include errors and fatal issues
    error_issues = [
        issue for issue in validation_result["issue"] 
        if issue.get("severity") in ["error", "fatal"]
    ]

    # Add detailed logging for validation issues
    if error_issues:
        print("\nValidation errors found:")
        for issue in error_issues:
            print(f"- Severity: {issue.get('severity')}")
            print(f"  Code: {issue.get('code')}")
            print(f"  Details: {issue.get('diagnostics', 'No details provided')}")
            print(f"  Location: {issue.get('location', ['No location'])}")
            print()

    return len(error_issues) == 0

def codes_match_acceptable(generated_codes, acceptable_codes):
    """
    Checks if generated codes are valid against acceptable codes list.
    Returns True if:
    - All generated codes are in the acceptable codes list
    """
    generated_codes = set(generated_codes)
    acceptable_codes = set(acceptable_codes)
    
    # If there are acceptable codes but nothing was generated, it's a fail
    if acceptable_codes and not generated_codes:
        return False
        
    # If codes were generated, they must all be in the acceptable list
    return generated_codes.issubset(acceptable_codes)

# Update the results storage to include more validation details
def store_validation_result(results, test, api, generated_fhir, validation_result, 
                          is_valid, correct_type, codes_match, expected_codes, 
                          generated_codes, latency):
    """Helper function to store validation results with detailed information."""
    
    # Extract error and fatal validation issues
    error_issues = [
        issue for issue in validation_result.get("issue", [])
        if issue.get("severity", "unknown") in ["error", "fatal"]
    ]

    results.append({
        "test_name": test["test_name"],
        "api": api,
        "valid_fhir": is_valid,
        "correct_resource_type": correct_type,
        "codes_match": codes_match,
        "expected_codes": list(expected_codes),
        "generated_codes": list(generated_codes),
        "latency": latency,
        "validation_details": {
            "total_issues": len(validation_result.get("issue", [])),
            "error_count": len(error_issues),
            "issues": error_issues
        },
        "output_file": os.path.join(OUTPUT_DIR, api, f"{test['test_name']}.json"),
        "us_core_profile": f"http://hl7.org/fhir/us/core/StructureDefinition/us-core-{generated_fhir.get('resourceType', '').lower()}"
    })

## Benchmarking Setup

# Create output directories for each API
OUTPUT_DIR = "generated_resources"
for api in LLM_APIs:
    api_dir = os.path.join(OUTPUT_DIR, api)
    os.makedirs(api_dir, exist_ok=True)

# Load Test Cases from JSON
with open("tests.json", "r") as f:
    test_cases = json.load(f)


## Run Benchmarks
results = []
for test in test_cases:
    print(f"\nRunning test case: {test['test_name']}")
    try:
        input_text = test.get("input_text", "")
        target_resource = test.get("target_profile", "")
        prompt = test.get("prompt", "")
        expected_resource_type = test.get("expected_resource_type", "")
        acceptable_codes = test.get("acceptable_codes", [])
        
        for api in LLM_APIs:
            try:
                print(f"Testing API: {api}")
                generated_fhir, latency = generate_fhir(target_resource, prompt, input_text, api)
                
                validation_result = validate_fhir(generated_fhir)
                is_valid = is_fhir_valid(validation_result)
                
                correct_type = generated_fhir.get("resourceType", "Unknown") == expected_resource_type
                generated_codes = extract_codes_from_resource(generated_fhir)
                codes_match = codes_match_acceptable(generated_codes, acceptable_codes)
                
                store_validation_result(
                    results, test, api, generated_fhir, validation_result,
                    is_valid, correct_type, codes_match, acceptable_codes, 
                    generated_codes, latency
                )
            except Exception as e:
                print(f"Error processing API {api}: {str(e)}")
                # Store error result
                store_validation_result(
                    results, test, api, {"resourceType": "Unknown"}, 
                    {"issue": [{"severity": "error", "diagnostics": str(e)}]},
                    False, False, False, acceptable_codes, set(), 0
                )
    except Exception as e:
        print(f"Error processing test case {test.get('test_name', 'unknown')}: {str(e)}")

# Save Results to JSON
with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Print Summary
for result in results:
    print(f"\nTest: {result['test_name']}, API: {result['api']}")
    print(f" - Valid FHIR: {result['valid_fhir']}")
    print(f" - Correct Resource Type: {result['correct_resource_type']}")
    print(f" - Codes Match: {result['codes_match']}")
    print(f" - Expected codes: {set(result['expected_codes'])}")
    print(f" - Generated codes: {set(result['generated_codes'])}")
    
    if not result["valid_fhir"]:
        print(f" - Validation Issues: {result['validation_details']}")
    print()
