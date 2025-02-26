import json
import requests
import os
from dotenv import load_dotenv
import base64
from openai import OpenAI
from anthropic import Anthropic
import time

# Load environment variables from secrets.env
load_dotenv('secrets.env')

# Constants
VALIDATOR_URL = "https://validator.fhir.org/validate"
MEDPLUM_OAUTH_TOKEN_URL_ = "http://localhost:8103/oauth2/token"

LLM_APIS = {
    "lang2FHIR": "https://experiment.pheno.ml/lang2fhir/create",
    "OpenAI": "https://api.openai.com/v1/completions",
    "Claude": "https://api.anthropic.com/v1/messages",
    "Gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + os.getenv('GEMINI_API_KEY')
}

# Create output directories for each API
OUTPUT_DIR = "generated_resources"
for api in LLM_APIS.keys():
    api_dir = os.path.join(OUTPUT_DIR, api)
    os.makedirs(api_dir, exist_ok=True)

# Load Test Cases from JSON
with open("tests.json", "r") as f:
    test_cases = json.load(f)

# Medplum auth
def get_medplum_token():
    response = requests.post(
        MEDPLUM_OAUTH_TOKEN_URL_,
        headers={'Content-Type': 'application/x-www-form-urlencoded'},
        data={
            'grant_type': 'client_credentials',
            'client_id': os.getenv('MEDPLUM_CLIENT_ID'),
            'client_secret': os.getenv('MEDPLUM_CLIENT_SECRET')
        }
    )
    response.raise_for_status()
    return response.json()['access_token']

# PhenoML auth
def get_phenoml_token():
    credentials = base64.b64encode(
        f"{os.getenv('PHENOML_USERNAME')}:{os.getenv('PHENOML_PASSWORD')}"
        .encode()
    ).decode()
    
    response = requests.post(
        'https://experiment.pheno.ml/auth/token',
        headers={
            'Accept': 'application/json',
            'Authorization': f'Basic {credentials}'
        }
    )
    response.raise_for_status()
    return response.json()['token']

# Example API calls
def call_medplum_api(endpoint, payload):
    token = get_medplum_token()
    response = requests.post(
        endpoint,
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        },
        json=payload
    )
    response.raise_for_status()
    return response.json()

def call_phenoml_api(resource, input_text):
    """Prepares payload and calls PhenoML API to generate FHIR resource."""
    start_time = time.time()
    token = get_phenoml_token()
    payload = {
        "version": "R4",
        "resource": resource,
        "text": input_text
    }
    
    try:
        response = requests.post(
            'https://experiment.pheno.ml/lang2fhir/create',
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

def call_openai_api(prompt, input_text):
    start_time = time.time()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    json_prompt = f"{prompt}\n\nPlease respond with a valid FHIR resource in JSON format.\n\n{input_text}"
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
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

def call_claude_api(prompt, input_text):
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

def call_gemini_api(prompt, input_text):
    """Calls Gemini API to generate FHIR resource."""
    start_time = time.time()
    full_text = prompt + "\n\n" + input_text
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={os.getenv('GEMINI_API_KEY')}",
            json={
                "contents": [{"parts": [{"text": full_text}]}],
                "generationConfig": { "response_mime_type": "application/json" }
            }
        )
        response.raise_for_status()
        response_json = response.json()
        
        # Get the text content from the first candidate's first part
        fhir_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(fhir_text)
    except (KeyError, json.JSONDecodeError, IndexError) as e:
        print(f"Failed to parse Gemini response: {e}")
        result = {"resourceType": "Unknown"}
    finally:
        end_time = time.time()
    
    return result, (end_time - start_time)

# Function to Validate FHIR Resources
def validate_fhir(resource):
    """Validates a FHIR resource using the public FHIR validator service."""
    # Handle non-dict resources
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

def generate_fhir(resource, prompt, input_text, model):
    """Generates a FHIR resource using the specified API."""
    try:
        if model == "lang2FHIR":
            response, latency = call_phenoml_api(resource, input_text)
        elif model == "Gemini":
            response, latency = call_gemini_api(prompt, input_text)
        elif model == "OpenAI":
            response, latency = call_openai_api(prompt, input_text)
        elif model == "Claude":
            response, latency = call_claude_api(prompt, input_text)
        else:
            raise ValueError("Unsupported model")

        # Ensure response is a dictionary
        if isinstance(response, list) and len(response) > 0:
            response = response[0] if isinstance(response[0], dict) else {"resourceType": "Unknown"}
        elif not isinstance(response, dict):
            print(f"Warning: Unexpected response type from {model}: {type(response)}")
            response = {"resourceType": "Unknown"}

        # Save the generated resource to a file
        output_path = os.path.join(OUTPUT_DIR, model, f"{test['test_name']}.json")
        with open(output_path, 'w') as f:
            json.dump(response, f, indent=2)

        return response, latency

    except Exception as e:
        print(f"Error generating FHIR resource with {model}: {str(e)}")
        return {"resourceType": "Unknown"}, 0

def extract_codes_from_resource(resource):
    """Extracts codes from 'code' field and any field ending with 'CodeableConcept' in the resource."""
    codes = set()
    
    # Handle non-dict resources
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

def is_fhir_valid(validation_result):
    """Checks if FHIR resource validation has only information/warning level issues."""
    if "issue" not in validation_result:
        return True  # No issues = valid

    # Filter issues by severity
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
    - At least one code is generated when acceptable codes exist
    """
    generated_codes = set(generated_codes)
    acceptable_codes = set(acceptable_codes)
    
    # If there are acceptable codes but nothing was generated, it's a fail
    if acceptable_codes and not generated_codes:
        return False
        
    # If codes were generated, they must all be in the acceptable list
    return generated_codes.issubset(acceptable_codes)

# Update the results storage to include more validation details
def store_validation_result(results, test, model, generated_fhir, validation_result, 
                          is_valid, correct_type, codes_match, expected_codes, 
                          generated_codes, latency):
    """Helper function to store validation results with detailed information."""
    
    # Extract validation issues by severity
    validation_issues = {
        "errors": [],
        "warnings": [],
        "information": []
    }
    
    for issue in validation_result.get("issue", []):
        severity = issue.get("severity", "unknown")
        if severity in ["error", "fatal"]:
            validation_issues["errors"].append(issue)
        elif severity == "warning":
            validation_issues["warnings"].append(issue)
        elif severity == "information":
            validation_issues["information"].append(issue)

    results.append({
        "test_name": test["test_name"],
        "model": model,
        "valid_fhir": is_valid,
        "correct_resource_type": correct_type,
        "codes_match": codes_match,
        "expected_codes": list(expected_codes),
        "generated_codes": list(generated_codes),
        "latency": latency,
        "validation_details": {
            "total_issues": len(validation_result.get("issue", [])),
            "error_count": len(validation_issues["errors"]),
            "warning_count": len(validation_issues["warnings"]),
            "info_count": len(validation_issues["information"]),
            "issues": validation_issues
        },
        "output_file": os.path.join(OUTPUT_DIR, model, f"{test['test_name']}.json"),
        "us_core_profile": f"http://hl7.org/fhir/us/core/StructureDefinition/us-core-{generated_fhir.get('resourceType', '').lower()}"
    })

# Run Benchmark Tests
results = []
for test in test_cases:
    print(f"\nRunning test case: {test['test_name']}")
    try:
        input_text = test.get("input_text", "")
        target_resource = test.get("target_profile", "")
        prompt = test.get("prompt", "")
        expected_resource_type = test.get("expected_resource_type", "")
        acceptable_codes = test.get("acceptable_codes", [])
        
        for model in LLM_APIS.keys():
            try:
                print(f"Testing model: {model}")
                generated_fhir, latency = generate_fhir(target_resource, prompt, input_text, model)
                
                validation_result = validate_fhir(generated_fhir)
                is_valid = is_fhir_valid(validation_result)
                
                correct_type = generated_fhir.get("resourceType", "Unknown") == expected_resource_type
                generated_codes = extract_codes_from_resource(generated_fhir)
                codes_match = codes_match_acceptable(generated_codes, acceptable_codes)
                
                store_validation_result(
                    results, test, model, generated_fhir, validation_result,
                    is_valid, correct_type, codes_match, acceptable_codes, 
                    generated_codes, latency
                )
            except Exception as e:
                print(f"Error processing model {model}: {str(e)}")
                # Store error result
                store_validation_result(
                    results, test, model, {"resourceType": "Unknown"}, 
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
    print(f"\nTest: {result['test_name']}, Model: {result['model']}")
    print(f" - Valid FHIR: {result['valid_fhir']}")
    print(f" - Correct Resource Type: {result['correct_resource_type']}")
    print(f" - Codes Match: {result['codes_match']}")
    print(f" - Expected codes: {set(result['expected_codes'])}")
    print(f" - Generated codes: {set(result['generated_codes'])}")
    
    if not result["valid_fhir"]:
        print(f" - Validation Issues: {result['validation_details']}")
    print()
