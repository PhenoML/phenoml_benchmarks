import json
import requests
import os
from dotenv import load_dotenv
import base64

# Load environment variables from secrets.env
load_dotenv('secrets.env')

# Constants
VALIDATOR_URL = "http://localhost:8103/fhir/R4/{resourceType}/$validate"
MEDPLUM_OAUTH_TOKEN_URL_ = "http://localhost:8103/oauth2/token"

LLM_APIS = {
    "lang2FHIR": "https://experiment.pheno.ml/lang2fhir/create",
    # "OpenAI": "https://api.openai.com/v1/completions",
    # "Claude": "https://api.anthropic.com/v1/messages",
    "Gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + os.getenv('GEMINI_API_KEY')
}

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
    token = get_phenoml_token()
    payload = {
        "version": "R4",
        "resource": resource,
        "text": input_text
    }
    
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
    return response.json()

# Function to Validate FHIR Resources
def validate_fhir(resource):
    """Validates a FHIR resource using Medplum's $validate operation."""
    resource_type = resource.get("resourceType", "Unknown")
    url = VALIDATOR_URL.format(resourceType=resource_type)
    
    headers = {
        "Authorization": f"Bearer {get_medplum_token()}",
        "Content-Type": "application/fhir+json"
    }
    
    response = requests.post(url, headers=headers, json=resource)
    return response.json()

# Function to Generate FHIR Resource using Different APIs
# def generate_fhir(resource, prompt, input_text, model):
#     """Generates a FHIR resource using the specified API."""
#     api_url = LLM_APIS[model]  # This is already a string, no need to index into it
#     phenoml_headers = {"Authorization": f"Bearer {get_phenoml_token()}"}

#     if model == "lang2FHIR":
#         response = call_phenoml_api(resource, input_text)
#     # elif model == "OpenAI":
#     #     response = requests.post(api_url, json={"prompt": input_text, "model": "gpt-4"}, headers=headers)
#     # elif model == "Claude":
#     #     response = requests.post(api_url, json={"prompt": input_text, "model": "claude-2"}, headers=headers)
#     elif model == "Gemini":
#         full_text = prompt + "\n\n" + input_text
#         response = requests.post(api_url, json={"contents": [{"parts": [{"text": full_text}]}]})
#     else:
#         raise ValueError("Unsupported model")
    
#     return response

def generate_fhir(resource, prompt, input_text, model):
    """Generates a FHIR resource using the specified API."""
    api_url = LLM_APIS[model]  # Retrieve API URL as a string
    phenoml_headers = {"Authorization": f"Bearer {get_phenoml_token()}"}

    if model == "lang2FHIR":
        response = call_phenoml_api(resource, input_text)  # This already returns JSON
    elif model == "Gemini":
        full_text = prompt + "\n\n" + input_text
        response = requests.post(api_url, json={"contents": [{"parts": [{"text": full_text}]}]})
        response.raise_for_status()  # Ensure we catch HTTP errors
        response = response.json()  # Convert Response object to JSON
    else:
        raise ValueError("Unsupported model")

    return response  # Always return a dictionary



def is_fhir_valid(validation_result):
    """Checks if FHIR resource validation has only error-level issues."""
    if "issue" not in validation_result:
        return True  # No issues = valid

    # Filter issues with severity "error"
    error_issues = [issue for issue in validation_result["issue"] if issue.get("severity") == "error"]

    return len(error_issues) == 0  # Valid if no errors


# Run Benchmark Tests
results = []
for test in test_cases:
    print(f"Running test case: {test['test_name']}")
    input_text = test["input_text"]
    target_resource = test["target_profile"]
    prompt = test["prompt"]
    expected_resource_type = test["expected_resource_type"]
    expected_codes = set(test["expected_codes"])
    
    for model in LLM_APIS.keys():
        # Generate FHIR resource
        generated_fhir = generate_fhir(target_resource, prompt, input_text, model)
        
        validation_result = validate_fhir(generated_fhir)
        is_valid = is_fhir_valid(validation_result) 
        
        # Check resource type
        correct_type = generated_fhir.get("resourceType", "Unknown") == expected_resource_type
        
        # Check expected codes
        generated_codes = set(
            coding["code"] for coding in generated_fhir.get("code", {}).get("coding", [])
        )
        codes_match = expected_codes == generated_codes
        
        # Store results
        results.append({
            "test_name": test["test_name"],
            "model": model,
            "valid_fhir": is_valid,
            "correct_resource_type": correct_type,
            "codes_match": codes_match,
            "validation_issues": validation_result.get("issue", [])
        })

# Save Results to JSON
with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Print Summary
for result in results:
    print(f"Test: {result['test_name']}, Model: {result['model']}")
    print(f" - Valid FHIR: {result['valid_fhir']}")
    print(f" - Correct Resource Type: {result['correct_resource_type']}")
    print(f" - Codes Match: {result['codes_match']}")
    if not result["valid_fhir"]:
        print(f" - Validation Issues: {result['validation_issues']}")
    print()
