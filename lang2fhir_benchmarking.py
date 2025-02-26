import json
import requests
import os
from dotenv import load_dotenv
import base64
from openai import OpenAI
from anthropic import Anthropic

# Load environment variables from secrets.env
load_dotenv('secrets.env')

# Constants
VALIDATOR_URL = "http://localhost:8103/fhir/R4/{resourceType}/$validate"
MEDPLUM_OAUTH_TOKEN_URL_ = "http://localhost:8103/oauth2/token"

LLM_APIS = {
    "lang2FHIR": "https://experiment.pheno.ml/lang2fhir/create",
    "OpenAI": "https://api.openai.com/v1/completions",
    "Claude": "https://api.anthropic.com/v1/messages",
    "Gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + os.getenv('GEMINI_API_KEY')
}

# Load Test Cases from JSON
with open("tests01.json", "r") as f:
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

def call_openai_api(prompt, input_text):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    # Add explicit JSON request to the prompt
    json_prompt = f"{prompt}\n\nPlease respond with a valid FHIR resource in JSON format.\n\n{input_text}"
    completion = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": json_prompt}
        ]
    )
    response_text = completion.choices[0].message.content
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from OpenAI response: {response_text}")
        return {"resourceType": "Unknown"}


def call_claude_api(prompt, input_text):
    client = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
    json_prompt = f"{prompt}\n\nPlease respond with a valid FHIR resource in JSON format.\n\n{input_text}"
    completion = client.messages.create(
        max_tokens=4096,
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": json_prompt}]
    )
    try:
        # Extract the text content and parse it as JSON
        response_text = completion.content[0].text
        return json.loads(response_text)
    except (json.JSONDecodeError, AttributeError, IndexError) as e:
        print(f"Failed to parse JSON from Claude response: {e}")
        return {"resourceType": "Unknown"}

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

def generate_fhir(resource, prompt, input_text, model):
    """Generates a FHIR resource using the specified API."""
    api_url = LLM_APIS[model]  # Retrieve API URL as a string

    if model == "lang2FHIR":
        response = call_phenoml_api(resource, input_text)  # This already returns JSON
    elif model == "Gemini":
        full_text = prompt + "\n\n" + input_text
        response = requests.post(api_url, json={"contents": [{"parts": [{"text": full_text}]}],"generationConfig": { "response_mime_type": "application/json" }})
        response.raise_for_status()  # Ensure we catch HTTP errors
        response = response.json()  # Convert Response object to JSON
    elif model == "OpenAI":
        response = call_openai_api(prompt, input_text)
    elif model == "Claude":
        response = call_claude_api(prompt, input_text)
    else:
        raise ValueError("Unsupported model")

    return response  # Always return a dictionary

def extract_codes_from_resource(resource):
    """Extracts codes from 'code' field and any field ending with 'CodeableConcept' in the resource."""
    codes = set()
    
    # Check for 'code' field
    if "code" in resource and isinstance(resource["code"], dict):
        codings = resource["code"].get("coding", [])
        codes.update(coding["code"] for coding in codings if "code" in coding)
    
    # Check for any field ending with 'CodeableConcept'
    for field, value in resource.items():
        if field.endswith("CodeableConcept") and isinstance(value, dict):
            codings = value.get("coding", [])
            codes.update(coding["code"] for coding in codings if "code" in coding)
    
    return codes

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
        generated_codes = extract_codes_from_resource(generated_fhir)
        codes_match = expected_codes == generated_codes
        
        # Store results
        results.append({
            "test_name": test["test_name"],
            "model": model,
            "valid_fhir": is_valid,
            "correct_resource_type": correct_type,
            "codes_match": codes_match,
            "expected_codes": list(expected_codes),  # Convert set to list for JSON serialization
            "generated_codes": list(generated_codes),  # Convert set to list for JSON serialization
            "validation_issues": validation_result.get("issue", [])
        })

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
        print(f" - Validation Issues: {result['validation_issues']}")
    print()
