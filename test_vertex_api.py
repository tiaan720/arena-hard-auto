import subprocess
import requests
import platform

# Test the vertex API call
def test_vertex_api():
    project_id = "research-su-llm-routing"
    region = "us-central1" 
    model = "gemini-2.5-flash"
    
    # Get access token
    gcloud_cmd = "gcloud.cmd" if platform.system() == "Windows" else "gcloud"
    access_token = subprocess.check_output(
        [gcloud_cmd, "auth", "application-default", "print-access-token"], 
        text=True
    ).strip()
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    
    # Simple test message
    data = {
        "contents": [
            {
                "parts": [{"text": "Say hello"}],
                "role": "user"
            }
        ]
    }
    
    url = (
        f"https://{region}-aiplatform.googleapis.com/v1/projects/"
        f"{project_id}/locations/{region}/publishers/google/models/"
        f"{model}:generateContent"
    )
    
    print(f"Making request to: {url}")
    print("Request data:", data)
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        print(f"Response status: {response.status_code}")
        print(f"Response: {response.text}")
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    test_vertex_api()
