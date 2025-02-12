import json
import boto3

# Initialize Bedrock Runtime client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

modelId = 'cohere.command-text-v14'  # Cohere model

def lambda_handler(event, context):
    try:
        print('Event Received:', json.dumps(event))

        # Ensure event['body'] exists and parse it safely
        request_body = json.loads(event.get('body', '{}'))
        prompt = request_body.get('prompt', '').strip()

        # Validate input
        if not prompt:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing or empty prompt"})
            }

        # Corrected request payload (Removed 'top_p' and 'stop_sequences')
        body = {
            "prompt": prompt,
            "max_tokens": 300,  # Cohere uses 'max_tokens'
            "temperature": 0.75,
            "k": 0  # Cohere uses 'k' instead of 'top_k'
        }

        # Invoke Amazon Bedrock model
        bedrockResponse = bedrock.invoke_model(
            modelId=modelId,
            body=json.dumps(body),
            accept='application/json',
            contentType='application/json'
        )

        # Read and decode response
        response_body = json.loads(bedrockResponse['body'].read().decode('utf-8'))
        response_text = response_body['generations'][0]['text']

        # Construct API response
        return {
            "statusCode": 200,
            "body": json.dumps({
                "prompt": prompt,
                "response": response_text
            })
        }

    except KeyError as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": f"Missing key: {str(e)}"})
        }

    except Exception as e:
        print(f"Internal Error: {str(e)}")  # Log error for debugging
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal Server Error", "details": str(e)})
        }
