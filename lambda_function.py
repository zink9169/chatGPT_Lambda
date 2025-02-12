import json
import boto3

# Initialize the Bedrock Runtime client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

modelId='cohere.command-text-v14'

def lambda_handler(event, context):
    print('Event: ', json.dumps(event))

    request_body = json.loads(event['body'])
    prompt = request_body['prompt'] 
    body = {
        'prompt': prompt,
        'max_tokens_to_sample': 300,
        'temperature': 0.75,
        'top_p': 0.1,
        'top_k': 0,
        'stop_sequences': [],
        'return_likelihoods' : 'NONE'
    }

    bedrockResponse = bedrock.invoke_model(modelId = modelId,
                                           body = json.dumps(body), accept='*/*', contentType = 'application/json')
    
    response = json.loads(bedrockResponse['body'].read())['generations'][0]['text']
    apiResponse = {
        'statusCode' : 200,
        'body' : json.dumps({
            'prompt' : prompt,
            'response' : response
        })
    }

    return apiResponse