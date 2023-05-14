import json 

def hallucinateData(data_name): 
    with open(f'data/{data_name}.json', 'r') as f: 
        data = json.load(f) 

    print(len(data)) 
    print(len(data[0]['utterances'])) 

    for conversation in data: 
        original_responses = [] 
        for turn in conversation['utterances']: 
            if turn['VRM'][0] != '': 
                for i in range(len(turn['VRM'])): 
                    if turn['VRM'][i] == ' edification' or turn['VRM'][i] == ' Edification': 
                        turn['VRM'][i] = 'Edification' 
                    elif turn['VRM'][i] == ' Advisement': 
                        turn['VRM'][i] = 'Advisement' 

            original_response = turn['original_response'] 
            turn['response'] = original_response 
            original_responses.append(original_response) 
            for o_index, h_index in enumerate(range(1, len(turn['history']), 2)): 
                turn['history'][h_index] = original_responses[o_index] 

    with open(f'data/hal_{data_name}.json', 'w') as outfile: 
        json_object = json.dumps(data, indent=2) 
        outfile.write(json_object) 

hallucinateData('train')
hallucinateData('valid')
hallucinateData('test')